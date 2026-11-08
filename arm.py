"""Motion API for a Waveshare RoArm-M2-S over USB serial, shared by the human CLI (cli.py) and the AI (mcp_server.py).

This module never prints or prompts: in the MCP server, stdout carries the protocol.
The bounds box and the named paths live in arm_state.json. Only cli.py writes that file (the bounds are drawn by hand);
the MCP server reads it once at startup and cannot change it.

Gestures (gestures.py) go through perform(): keyframes relative to the current pose, flowing without stops, every
20 ms sample checked against the same limits before anything moves.
Every other move goes through go_to(): a straight line, refused before anything is sent if the target is outside the box, out
of reach, or the path passes too close to the base. The box is convex, so a straight line between two points inside it
stays inside it. The box limits the claw TIP only, not the elbow or forearm: draw it smaller than the real free space.
Opening the port resets the board, and the arm then moves to its start pose (~10 s), which may be outside the box.

Protocol (Waveshare wiki), 115200 baud, one JSON object per line, angles in radians:
    {"T":105}                              request feedback; reply {"T":1051,"x","y","z","b","s","e","t",...}
    {"T":1041,"x":..,"y":..,"z":..,"t":..} go to XYZ (mm), direct and unsmoothed; t = gripper angle
    {"T":106,"cmd":t,"spd","acc"}          gripper only
    {"T":122,"b","s","e","h","spd","acc"}  joint angles in DEGREES; its spd/acc stay set for later T:1041 moves
    {"T":102,"base","shoulder","elbow","hand","spd","acc"}   go to joint angles (spd 0 = fastest)
    {"T":210,"cmd":0|1}                    joints free (move by hand) / locked
    {"T":114,"led":0..255}                 gripper LED brightness
"""

import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

STATE_FILE = Path(__file__).with_name("arm_state.json")
RATE_HZ = 50        # every move is streamed as a new target this often, so a stop takes effect within 20 ms
# Speeds measured on the arm (2026-11-07, 370 mm moves): smooth and on the line from 100 to 500 mm/s. Above ~600 the
# servos saturate (800 asked gave ~700, 26 mm off the line). Below 100 the 50 Hz micro-steps are too small and the
# motion turns stop-go jerky, so slower requests are raised to MIN_SPEED.
MAX_SPEED = 500      # mm/s hard cap on the PEAK speed of any move, whatever the caller asks for
MIN_SPEED = 100      # mm/s
DEFAULT_SPEED = 300  # mm/s
TOLERANCE_MM = 15   # a move that ends farther than this from the target (after the sag correction) is reported as failed
# Under its own weight the arm stops short, worst moving UP to high, stretched poses (10-20 mm low at home). go_to then
# aims past the target by the measured error (up to 2 passes); the aim point may be up to SAG_MAX_MM outside the box,
# the claw is not. Sometimes (after boot, or with warm servos?) it stalls 25-35 mm low; the capped push still helps then.
SAG_MIN_MM, SAG_MAX_MM = 3, 25
# The firmware does not guard T:1041 against targets the arm cannot reach (its IK yields NaN, which becomes a garbage
# servo goal), so go_to refuses them first. Targets are measured from the shoulder pivot (the x, y, z origin); the arm's
# two links are ~239 and ~280 mm, so it reaches 41..519 mm. These limits keep well inside that. Derived from firmware constants.
REACH_MIN_MM, REACH_MAX_MM = 100, 480
BASE_KEEPOUT_MM = 100  # the claw must stay this far (in x, y) from the base's vertical axis: nearer is a folded arm that may hit itself
XYZ_SERVO_SPEED_DEG = 360  # deg/s servo speed limit set before xyz moves: headroom above MAX_SPEED, not a target speed
XYZ_SERVO_ACC_DEG = 20    # servo acceleration limit for xyz moves; 20 lags the stream least at speed (2 lagged ~25% more)
JOINT_ACC = 10            # servo-side acceleration limit for lock()
GRIP_CLOSED, GRIP_OPEN = 3.14, 1.08  # gripper angle t (radians), wiki default clamp range; check on the arm
HOME = (309.0, 4.0, 228.0)  # mm: the pose the arm boots to (measured over 4 boots, 2026-11-07), gripper shut
GRIP_SPEED, GRIP_ACC = 8.0, 50  # rad/s peak and servo acceleration: full range ~1.6 s; the servo itself caps it there


def load_state() -> dict:
    """{"bounds": {"x": [lo, hi], "y": .., "z": ..} or None, "bounds_drawn": iso time, "paths": {name: [[x, y, z, t], ..]}}"""
    try:
        state = json.loads(STATE_FILE.read_text())
    except FileNotFoundError:
        state = {}
    return {"bounds": None, "bounds_drawn": None, "paths": {}} | state


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATE_FILE)  # atomic: a crash mid-write cannot leave a half-written bounds file


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def check_target(bounds: dict, x, y, z) -> None:
    """Raise ValueError unless (x, y, z) is numbers inside the box and the arm's safe reach."""
    for axis, v in dict(x=x, y=y, z=z).items():
        lo, hi = bounds[axis]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:  # also rejects NaN
            raise ValueError(f"{axis} must be a number in [{lo:g}, {hi:g}], got {round(v, 1) if isinstance(v, float) else v!r}")
    if not REACH_MIN_MM <= math.dist((0, 0, 0), (x, y, z)) <= REACH_MAX_MM:
        raise ValueError(f"({x:g}, {y:g}, {z:g}) is outside the arm's safe reach "
                         f"({REACH_MIN_MM}-{REACH_MAX_MM} mm from the shoulder pivot)")


def _cart(p):
    """(r mm from the base axis, a degrees of base turn, z mm, gripper opening 0..1, led) -> (x, y, z, t, led)."""
    a = math.radians(p[1])
    return p[0] * math.cos(a), p[0] * math.sin(a), p[2], GRIP_CLOSED - p[3] * (GRIP_CLOSED - GRIP_OPEN), p[4]


def trajectory(bounds: dict, start, keys, size: float, shift=(0, 0)) -> list:
    """The 50 Hz samples (x, y, z, t, led) of a gesture from `start` (r, a, z, opening, led), eased into and out of each
    keyframe. A nonzero `shift` (dr, dz) first moves the claw there (same direction) and plays the gesture around that.
    Interpolated in cylindrical coordinates, so a change of `a` is a turn of the base, not a slide. Segments
    are stretched so the peak speed stays under MAX_SPEED. Raises ValueError if any sample leaves the bounds or reach,
    or comes nearer the base axis than the keep-out (or than the start, if that is nearer already)."""
    base = (start[0] + shift[0], start[1], start[2] + shift[1])
    lead = [(0.6, *base, start[3], start[4])] if any(shift) else []
    prev, out = start, []
    for secs, *frame in lead + [(secs, base[0] + dr * size, base[1] + da * size, base[2] + dz * size,
                                 start[3] if grip is None else grip, start[4] if led is None else led)
                                for secs, dr, da, dz, grip, led in keys]:
        seg = lambda n: [_cart([a + (b - a) * u * u * (3 - 2 * u) for a, b in zip(prev, frame)])
                         for u in (j / n for j in range(1, n + 1))]
        coarse = [_cart(prev)] + seg(20)
        length = sum(math.dist(p[:3], q[:3]) for p, q in zip(coarse, coarse[1:]))
        out += seg(max(1, round(max(secs, length * 1.5 / MAX_SPEED) * RATE_HZ)))
        prev = frame
    for x, y, z, _, _ in out:
        check_target(bounds, x, y, z)
        if math.hypot(x, y) < min(BASE_KEEPOUT_MM, start[0] - 0.5):
            raise ValueError(f"it would pass {math.hypot(x, y):.0f} mm from the base axis (keep-out {BASE_KEEPOUT_MM} mm)")
    return out


def find_port() -> str:
    ports = [p.device for p in list_ports.comports() if p.vid]  # USB serial devices only
    if len(ports) != 1:
        raise RuntimeError(f"cannot pick the arm's port (found {ports or 'none'}); pass it as Arm(bounds, port)")
    return ports[0]


class Arm:
    def __init__(self, bounds: dict | None, port: str | None = None):
        self.target = None  # last (x, y, z) go_to aimed for: move_by builds on it, so sag does not add up
        self.commanded = None  # last (x, y, z) actually sent: the target plus the sag correction
        self.led_level = 0  # the arm does not report its LED; it boots dark
        self.bounds = bounds  # {"x": (lo, hi), "y": ..., "z": ...} in mm; every move is refused while this is None
        self.lock_ = threading.RLock()  # one physical action at a time (MCP tools run in worker threads)
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port or find_port(), 115200, 0.2
        self.ser.dtr = self.ser.rts = False  # set before open(): asserting them holds the ESP32 in reset
        self.ser.open()
        deadline = time.monotonic() + 25  # the arm moves to its start pose while it boots
        while True:
            try:
                self._feedback()
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    self.ser.close()
                    raise RuntimeError("arm never answered; is the 12 V adapter plugged in and switched on?")

    def _send(self, cmd: dict) -> None:
        self.ser.write((json.dumps(cmd) + "\n").encode())

    def _feedback(self) -> dict:
        with self.lock_:
            self.ser.reset_input_buffer()
            self.ser.write(b'{"T":105}\n')
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    msg = json.loads(self.ser.readline())
                except ValueError:  # boot text, echoes, partial lines, read timeouts
                    continue
                if isinstance(msg, dict) and msg.get("T") == 1051:
                    return msg
        raise RuntimeError("no feedback")

    def read(self) -> tuple[float, float, float, float]:
        """Measured (x, y, z, t): claw position in mm and gripper angle in radians."""
        fb = self._feedback()
        return fb["x"], fb["y"], fb["z"], fb["t"]

    def joints(self) -> tuple[float, float, float, float]:
        fb = self._feedback()
        return fb["b"], fb["s"], fb["e"], fb["t"]

    def _glide(self, start, goal, duration, make_cmd) -> None:
        """Stream targets from start to goal in a straight line, speeding up and slowing down smoothly (smoothstep: speed
        0 at both ends, peak 1.5x the average)."""
        t0 = time.monotonic()  # positions come from the clock, not from summed sleeps, so timing jitter cannot cause jerks
        while True:
            u = min((time.monotonic() - t0) / duration, 1.0)
            f = u * u * (3 - 2 * u)
            self._send(make_cmd([a + (b - a) * f for a, b in zip(start, goal)]))
            if u >= 1.0:
                return
            time.sleep(1 / RATE_HZ)

    def _settle(self, read, eps):
        """The firmware has no 'move finished' signal: settled = two consecutive reads agree to within eps."""
        last = now = read()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            time.sleep(0.1)
            now = read()
            if math.dist(now, last) < eps:
                break
            last = now
        return now

    def _xyz_mode(self, fb) -> None:
        """T:1041 takes its servo speed and acceleration from firmware globals, and every T:102 (lock) resets them to
        0 = unlimited, which turns 50 Hz micro-steps into stop-go jerks. T:122 sets them and leaves them set, so send it
        before streaming, with the current angles (no motion)."""
        b, sh, e, h = (math.degrees(fb[k]) for k in "bset")
        self._send({"T": 122, "b": b, "s": sh, "e": e, "h": h, "spd": XYZ_SERVO_SPEED_DEG, "acc": XYZ_SERVO_ACC_DEG})

    def _correct(self, goal, aim, t, now):
        """Sag correction: `aim` was last sent and the claw settled at `now`. Aim past `goal` by what it missed, up to 2
        passes; the push past `goal` never exceeds SAG_MAX_MM. Returns the measured position."""
        for _ in range(2):
            if math.dist(now, goal) <= SAG_MIN_MM:
                break
            push = [a - n for a, n in zip(aim, now)]  # (aim - goal) + (goal - now): the old push plus the new miss
            k = min(1, SAG_MAX_MM / math.hypot(*push))
            new = [g + p * k for g, p in zip(goal, push)]
            if math.dist((0, 0, 0), new) > REACH_MAX_MM + SAG_MAX_MM:
                break
            self._glide((*aim, t), (*new, t), 0.3, lambda p: {"T": 1041, "x": round(p[0], 1), "y": round(p[1], 1),
                                                              "z": round(p[2], 1), "t": round(p[3], 3)})
            aim, now = new, self._settle(lambda: self.read()[:3], 0.5)
        self.commanded = tuple(aim)
        return now

    def go_to(self, x: float, y: float, z: float, speed: float = DEFAULT_SPEED, t: float | None = None):
        """Move the claw to (x, y, z) mm in a straight line at `speed` mm/s (peak), and the gripper to angle t (None = keep
        it). Returns the measured final (x, y, z). Raises ValueError, before anything moves, if the target is outside the
        bounds or reach or the path passes too near the base; RuntimeError if the arm does not get there."""
        if not self.bounds:
            raise RuntimeError("no bounds: draw them first with cli.py")
        check_target(self.bounds, x, y, z)
        if t is not None and not GRIP_OPEN <= t <= GRIP_CLOSED:
            raise ValueError(f"gripper angle {t!r} is not in [{GRIP_OPEN}, {GRIP_CLOSED}]")
        speed = min(max(speed, MIN_SPEED), MAX_SPEED)
        with self.lock_:
            fb = self._feedback()
            start, goal = (fb["x"], fb["y"], fb["z"]), (x, y, z)
            d = [g - a for a, g in zip(start[:2], goal[:2])]  # closest approach of the straight path to the base axis, in plan view
            f = min(max(-sum(a * v for a, v in zip(start, d)) / (sum(v * v for v in d) or 1), 0), 1)
            closest = math.hypot(*(a + f * v for a, v in zip(start, d)))
            if closest < BASE_KEEPOUT_MM and closest < math.hypot(*start[:2]) - 0.5:  # moving away from the base is always allowed
                raise ValueError(f"the straight path to ({x:g}, {y:g}, {z:g}) passes {closest:.0f} mm from the base axis "
                                 f"(keep-out {BASE_KEEPOUT_MM} mm); choose a target that avoids it")
            self._xyz_mode(fb)
            t = fb["t"] if t is None else t
            self._glide((*start, fb["t"]), (*goal, t), max(math.dist(start, goal) * 1.5 / speed, 0.2),
                        lambda p: {"T": 1041, "x": round(p[0], 1), "y": round(p[1], 1), "z": round(p[2], 1),
                                   "t": round(p[3], 3)})
            self.target = self.commanded = goal
            now = self._settle(lambda: self.read()[:3], 0.5)
            now = self._correct(goal, goal, t, now)
        error = math.dist(now, goal)
        if error > TOLERANCE_MM:
            raise RuntimeError(f"arm ended {error:.0f} mm from the target, at ({now[0]:.0f}, {now[1]:.0f}, {now[2]:.0f}): "
                               "sagging under load, or blocked. It is safe where it is")
        return now

    def gripper(self, t: float) -> float:
        """Move only the gripper to angle t (GRIP_CLOSED = shut, GRIP_OPEN = wide). Returns the measured angle, which
        stops short of t when the gripper closes on something."""
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not GRIP_OPEN <= t <= GRIP_CLOSED:
            raise ValueError(f"gripper angle {t!r} is not in [{GRIP_OPEN}, {GRIP_CLOSED}]")
        with self.lock_:  # streamed like go_to: a single T:106 starts late, and _settle mistook that for done
            t0 = self.read()[3]
            self._glide((t0,), (t,), max(abs(t - t0) * 1.5 / GRIP_SPEED, 0.2),
                        lambda p: {"T": 106, "cmd": round(p[0], 3), "spd": 0, "acc": GRIP_ACC})
            return self._settle(lambda: self.read()[3:], 0.01)[0]

    def play(self, points: list, speed: float = DEFAULT_SPEED) -> None:
        """Visit [x, y, z, t] points in order with go_to (straight lines, gripper interpolated). Every point is checked
        against the bounds before anything moves, so a path recorded under older, bigger bounds is refused whole."""
        if not points:
            raise ValueError("the path is empty")
        for i, (x, y, z, _) in enumerate(points, 1):
            try:
                check_target(self.bounds or {}, x, y, z)
            except (ValueError, KeyError) as e:
                raise ValueError(f"point {i} of the path is not allowed: {e or 'no bounds'}") from None
        with self.lock_:
            for x, y, z, t in points:
                self.go_to(x, y, z, speed, min(max(t, GRIP_OPEN), GRIP_CLOSED))

    def perform(self, keys, size: float = 1.0):
        """Play a gesture: keyframes (secs, dr, da, dz, grip, led) relative to the claw's pose now: dr mm out from the
        base axis, da degrees of base turn (+ = left), dz mm up, all times `size`; grip (0 shut .. 1 wide) and led
        (0..255) are absolute, None = as at the start. `keys` may instead be a function (ref (x, y, z), bounds) ->
        keyframes, for gestures aimed at a place; those ignore `size`. If the path does not fit inside the limits, the
        claw first slides (facing the same way) up to all the way to the middle of the box's x range and height, and
        plays it there; only if that does not fit either is it shrunk (0.7, 0.5, 0.35 of the size). Unlike play, it flows
        through the keyframes without settling at each, and every sample is checked before anything moves. It ends with
        go_to's sag correction. Returns (measured (x, y, z), size used, mm slid first)."""
        if not self.bounds:
            raise RuntimeError("no bounds: draw them first with cli.py")
        if isinstance(size, bool) or not isinstance(size, (int, float)) or not 0.2 <= size <= 1.5:
            raise ValueError(f"size {size!r} is not in [0.2, 1.5]")
        with self.lock_:
            fb = self._feedback()
            here = fb["x"], fb["y"], fb["z"]
            if self.target and math.dist(here, self.target) <= TOLERANCE_MM:  # keep go_to's sag correction going
                ref, off = self.target, [c - g for c, g in zip(self.commanded, self.target)]
            else:
                ref, off = here, (0, 0, 0)
            slides = (0, .25, .5, .75, 1)
            if callable(keys):
                keys, size, slides = keys(ref, self.bounds), 1.0, (0,)  # aimed gestures go where they are aimed
            opening = min(max((GRIP_CLOSED - fb["t"]) / (GRIP_CLOSED - GRIP_OPEN), 0), 1)
            start = (math.hypot(*ref[:2]), math.degrees(math.atan2(ref[1], ref[0])), ref[2], opening, self.led_level)
            mid = (sum(self.bounds["x"]) / 2 - start[0], sum(self.bounds["z"]) / 2 - start[2])
            error = None
            for k, f in ((k, f) for k in (1, 0.7, 0.5, 0.35) for f in slides):
                try:
                    samples = trajectory(self.bounds, start, keys, size * k, (mid[0] * f, mid[1] * f))
                    break
                except ValueError as e:
                    error = error or e
            else:
                raise ValueError(f"the gesture does not fit here, even moved toward the middle of the box and shrunk to "
                                 f"{size * k:.2g}x: {error}")
            self._xyz_mode(fb)
            t0, led = time.monotonic(), round(self.led_level)
            for i, (x, y, z, t, level) in enumerate(samples):
                time.sleep(max(0, t0 + i / RATE_HZ - time.monotonic()))  # clock-timed, like _glide
                self._send({"T": 1041, "x": round(x + off[0], 1), "y": round(y + off[1], 1), "z": round(z + off[2], 1),
                            "t": round(t, 3)})
                if round(level) != led:
                    led = round(level)
                    self._send({"T": 114, "led": led})
            self.led_level = led
            self.target = samples[-1][:3]
            aim = tuple(a + o for a, o in zip(self.target, off))
            now = self._correct(self.target, aim, samples[-1][3], self._settle(lambda: self.read()[:3], 0.5))
            return now, size * k, math.hypot(mid[0] * f, mid[1] * f)

    def led(self, level: int) -> None:
        """Gripper LED brightness, 0 (off) .. 255 (full). Tested on the arm 2026-11-07. The arm does not report it back."""
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 255:
            raise ValueError(f"LED level {level!r} is not an integer in [0, 255]")
        with self.lock_:
            self._send({"T": 114, "led": level})
            self.led_level = level

    def free(self) -> None:
        """Torque off: the joints go limp so the arm can be moved by hand. Human use only."""
        self._send({"T": 210, "cmd": 0})
        self.target = None

    def lock(self) -> None:
        """Lock the joints where they are now. A rotation command re-locks them, so command the measured angles."""
        with self.lock_:
            b, s, e, h = self.joints()
            self.target = None
            self._send({"T": 102, "base": b, "shoulder": s, "elbow": e, "hand": h, "spd": 0, "acc": JOINT_ACC})
            time.sleep(0.2)
            fb = self._feedback()
            if not all(fb[k] for k in ("torswitchB", "torswitchS", "torswitchE", "torswitchH")):
                self._send({"T": 210, "cmd": 1})  # fallback: plain torque on

    def close(self) -> None:
        self.ser.close()
