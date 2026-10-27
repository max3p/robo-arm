"""Arm interface, a console mock, and the real Waveshare RoArm-M2-S over USB serial.

RoArm-M2-S protocol (Waveshare wiki): 115200 baud, one JSON object per line.
    {"T":1041,"x":..,"y":..,"z":..,"t":..}   go to XYZ (mm) + gripper angle t (rad), direct, non-blocking
    {"T":106,"cmd":3.14,"spd":0,"acc":0}       gripper angle (3.14 closed, ~1.08 fully open)
    {"T":105}                                  request feedback; reply {"T":1051,"x","y","z","b","s","e","t","v",..}
    {"T":210,"cmd":0|1}                        torque off / on
The gripper is the 4th joint, so every XYZ command carries the gripper angle or the gripper moves too.
"""

import json
import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
STEP_MM = 3  # RoArmSerial streams moves as small T:1041 steps: the firmware does not interpolate them


def load_config(path=CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float

    def dist(self, other: "Pose") -> float:
        return math.dist(vars(self).values(), vars(other).values())

    def lerp(self, other: "Pose", f: float) -> "Pose":
        return Pose(*(a + (b - a) * f for a, b in zip(vars(self).values(), vars(other).values())))

    def as_dict(self) -> dict:
        return {k: round(v, 1) for k, v in vars(self).items()}


@dataclass
class MoveResult:
    ok: bool
    final_pose: Pose
    target: Pose
    error_mm: float
    message: str = ""


class Arm(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.stopped = False

    @abstractmethod
    def get_pose(self) -> Pose: ...

    @abstractmethod
    def _go(self, target: Pose, speed: float) -> Pose:
        """Move to target, wait until settled, return the measured pose."""

    @abstractmethod
    def set_gripper(self, closed: bool) -> float:
        """Returns the measured gripper angle."""

    @abstractmethod
    def torque(self, enabled: bool) -> None: ...

    def move_to(self, target: Pose, speed: float) -> MoveResult:
        if self.stopped:
            raise RuntimeError("arm is stopped; restart the program to move again")
        final = self._go(target, speed)
        error = final.dist(target)
        ok = error <= self.cfg["tolerance_mm"]
        return MoveResult(ok, final, target, round(error, 1),
                          "" if ok else f"arm ended {error:.0f} mm from the target (unreachable or blocked?)")

    def stop(self) -> None:
        """Emergency stop: refuse further motion, hold position with torque on."""
        self.stopped = True
        self.torque(True)

    def close(self) -> None:
        pass


class MockArm(Arm):
    """Logs commands to the console. drift_mm offsets every final pose along x; targets farther
    than reach_mm from the base stop short. Both exist to exercise the error paths."""

    def __init__(self, cfg: dict, drift_mm: float = 0.0, reach_mm: float | None = None):
        super().__init__(cfg)
        self.pose = Pose(*cfg["home"])
        self.drift_mm, self.reach_mm = drift_mm, reach_mm

    def get_pose(self) -> Pose:
        return self.pose

    def _go(self, target, speed):
        print(f"[MOCK ARM] move_to x={target.x:.1f} y={target.y:.1f} z={target.z:.1f} speed={speed:.0f}")
        reach = math.dist((0, 0, 0), vars(target).values())
        f = min(1.0, self.reach_mm / reach) if self.reach_mm else 1.0
        self.pose = Pose(target.x * f + self.drift_mm, target.y * f, target.z * f)
        return self.pose

    def set_gripper(self, closed):
        print(f"[MOCK ARM] gripper {'close' if closed else 'open'}")
        return self.cfg["gripper"]["closed" if closed else "open"]

    def torque(self, enabled):
        print(f"[MOCK ARM] torque {'on' if enabled else 'off'}")


def find_port() -> str:
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if p.vid]  # USB devices only
    if len(ports) != 1:
        raise RuntimeError(f"cannot pick the arm's serial port (found {ports or 'none'}); pass --port")
    return ports[0]


class RoArmSerial(Arm):
    def __init__(self, cfg: dict, port: str | None = None, ser=None):
        super().__init__(cfg)
        self.lock = threading.Lock()
        if ser is None:
            import serial
            ser = serial.Serial()
            ser.port, ser.baudrate, ser.timeout = port or find_port(), 115200, 0.2
            ser.dtr = ser.rts = False  # set before open: asserting them holds the ESP32 in reset
            ser.open()
            time.sleep(2)  # let the firmware finish booting
        self.ser = ser
        fb = self.feedback()
        self.t = fb["t"]  # commanded gripper angle, sent with every move
        print(f"[ROARM] {ser.port}: x={fb['x']:.0f} y={fb['y']:.0f} z={fb['z']:.0f} "
              f"gripper={fb['t']:.2f} supply={fb.get('v', 0) / 100:.1f} V")

    def _send(self, cmd: dict) -> None:
        self.ser.write((json.dumps(cmd) + "\n").encode())

    def feedback(self) -> dict:
        with self.lock:
            self.ser.reset_input_buffer()
            self._send({"T": 105})
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    msg = json.loads(self.ser.readline())
                except ValueError:  # boot messages, echoes, partial lines, read timeouts
                    continue
                if isinstance(msg, dict) and msg.get("T") == 1051:
                    return msg
        raise RuntimeError("no T:1051 feedback from the arm")

    def get_pose(self) -> Pose:
        fb = self.feedback()
        return Pose(fb["x"], fb["y"], fb["z"])

    def _go(self, target, speed):
        start = self.get_pose()
        n = max(1, math.ceil(start.dist(target) / STEP_MM))
        for i in range(1, n + 1):
            p = start.lerp(target, i / n)
            with self.lock:
                self._send({"T": 1041, "x": round(p.x, 1), "y": round(p.y, 1), "z": round(p.z, 1), "t": self.t})
            time.sleep(start.dist(target) / n / speed)
        # The firmware reports no "move finished"; settled = two consecutive reads agree.
        last, deadline = self.get_pose(), time.monotonic() + self.cfg["settle_timeout_s"]
        while time.monotonic() < deadline:
            time.sleep(0.1)
            pose = self.get_pose()
            if pose.dist(last) < 0.5:
                break
            last = pose
        return pose

    def set_gripper(self, closed):
        # Keep the commanded angle for later moves: re-sending the measured one would loosen a grip.
        self.t = self.cfg["gripper"]["closed" if closed else "open"]
        with self.lock:
            self._send({"T": 106, "cmd": self.t, "spd": 0, "acc": 0})
        time.sleep(1.0)
        return self.feedback()["t"]

    def torque(self, enabled):
        with self.lock:
            self._send({"T": 210, "cmd": int(enabled)})

    def close(self):
        self.ser.close()
