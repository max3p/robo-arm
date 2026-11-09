"""Following a person with the camera in real time (Phase 3.3), without the LLM: visual servoing.

Every frame the eyes give `target`: how far the face (or, with none, the head of a person) is right of and above the
middle of the picture. ~50 times a second the tracker turns the base and tilts the forearm (the camera is on it) at
GAIN degrees a second for every degree it is off, and streams the claw there (Arm.follow). It needs no positions in
the room, only where the target is in the picture, so it is not thrown by bad distances. The arm lags its targets
~0.3 s and the camera ~0.1 s more, which bounds GAIN: about 2 keeps it steady (phase margin ~45 degrees) and still
turns 40 degrees a second for a target 20 degrees off, so someone walking past ~1.5 m away stays in the picture.

When the target is lost it keeps turning the way it went for SEARCH_S (someone standing up leaves out of the top),
then gives up: the body's idling looks for them from there. Every streamed point is checked like any other move
(Arm.follow: the box, the reach), it never turns past BEHIND_DEG, and the claw keeps out of the base's keep-out.

How it tilts: the arm has no wrist, so the camera's tilt is the forearm's (`p`, up from level). The tracker sets `p`
and keeps the upper arm (`q`, up from level) where it was, raising it only as far as the reach limit needs (the two
links at least REACH_GAP_DEG apart, 480 mm). Down to ~-38 and up to ~55 degrees from the resting pose.
"""

import math
import time

from motion import BASE_KEEPOUT_MM, BEHIND_DEG, FOREARM_MM, UPPER_ARM_MM, check_target

GAIN = 2.0            # degrees a second, for each degree the target is off the middle of the picture
DEADBAND_DEG = 2      # nearer the middle than this is centred
MAX_RATE = (60, 40)   # degrees a second: turning, tilting
ACCEL = 240           # degrees a second, per second: how fast the rates change (no jerks)
FRESH_S = 0.25        # a target older than this is not used
SEARCH_S = 1.5        # lost, it keeps turning the way the target went this long
CALM = 5              # degrees a second: slower, the camera's frames are sharp enough to place things by
REACH_GAP_DEG = 46    # the angle between the links at 480 mm reach is 44.8 degrees


def joints(r, z):
    """The upper arm's and the forearm's angles up from level (degrees), elbow up, for the claw r mm out, z up."""
    d, a = math.hypot(r, z), math.atan2(z, r)
    q = a + math.acos(min(1, max(-1, (UPPER_ARM_MM ** 2 + d * d - FOREARM_MM ** 2) / (2 * UPPER_ARM_MM * d))))
    return math.degrees(q), math.degrees(math.atan2(z - UPPER_ARM_MM * math.sin(q), r - UPPER_ARM_MM * math.cos(q)))


def claw(q, p):
    """(r, z) mm of the claw for the links at q and p degrees up from level."""
    q, p = math.radians(q), math.radians(p)
    return (UPPER_ARM_MM * math.cos(q) + FOREARM_MM * math.cos(p),
            UPPER_ARM_MM * math.sin(q) + FOREARM_MM * math.sin(p))


class Tracker:
    def __init__(self, arm, eyes):
        self.arm, self.eyes = arm, eyes
        self.rate = [0.0, 0.0]  # degrees a second now: turning (+ = left), tilting (+ = up)
        self.fb = {}            # where it is, like the arm's feedback, for the eyes to place things by while it tracks

    def calm(self) -> bool:
        return max(map(abs, self.rate)) < CALM

    def run(self, keep) -> str:
        """Follow the target until it has been lost for SEARCH_S, or keep() is False (checked ~50 times a second).
        Raises Interrupted if the arm is told to stop (Arm.abort)."""
        fb = self.arm._feedback()
        here = fb["x"], fb["y"], fb["z"]
        if self.arm.target and math.dist(here, self.arm.target) < 15:  # carry on from the last target, as perform does
            here = self.arm.target
        yaw, (q0, p) = math.degrees(math.atan2(here[1], here[0])), joints(math.hypot(*here[:2]), here[2])
        grip, led = fb["t"], self.arm.led_level
        rate, want, state = self.rate, [0.0, 0.0], {"seen": time.monotonic(), "t": 0.0, "why": "stopped"}
        rate[:] = [0.0, 0.0]

        def place(yaw, p):  # (x, y, z) for the camera at yaw and p, or None outside the limits
            r, z = claw(max(q0, p + REACH_GAP_DEG), p)
            x, y = r * math.cos(math.radians(yaw)), r * math.sin(math.radians(yaw))
            try:
                if r < BASE_KEEPOUT_MM + 10 or abs(yaw) > BEHIND_DEG - 1:
                    raise ValueError
                check_target(self.arm.bounds, x, y, z)
            except ValueError:
                return None
            return x, y, z

        at = place(yaw, p) or here

        def pose(t):
            nonlocal yaw, p, at
            dt, state["t"] = t - state["t"], t
            now = time.monotonic()
            if not keep():
                return None
            target = self.eyes.target
            if target and now - target[0] < FRESH_S:
                state["seen"] = now
                want[:] = [max(-m, min(m, k * GAIN * (abs(e) - DEADBAND_DEG) * (abs(e) > DEADBAND_DEG)
                                       * math.copysign(1, e))) for e, k, m in zip(target[1:3], (-1, 1), MAX_RATE)]
            elif now - state["seen"] > SEARCH_S:
                state["why"] = "lost them"
                return None
            for i in (0, 1):  # toward the rates wanted, no faster than ACCEL
                rate[i] += max(-ACCEL * dt, min(ACCEL * dt, want[i] - rate[i]))
            for i in (0, 1):  # turn and tilt separately: where the box stops one, the other goes on
                yaw2, p2 = (yaw + rate[0] * dt, p) if i == 0 else (yaw, p + rate[1] * dt)
                if (moved := place(yaw2, p2)) is not None:
                    yaw, p, at = yaw2, p2, moved
                else:
                    rate[i] = want[i] = 0
            self.fb = {"x": at[0], "y": at[1], "z": at[2], "b": math.radians(yaw), "s": 0.0, "e": math.radians(90 - p)}
            return (*at, grip, led)

        self.arm.follow(pose)
        rate[:] = [0.0, 0.0]
        return state["why"]
