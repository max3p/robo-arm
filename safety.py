"""Deterministic checks between the LLM and the arm. Rejects bad targets instead of clamping them,
so the LLM gets a clear reason and can choose a valid one. Post-move verification is in Arm.move_to."""

import math

from arm import Pose


class Rejected(ValueError):
    pass


def check_target(values: dict, bounds: dict) -> Pose:
    for axis in "xyz":
        v = values.get(axis)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise Rejected(f"{axis}={v!r} is not a finite number")
        lo, hi = bounds[axis]
        if not lo <= v <= hi:
            raise Rejected(f"{axis}={v:g} is outside allowed range [{lo}, {hi}]")
    return Pose(values["x"], values["y"], values["z"])


def waypoints(start: Pose, end: Pose, max_segment_mm: float) -> list[Pose]:
    """Split a move into segments of at most max_segment_mm, ending exactly at `end`."""
    n = max(1, math.ceil(start.dist(end) / max_segment_mm))
    return [start.lerp(end, i / n) for i in range(1, n + 1)]


def cap_speed(speed: float, max_speed: float) -> float:
    return min(max(speed, 1.0), max_speed)
