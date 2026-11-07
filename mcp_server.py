"""MCP server: lets an LLM move the arm, only inside the bounds drawn by hand with cli.py.

The bounds and paths are read from arm_state.json once, at startup; no tool can change them. They are sent to the model
before it does anything (in the server instructions and in each motion tool's description). Moves are capped at
MAX_SPEED (raised to MIN_SPEED when slower) and every target is checked by arm.go_to before anything is sent.
The serial port opens on the first tool call that needs the arm, which resets the board (~10 s to its start pose).

    claude mcp add robo-arm -- /path/to/.venv/bin/python /path/to/mcp_server.py
"""

import functools
import math
import sys
import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from arm import (BASE_KEEPOUT_MM, DEFAULT_SPEED, HOME, MAX_SPEED, MIN_SPEED, GRIP_CLOSED, GRIP_OPEN, REACH_MAX_MM,
                 REACH_MIN_MM, TOLERANCE_MM, Arm, load_state)


STATE = load_state()
if not STATE["bounds"]:
    sys.exit("no bounds in arm_state.json: run `python cli.py` and draw them first")
BOUNDS, PATHS = STATE["bounds"], STATE["paths"]

LIMITS = (f"Hard limits (mm; origin at the shoulder pivot; x forward, y left, z up): "
          + ", ".join(f"{a} {lo:g} to {hi:g}" for a, (lo, hi) in BOUNDS.items())
          + f". The claw must also stay {REACH_MIN_MM}-{REACH_MAX_MM} mm from the origin, and a straight path may not pass"
          f" within {BASE_KEEPOUT_MM} mm of the base's vertical axis (x=y=0). Targets outside are refused before the arm"
          f" moves. Speed (peak, mm/s) is {MIN_SPEED}-{MAX_SPEED}, default {DEFAULT_SPEED}; moves are smooth and eased at both ends.")
INSTRUCTIONS = (f"Controls a real 4-DOF desk robot arm (Waveshare RoArm-M2-S) that moves its claw in straight lines. "
                f"{LIMITS} Bounds drawn by a human on {STATE['bounds_drawn']}; you cannot change them. "
                f"Saved movements (play them by name): {', '.join(PATHS) or 'none'}. "
                "Every result reports the measured pose. When a move is refused, pick a target inside the limits "
                "instead of repeating it. Moves block until the arm has settled.")

mcp = MCPServer("robo-arm", instructions=INSTRUCTIONS)
_arm: Arm | None = None


def arm() -> Arm:
    global _arm
    if _arm is None:
        _arm = Arm(BOUNDS)
    return _arm


def pose() -> dict:
    x, y, z, t = arm().read()
    return {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1), "gripper_open": round(to_opening(t), 2)}


def to_opening(t: float) -> float:  # gripper angle -> 0 (shut) .. 1 (wide)
    return (GRIP_CLOSED - t) / (GRIP_CLOSED - GRIP_OPEN)


def tool(description: str):
    """Register a tool whose refusals and failures reach the model as readable errors."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (ValueError, RuntimeError, OSError) as e:
                raise ToolError(f"{e}. Current pose: {pose_or_unknown()}") from e
        return mcp.tool(description=description)(wrapper)
    return deco


def pose_or_unknown():
    try:
        return pose()
    except Exception:
        return "unknown"


@tool(f"The arm's limits, saved movements and current pose. {LIMITS}")
def workspace() -> dict:
    return {"bounds_mm": BOUNDS, "bounds_drawn": STATE["bounds_drawn"], "reach_mm": [REACH_MIN_MM, REACH_MAX_MM],
            "base_keepout_mm": BASE_KEEPOUT_MM, "speed_mm_s": [MIN_SPEED, MAX_SPEED],
            "movements": {name: len(points) for name, points in PATHS.items()}, "pose": pose()}


@tool("Measured claw position (mm) and gripper opening (0 shut .. 1 wide).")
def get_pose() -> dict:
    return pose()


@tool(f"Move the claw in a straight line to (x, y, z) mm at `speed` mm/s. {LIMITS}")
def move_to(x: float, y: float, z: float, speed: float = DEFAULT_SPEED) -> dict:
    arm().go_to(x, y, z, speed)
    return pose()


@tool(f"Move the claw by (dx, dy, dz) mm in a straight line, from the last target (or from where it is, if it is "
      f"not near the last target). {LIMITS}")
def move_by(dx: float = 0, dy: float = 0, dz: float = 0, speed: float = DEFAULT_SPEED) -> dict:
    here, last = arm().read()[:3], arm().target
    x, y, z = last if last and math.dist(here, last) <= TOLERANCE_MM else here  # building on the target: sag cannot add up
    arm().go_to(x + dx, y + dy, z + dz, speed)
    return pose()


@tool(f"Return to the starting pose the arm boots to: claw at (x, y, z) = {HOME} mm, gripper shut. "
      "Straight-line move, same limits as move_to.")
def go_home(speed: float = DEFAULT_SPEED) -> dict:
    arm().go_to(*HOME, speed, GRIP_CLOSED)
    return pose()


@tool("Set the LED on the gripper: brightness 0 = off .. 1 = full. With blinks > 0 it first flashes that many times "
      "(0.3 s on, 0.3 s off), then stays at `brightness`. White only. The arm does not report the LED, so the result echoes it.")
def light(brightness: float, blinks: int = 0) -> dict:
    if not 0 <= brightness <= 1 or not 0 <= blinks <= 20:
        raise ValueError(f"brightness must be in [0, 1] and blinks in [0, 20], got {brightness!r}, {blinks!r}")
    for _ in range(blinks):
        arm().led(255); time.sleep(0.3)
        arm().led(0); time.sleep(0.3)
    arm().led(round(brightness * 255))
    return {"brightness": brightness}


@tool("Set the gripper opening: 0 = shut, 1 = wide open. If it closes on an object it stops short; "
      "the result's gripper_open shows where it ended.")
def gripper(opening: float) -> dict:
    if not 0 <= opening <= 1:
        raise ValueError(f"opening {opening!r} is not in [0, 1]")
    arm().gripper(GRIP_CLOSED - opening * (GRIP_CLOSED - GRIP_OPEN))
    return pose()


@tool(f"Play a movement a human taught by hand, by name. Saved: {', '.join(PATHS) or 'none'}. "
      "The claw first goes in a straight line from where it is to the movement's first point.")
def play(name: str, speed: float = DEFAULT_SPEED) -> dict:
    if name not in PATHS:
        raise ValueError(f"no movement '{name}'; saved: {', '.join(PATHS) or 'none'}")
    arm().play(PATHS[name], speed)
    return pose()


if __name__ == "__main__":
    mcp.run()
