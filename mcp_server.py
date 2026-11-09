"""MCP server: lets an LLM move the arm, only inside the bounds drawn by hand with cli.py.

The bounds and paths are read from arm_state.json once, at startup; no tool can change them. They are sent to the model
before it does anything (in the server instructions and in each motion tool's description). Moves are capped at
MAX_SPEED (raised to MIN_SPEED when slower) and every target is checked by Arm.go_to (motion.py) before anything is sent.
The serial port opens on the first tool call that needs the arm, which resets the board (~10 s to its boot pose), then
the arm moves to the resting pose START.

    claude mcp add robo-arm -- /path/to/.venv/bin/python /path/to/mcp_server.py
"""

import functools
import math
import sys
import time
from typing import Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from motion import (BASE_KEEPOUT_MM, DEFAULT_SPEED, START, MAX_SPEED, MIN_SPEED, GRIP_CLOSED, GRIP_OPEN, REACH_MAX_MM,
                 REACH_MIN_MM, TOLERANCE_MM, Arm, describe_limits, load_state)
from eyes import snapshot
from gestures import GESTURES, look_at as look_keys


STATE = load_state()
if not STATE["bounds"]:
    sys.exit("no bounds in arm_state.json: run `python cli.py` and draw them first")
BOUNDS, PATHS = STATE["bounds"], STATE["paths"]

LIMITS = describe_limits(BOUNDS)
INSTRUCTIONS = (f"Controls a real 4-DOF desk robot arm (Waveshare RoArm-M2-S) that moves its claw in straight lines. "
                f"{LIMITS} Bounds drawn by a human on {STATE['bounds_drawn']}; you cannot change them. "
                f"Saved movements (play them by name): {', '.join(PATHS) or 'none'}. "
                "It is meant to act like a pet: use `gesture` and `look_at` to react with body language. "
                "Gestures play wherever the claw is and facing, so chain them with look_at and moves: e.g. look_at a "
                "person, curious, turn the other way, nod. "
                f"It starts at the resting pose {START} (go_home returns there). "
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
    if _arm is None:  # not opened yet (a failed look): opening it just for this would reset the arm
        return "unknown"
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


@tool(f"Return to the resting pose: claw folded back over the base at (x, y, z) = {START} mm, gripper shut. The arm "
      "moves there when it connects, and it is the pose to rest in and come back to between things. Straight-line move, "
      "same limits as move_to.")
def go_home(speed: float = DEFAULT_SPEED) -> dict:
    arm().go_to(*START, speed, GRIP_CLOSED)
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


@tool("Body language. " + "; ".join(f"{n}: {m}" for n, (m, _) in GESTURES.items())
      + f". All but sleep and wake_up end where they started. `size` (0.2-1.5, default 1) scales the motion of those;"
      " a gesture that does not fit where the claw is first slides it (facing the same way) toward the middle of the box,"
      " and only then shrinks it, down to 0.35x; the result gives size_used and slid_mm.")
def gesture(name: Literal[tuple(GESTURES)], size: float = 1) -> dict:
    _, used, slid = arm().perform(GESTURES[name][1], size)
    return pose() | {"size_used": round(used, 2), "slid_mm": round(slid)}


@tool("Turn and tilt the camera above the claw to face the point (x, y, z) mm, like looking at it, then perk up "
      "slightly if there is room. It does not reach for it (use move_to for that) and stays facing it. The point itself "
      "may be outside the limits; if the turn or tilt does not fit it goes as far as it can.")
def look_at(x: float, y: float, z: float) -> dict:
    _, used, _ = arm().perform(look_keys(x, y, z))
    return pose() | {"turned_fraction": round(used, 2)}


@tool(f"Play a movement a human taught by hand, by name. Saved: {', '.join(PATHS) or 'none'}. "
      "The claw first goes in a straight line from where it is to the movement's first point.")
def play(name: str, speed: float = DEFAULT_SPEED) -> dict:
    if name not in PATHS:
        raise ValueError(f"no movement '{name}'; saved: {', '.join(PATHS) or 'none'}")
    arm().play(PATHS[name], speed)
    return pose()


@tool("A picture from the camera on the arm's head, just above the claw, looking wherever the claw faces. It does not "
      "move the arm; turn it first (look_at, move_to) to see somewhere else.")
def look() -> Image:
    return Image(data=snapshot(), format="jpeg")


if __name__ == "__main__":
    mcp.run()
