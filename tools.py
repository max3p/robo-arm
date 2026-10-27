"""Tools the LLM can call: thin wrappers over safety + arm. Each returns a small JSON-able dict
describing the observed outcome, and every call is logged as one JSONL line."""

import json
import threading
import time

from safety import Rejected, cap_speed, check_target, waypoints

NUM = {"type": "number"}


def _schema(name, description, **props):
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


SCHEMAS = [
    _schema("move_to", "Move the gripper tip to absolute x, y, z (mm). Returns the measured final pose.",
            x=NUM, y=NUM, z=NUM),
    _schema("move_by", "Move the gripper tip relative to its current pose by dx, dy, dz (mm).",
            dx=NUM, dy=NUM, dz=NUM),
    _schema("get_pose", "Return the current measured pose."),
    _schema("gripper", "Open or close the gripper.", action={"type": "string", "enum": ["open", "close"]}),
    _schema("go_home", "Move to the home pose."),
    _schema("get_bounds", "Return the box (mm) the gripper tip must stay inside."),
    _schema("done", "End the task, with the reason: goal achieved, or why it cannot be.",
            reason={"type": "string"}),
]
TOOL_NAMES = {s["name"] for s in SCHEMAS}


class Tools:
    def __init__(self, arm, cfg: dict, log_path=None, speed: float | None = None):
        self.arm, self.cfg = arm, cfg
        self.speed = cap_speed(speed or cfg["speed"], cfg["max_speed"])
        self.lock = threading.Lock()  # one physical action at a time
        self.log_path = log_path

    def call(self, name: str, args: dict) -> dict:
        t0 = time.monotonic()
        with self.lock:
            try:
                if name not in TOOL_NAMES:
                    raise Rejected(f"unknown tool {name!r}")
                result = getattr(self, name)(**args)
            except Rejected as e:
                result = {"status": "rejected", "reason": str(e)}
            except TypeError as e:  # wrong or missing arguments
                result = {"status": "rejected", "reason": f"bad arguments: {e}"}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"time": time.time(), "tool": name, "args": args, "result": result,
                                    "ms": round((time.monotonic() - t0) * 1000)}, default=str) + "\n")
        return result

    def _move(self, values: dict) -> dict:
        # Waypoints lie on the straight line to a checked target inside a box, so they are inside too.
        target = check_target(values, self.cfg["bounds"])
        for wp in waypoints(self.arm.get_pose(), target, self.cfg["max_segment_mm"]):
            r = self.arm.move_to(wp, self.speed)
            if not r.ok:  # stop the sequence here
                return {"status": "error", "reason": r.message, "target": target.as_dict(),
                        "final_pose": r.final_pose.as_dict(), "error_mm": r.error_mm}
        return {"status": "ok", "final_pose": r.final_pose.as_dict(), "error_mm": r.error_mm}

    def move_to(self, x, y, z):
        return self._move({"x": x, "y": y, "z": z})

    def move_by(self, dx, dy, dz):
        p = self.arm.get_pose()
        return self._move({"x": p.x + dx, "y": p.y + dy, "z": p.z + dz})

    def get_pose(self):
        return {"status": "ok", "pose": self.arm.get_pose().as_dict()}

    def gripper(self, action):
        if action not in ("open", "close"):
            raise Rejected(f"action must be 'open' or 'close', not {action!r}")
        angle = self.arm.set_gripper(action == "close")
        return {"status": "ok", "gripper": action, "measured_angle": round(angle, 2)}

    def go_home(self):
        return self._move(dict(zip("xyz", self.cfg["home"])))

    def get_bounds(self):
        return {"status": "ok", "bounds": self.cfg["bounds"], "frame": "mm; x forward, y left, z up"}

    def done(self, reason):
        return {"status": "done", "reason": reason}
