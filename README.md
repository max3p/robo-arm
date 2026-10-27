# Arm Brain V2

Claude drives a Waveshare RoArm-M2-S through tool calls (`move_to`, `move_by`, `gripper`, ...).
Deterministic code validates every call before the arm moves. See [project-v2.md](project-v2.md).

```
agent.py (Claude) -> tools.py -> safety.py -> arm.py (MockArm | RoArmSerial)
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
sudo usermod -aG dialout $USER     # Linux serial access; log out and back in afterwards
```

## Run

```bash
python main.py --interactive                                  # type tool calls by hand: move_to 200 0 150
python main.py --goal "move to the centre, then wave the gripper"   # needs ANTHROPIC_API_KEY
python main.py --interactive --mock-drift 15                  # exercise the post-move error path
python main.py --arm roarm --goal "..."                       # real arm, after calibration
```

Every tool call is logged to `logs/run-*.jsonl`. Ctrl-C stops: no further motion, torque stays on.

## When the hardware arrives

1. Use the supplied 12 V adapter. Clamp the arm, clear the workspace, and keep a hand on the power switch.
2. `python calibrate.py`: torque goes off, so support the arm. Move it by hand to the workspace extremes
   (`m` at each, including the lowest safe height). Then `h` at a good home pose, `o`/`c` with the
   gripper open/closed, and `s` to write `config.yaml`.
3. `python main.py --arm roarm --interactive --speed 15`: try `get_pose`, then small `move_by` steps.
   Check that x is forward, y is left, and the reported pose matches reality.
4. `python main.py --arm roarm --goal "..."`.

Check on the real arm first:
- Does `t` (gripper) use radians, as the Waveshare page says?
- Is `tolerance_mm` realistic?
- What happens at unreachable targets? (The final pose is always compared with the target.)

## Notes

- **Agent:** a plain Anthropic SDK tool loop (~45 lines) instead of LangGraph. It needs fewer
  dependencies and gives the same LLM → tool → LLM loop.
  - A run stops when Claude calls `done`, at `--max-steps`, when it repeats a rejected call, or when a move ends off target.
- **Moves:** `RoArmSerial` streams each move as 3 mm `T:1041` steps at the configured mm/s, then
  polls `T:105` until two consecutive reads agree.
  - The tip therefore travels in a straight line. Waypoints every `max_segment_mm` are each verified against the measured pose.
  - Because every point on a straight line between two points inside the box is also inside it, only the target needs a bounds check.
- **Limitation:** only the gripper tip is bounded. The elbow and forearm can still hit things inside the box, so keep the box conservative.
- **Parked code:** the v1 camera code is in `parked/v1/`, kept for Phase 3. `skills/` is reserved for Phase 2.
