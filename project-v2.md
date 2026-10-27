# Arm Brain V2: LLM tool-calling for a Waveshare RoArm-M2-S

This file supersedes `PROJECT.md`. The design changed after more research and discussion: the arm is now
controlled through **absolute XYZ coordinates**, and the camera/vision work is deferred to a later phase.

## Vision

An LLM agent controls a desk robot arm through tool calls, the way a LangGraph agent calls web tools, but the
tools act in the physical world. The project grows in three phases:

1. **Phase 1 (build now):** a safe `move_to(x, y, z)` tool, plus gripper, home and pose tools, driven by an LLM agent.
2. **Phase 2:** a growing **catalog of known movements** (skills) such as point, wave, handshake and rock-paper-scissors, exposed to the LLM as extra tools.
3. **Phase 3 (later, do not build yet):** a camera on the arm and a perception worker that feeds scene information back to the LLM.

The hardware has been ordered and may not have arrived. **Everything must run against a mock arm that logs
commands to the console.** Real hardware is selected with a CLI flag.

## Hardware (Waveshare RoArm-M2-S)

- 4-DOF desktop arm (base, shoulder, elbow, end tool/gripper), ESP32 controller, serial bus servos.
- Pre-assembled. Do not disassemble it.
- Power: 7-12.6 V only. Use the supplied 12 V 5 A adapter. Higher voltage can damage it.
- The servos are strong. Keep hands and face out of its range while it moves. Clamp it to the table.
- Connection: USB serial (115200 baud) with JSON lines, or Wi-Fi/HTTP. In AP mode the arm broadcasts SSID `RoArm-M2`,
  password `12345678`, and serves a web control page.
- The arm runs inverse kinematics itself. We send Cartesian targets.
- Payload is about 0.5 kg at 0.5 m reach. It cannot orient the gripper freely (4-DOF).

## Arm command reference

Sources: Waveshare wiki pages "RoArm-M2-S Robotic Arm Control" and "RoArm-M2-S JSON Command Meaning".
Verify every item below against the wiki and a real arm before relying on it. Commands are JSON objects, one per line.

| Purpose | Command | Notes |
|---|---|---|
| Go to XYZ, smooth (IK) | `{"T":104,"x":235,"y":0,"z":234,"t":3.14,"spd":0.25}` | x, y, z in **mm**; `t` = end-tool angle in **radians**; may block |
| Go to XYZ, direct | `{"T":1041,"x":235,"y":0,"z":234,"t":3.14}` | Non-blocking, no interpolation, moves as fast as possible; successive targets must not differ greatly |
| Joint angles (radians) | `{"T":102,"base":0,"shoulder":0,"elbow":1.57,"hand":3.14,"spd":0,"acc":10}` | Also `T:101` for a single joint |
| Joint angles (degrees) | `{"T":122,"b":0,"s":0,"e":90,"h":180,"spd":10,"acc":10}` | Also `T:121` for a single joint |
| Gripper | `{"T":106,"cmd":3.14,"spd":0,"acc":0}` | Default clamp range 1.08-3.14; **decreasing the value opens** the clamp, so 3.14 = closed, about 1.08 = fully open |
| Read pose and status | `{"T":105}` | Returns JSON with `"T":1051` and: `x,y,z` (mm), `b,s,e,t` (radians), `torB/S/E/H` (joint load), `torswitchB/S/E/H` (torque lock), `v` (voltage, 0.01 V units) |
| Torque lock | `{"T":210,"cmd":0}` | 0 = joints free (move by hand), 1 = locked; any rotation command re-locks |
| Force adaptation | `{"T":112,"mode":1,"b":60,"s":110,"e":50,"h":50}` | Yields to external force above the limits, then returns; use for anything that touches a person |

Known uncertainties (resolve by testing on the real arm, and write results into `config.yaml` and the README):
- Third-party docs list `t` in degrees; Waveshare's own page says radians. Treat radians as correct until verified.
- The docs do not describe a "move finished" signal. Implement settle detection by polling `T:105` until
  the pose is within a tolerance of the target and stable across consecutive reads. Add a timeout.
- Behavior for unreachable targets is undocumented. Always compare the final pose against the target after every move.
- Home pose, workspace limits and gripper angles are unknown. The wiki examples use roughly (235, 0, 234) and (320, -3, 212).
  Use these only as hints for a safe start, and calibrate on the real arm.
- Waveshare's own "missions" feature (create, append position, play) exists on the arm (`T:220-242`, unverified). It is optional.

## Architecture (Phase 1)

```
LLM agent (LangGraph) ──tool call──► Tool layer ──► Safety layer ──► Arm driver ──► arm (or mock)
        ▲                                │                                │
        └────────── result: final pose / error ◄───────── pose feedback ───┘
```

Principles:
- The LLM only **proposes** actions. Deterministic code validates and executes. No safety logic lives only in a prompt.
- Tool results report **observed outcomes** (the measured final pose from `T:105`), never just "ok".
- One physical action at a time. Guard the arm driver with a lock.
- Every layer sits behind an interface so it can be mocked: arm driver, safety, tools, agent.

### Project layout

```
arm_brain/
  PROJECT_V2.md
  config.yaml            bounds, home pose, gripper angles, speeds, port (loaded at startup)
  arm.py                 Arm ABC, Pose, MockArm (console), RoArmSerial (real)
  safety.py              BoundsChecker, waypoint splitting, validation
  tools.py               Tool functions the LLM can call (thin wrappers over safety + arm)
  agent.py               LangGraph agent wiring the tools to an LLM
  main.py                CLI entry point
  tests/                 pytest
  logs/                  one .jsonl per run
  skills/                Phase 2 skill files (create the folder, leave it empty for now)
```

Existing code from the earlier scaffold (`arm.py`, `safety.py`, `brain.py`, `main.py`, `vision.py`) used relative
joystick-style moves with a simulated camera. Keep `arm.py`'s `Arm`, `Pose` and `MockArm` ideas, rework
`safety.py` and `main.py`, and park `vision.py` and the old `brain.py` for Phase 3. Do not delete them.

### Arm interface (`arm.py`)

- `get_pose() -> Pose` queries real feedback (`T:105`). `MockArm` returns its simulated pose.
- `move_to(Pose, speed) -> MoveResult` sends the move, waits until settled (or times out), then returns the
  measured final pose and any error. `MoveResult` has `ok`, `final_pose`, `target`, `error_mm`, `message`.
- `set_gripper(closed: bool)`, `home()`, `torque(enabled: bool)`, `close()`.
- `MockArm` logs each command as `[MOCK ARM] move_to x=.. y=.. z=..` and simulates travel. Optionally simulate
  an unreachable target and a drifting final pose so error paths can be tested.
- `RoArmSerial` uses `pyserial`, sends the JSON commands above, parses `T:1051` responses, and has a reader that
  tolerates unrelated lines the arm may print. Use a lock around serial I/O.

### Safety layer (`safety.py`)

Bounds are an axis-aligned box (invisible walls) loaded from `config.yaml`, plus extra rules.

- **Target check:** reject any target outside the box or with non-finite or non-numeric values. Rejecting is
  preferred over silently clamping. Return a clear error so the LLM can retry with a valid target.
- **Path check:** a safe start and a safe end do not make a safe path, because the arm may interpolate in joint space.
  Split any move longer than `max_segment_mm` into waypoints, check each, and move through them in order.
- **Floor:** a minimum `z` keeps the gripper off the table. Add a keep-out near the base if needed.
- **Whole-arm risk:** the elbow and forearm can hit things while the tip stays inside the box. Keep the box
  conservative, and note this limitation in the README.
- **Speed cap:** clamp commanded speed to `max_speed`.
- **Post-move verification:** after each move, compare the measured pose with the target. If the error exceeds
  `tolerance_mm`, stop, report it, and do not continue a sequence.
- **Emergency stop:** Ctrl-C (and Esc in any preview window) stops, disables further commands and, if the arm
  is connected, leaves it in a safe state (torque on, no new motion). A hardware power switch remains the real e-stop.
- First real-hardware runs: tight bounds, low speed, arm clamped, workspace clear, someone ready to cut power.

### Tools exposed to the LLM (`tools.py`)

| Tool | Behavior |
|---|---|
| `move_to(x, y, z)` | Validate target, split into waypoints, move, verify. Returns final pose or an error |
| `move_by(dx, dy, dz)` | Relative nudge, same validation (useful for fine adjustment) |
| `get_pose()` | Returns the current measured pose |
| `gripper(action)` | `"open"` or `"close"` |
| `go_home()` | Move to the configured home pose through validated waypoints |
| `get_bounds()` | Returns the allowed box so the LLM can plan within it |
| `done(reason)` | Ends the task |

All results are small JSON objects like `{"status":"ok","final_pose":{...},"error_mm":1.2}` or
`{"status":"rejected","reason":"x=500 is outside allowed range [120, 320]"}`.

### Agent (`agent.py`)

Use LangGraph with a simple loop: LLM node (with tools bound) -> tool node -> back to the LLM until `done` is
called or a step limit is hit. Provider model is configurable (`--model`); default to a fast Claude model.
Keep the system prompt short: coordinate convention (mm, x forward, y left, z up), the bounds, and the rule that the
agent should recover from rejections by choosing a valid target rather than repeating the same call.
Include a `--max-steps` limit and log every tool call and result to `logs/*.jsonl`.

### CLI

```
python main.py --arm mock --goal "move to the centre of the workspace, then wave the gripper"
python main.py --arm roarm --port /dev/ttyUSB0 --goal "..."      # real hardware, only after calibration
python main.py --arm mock --interactive                           # type tool calls by hand to test safety
```

## Phase 2: catalog of known movements (design now, build after Phase 1 works)

Three layers:
1. **Primitives:** the Phase 1 tools. They are hand-written and safety-checked.
2. **Skills:** named, parameterized sequences of primitives, stored as files in `skills/` (YAML or JSON) with a name, a
   plain-English description (this becomes the tool description the LLM reads), parameters, ordered steps,
   a `status` (`draft` or `approved`), and a version.
3. **Catalog loader:** at startup, load every `approved` skill and register it as a LangGraph tool.

Rules:
- A skill may only call primitives, so every move still goes through the safety layer. Validate a skill on save (dry run
  through the checker against the mock arm) and again every time it runs.
- New skills start as `draft`. A human approves them before they appear in the agent's toolbelt.
- Ways to create skills: (a) **teach by demonstration**: torque off, move the arm by hand, record poses with `T:105`, save
  as waypoints (build a small `record_skill.py`); (b) **LLM-proposed**: the agent composes primitives, a human reviews a
  dry run, then approves; (c) **parameterized**: generalize a working skill (e.g. `point_at(x,y,z)`).
- First skills to try: `wave`, `point_at(x,y,z)`, `nod`. Later: `handshake` (use force-adaptation `T:112`, low speed, test with an
  object first), and `rock_paper_scissors` (gripper only: open/half/closed approximates paper/scissors/rock).
- Timing-critical skills (like rock-paper-scissors) must be deterministic code, not LLM-driven; the LLM chooses *when* to run them.
- When the catalog grows past a few dozen skills, retrieve the top matches by embedding descriptions instead of listing all.

## Phase 3: camera and perception (do NOT build yet; context only)

Plan so Phase 1 and 2 do not block it:
- A camera mounted on the arm. A **perception worker** (separate thread or process) owns the camera, runs detectors
  continuously, keeps a latest-scene record, and emits debounced events.
- After each tool call, the agent receives one **observation package**: tool result, optional downscaled image, and a JSON scene
  with entities (label, confidence, pixel box, world position in mm, distance from gripper, position method and confidence).
- 3D positions need depth. Options: table-plane assumption (cheapest), depth camera, monocular depth model, known object sizes.
  A one-time hand-eye calibration converts camera coordinates to arm base coordinates using the pose from `T:105`.
- Use open-vocabulary detectors (YOLO-World, OWL-ViT, Grounding DINO) so new recognizable things are just new text labels.
  Add object tracking and a world-frame short-term memory with timestamps.
- Keep the tool interface stable: Phase 3 only enriches the tool results and adds a `look()` tool.

## Tasks for Claude Code (Phase 1)

Work in order and verify each step before moving on. Read the Waveshare wiki pages before touching `RoArmSerial`.

1. Create `config.yaml` with conservative placeholder bounds, home pose, gripper angles, speeds and tolerances, and a loader. Mark every
   placeholder with a comment.
2. Rework `arm.py` to the interface above (including `MoveResult` and settle detection). Implement `MockArm` fully, with an optional
   error simulation.
3. Rewrite `safety.py`: target validation, waypoint splitting, floor, speed cap and post-move verification. Write thorough `pytest` tests
   (outside the box, NaN/strings, long moves, boundary values, error-after-move).
4. Implement `tools.py` and an `--interactive` mode so tools can be called by hand against the mock.
5. Build the LangGraph agent in `agent.py` and run it end to end against the mock with a real LLM key (`ANTHROPIC_API_KEY` from the
   environment, never in code). Check that rejections are recovered from, not looped on.
6. Implement `RoArmSerial` for real, from the wiki. Do **not** run it against hardware; stop and tell me what to test first.
7. Write `calibrate.py`: jog the arm by hand (torque off), print live poses from `T:105`, and write measured bounds, home pose and gripper
   angles into `config.yaml`.
8. Only then start on Phase 2 (`skills/` format, loader, recorder, and the dry-run validator).

## Constraints

- Python 3.10+. Minimal dependencies: `pyserial`, `pyyaml`, `langgraph`, a Claude/Anthropic client package, `pytest`. No OpenCV needed in Phase 1.
- Logs: one JSONL line per tool call with the request, the safety decision, the arm result and timings.
- Keep each layer small and readable. Do not add frameworks the task does not need.
- No secrets in code. Read keys from environment variables.