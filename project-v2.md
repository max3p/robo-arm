# Arm Brain V2: LLM tool-calling for a Waveshare RoArm-M2-S

## Vision

An LLM controls a desk robot arm through tool calls. The tools act in the physical world. The project has three phases:

1. **Phase 1 (done, 2026-11-07):** safe motion primitives (move, nudge, home, gripper, LED, replay a taught path),
   exposed to Claude as an MCP server. A human draws the workspace bounds by hand, and the AI cannot cross them.
2. **Phase 2 (next):** a growing **catalog of known movements** (skills) such as wave, nod, point and rock-paper-scissors,
   exposed to the LLM as extra tools.
3. **Phase 3 (later):** a camera on the arm and a perception worker that feeds scene information back to the LLM.

## Quick start

```
python cli.py      # human control: draw the bounds, teach paths, test moves (see the docstring for commands)
claude mcp add robo-arm -- $PWD/.venv/bin/python $PWD/mcp_server.py     # already registered locally for this project
```

- The MCP server and the CLI cannot run at the same time, because the serial port is exclusive.
- The server reads `arm_state.json` only when it starts. After you change the bounds or paths, run `/mcp` to reconnect it.
- Opening the port resets the board. The arm then moves to its boot pose, which takes up to about 10 s. Keep the workspace clear every time.
- The user account must be in the `dialout` group to open `/dev/ttyUSB0`.

## Hardware (Waveshare RoArm-M2-S)

- It is a 4-DOF desk arm: base, shoulder, elbow, and the gripper as the end tool. It has an ESP32 controller and serial bus servos.
  It cannot orient the gripper freely.
- Power is 7-12.6 V only. Use the supplied 12 V 5 A adapter. The arm reports about 12.17 V on that adapter.
- The servos are strong. Clamp the arm to the table, and keep hands and faces out of its range while it moves.
  **The power switch is the only real e-stop.**
- It connects over USB serial (115200 baud, one JSON object per line), on `/dev/ttyUSB0` here. It also supports Wi-Fi/HTTP,
  which is unused: in AP mode the SSID is `RoArm-M2` and the password is `12345678`.
- The arm runs inverse kinematics itself, so we send Cartesian targets.
  - Coordinates are in mm, with the origin at the shoulder pivot: x forward, y left, z up.
  - The two links are about 239 and 280 mm, so the claw can reach 41-519 mm from the origin.
- The boot pose is about (309, 4, 228) mm with the gripper shut. Joints read b≈0°, s≈0°, e≈91°, h≈180°.

## Arm command reference

Sources: Waveshare wiki pages "RoArm-M2-S Robotic Arm Control", "JSON Command Meaning" and "Step Recording and Reproduction".
In the Status column, *verified* means it was used and checked on our arm.

| Purpose | Command | Status / notes |
|---|---|---|
| Read pose | `{"T":105}` | **Verified.** Replies `{"T":1051, x,y,z (mm), b,s,e,t (rad), torB/S/E/H (load), torswitchB/S/E/H, v (0.01 V)}`. Other lines (boot text, echoes) are mixed in, so skip anything that isn't 1051 |
| Go to XYZ, direct | `{"T":1041,"x","y","z","t"}` | **Verified.** No interpolation. We stream it at 50 Hz for every move. `t` is the gripper angle in **radians** |
| Joint angles, degrees | `{"T":122,"b","s","e","h","spd","acc"}` | **Verified.** Its `spd`/`acc` **persist as firmware globals** and set the servo limits for later T:1041 moves |
| Joint angles, radians | `{"T":102,"base","shoulder","elbow","hand","spd","acc"}` | **Verified.** **Resets those globals to 0 (unlimited)**, which makes streamed T:1041 moves jerky. Used only by `lock()` |
| Gripper | `{"T":106,"cmd":t,"spd","acc"}` | **Verified.** 3.14 = shut, 1.08 = wide open. A single command starts late, so stream it |
| LED | `{"T":114,"led":0..255}` | **Verified** (brightness). The `r`/`g`/`b` colour fields come only from a community library and are untested |
| Torque | `{"T":210,"cmd":0\|1}` | **Verified.** 0 = limp (the arm sags; support it), 1 = locked |
| Go to XYZ, smooth | `{"T":104,...,"spd"}` | Unused. We do our own smooth streaming |
| Force adaptation | `{"T":112,"mode":1,"b","s","e","h"}` | Unverified. A candidate for a safe handshake |
| On-arm missions | `T:220-242` | Unverified and unused |

Things the firmware does not do, which our code handles:
- **It does not reject unreachable T:1041 targets.** IK produces NaN, which becomes a garbage servo goal. Always check reach first.
- **It sends no "move finished" signal.** Poll T:105 until two reads 0.1 s apart agree.

## Architecture (as built)

```
Claude (Code / Desktop) ──MCP tool call──► mcp_server.py ──► arm.py (Arm) ──serial──► arm
         ▲                                       │                  │
         └──── result: measured pose / error ◄───┘◄── T:105 feedback┘

Human ──► cli.py ──► arm.py ──► arm          cli.py writes arm_state.json (bounds + taught paths)
                                             mcp_server.py reads it once at startup, never writes it
```

The MCP client is the agent; there is no agent loop, LangGraph or API-key handling of our own. The design is split by who is in control:
- **The human side** (`cli.py`) does everything that needs a person holding the arm: drawing bounds and teaching paths.
- **The AI side** (`mcp_server.py`) can only move inside the bounds.

| File | Role |
|---|---|
| `arm.py` | Motion API: `Arm.go_to` (straight-line, safety-checked, sag-corrected), `gripper`, `play(points)`, `led`, `read`, `free`, `lock`; `check_target`; `load_state`/`save_state`; all tuned constants with the measurements behind them. **Never prints or prompts**: stdout carries the MCP protocol |
| `cli.py` | Human REPL. Commands: `read`, `move`, `grip` (radians), `home`, `led`, `bounds` (free the arm and sweep the claw; samples every 100 ms; the min/max becomes the box), `trace` (show the 6 faces of the box), `record NAME` (Enter per point, q to save; one point = a pose), `play NAME`, `paths`, `delete NAME`. Ctrl-C stops motion |
| `mcp_server.py` | AI tools (see below). The bounds are frozen at startup and the port opens lazily on the first tool call. `ValueError`/`RuntimeError` become `ToolError`, so the model sees the real reason plus the current pose |
| `arm_state.json` | `{"bounds": {"x": [lo, hi], "y": .., "z": ..}, "bounds_drawn": iso, "paths": {name: [[x, y, z, t], ..]}}`. Current box: x 117.6-478.4, y -102.4-368.1, z -102-231.4 |

### MCP tools

| Tool | Behaviour |
|---|---|
| `workspace()` | Bounds, reach, keep-out, speed range, saved movements and current pose |
| `get_pose()` | Measured `{x, y, z, gripper_open}`, where `gripper_open` runs from 0 (shut) to 1 (wide) |
| `move_to(x, y, z, speed=300)` | Straight-line move. Returns the measured pose, or an error |
| `move_by(dx, dy, dz, speed=300)` | Relative move from the **last target**, so sag doesn't accumulate. If the arm isn't near that target, it moves from the measured pose |
| `go_home(speed=300)` | Goes to the boot pose (309, 4, 228) and shuts the gripper |
| `gripper(opening)` | 0..1. It stops short on an object, and the result shows where it ended |
| `light(brightness, blinks=0)` | LED from 0 to 1, with optional blinks first |
| `play(name, speed=300)` | Replays a hand-taught path from `arm_state.json` |

The bounds reach the model before it acts in three ways: in the server instructions, in every motion tool's description, and from `workspace()`.

### Safety rules (enforced in code, never only in a prompt)

- **Only a human draws the bounds**, using `cli.py`. The MCP server refuses to start without them and has no tool that changes them.
- **Every move is a straight line through `Arm.go_to`.** It refuses the move before anything is sent if any of these is true:
  - the target is outside the box,
  - the target is outside 100-480 mm from the origin,
  - the target is not a finite number,
  - the straight path passes within 100 mm of the base axis.

  The box is convex, so a straight line between two points inside it stays inside it.
- **Paths replay in Cartesian space** as straight lines between recorded points, so they also stay inside the box.
  A path with any point outside the current box is refused whole. (Joint-space replay was dropped because it can bulge out of the box.)
- **Speed is 100-500 mm/s peak**, with smoothstep easing at both ends, for both the CLI and the AI.
- **Sag correction:** after settling, `go_to` aims past the target by the miss. The push is capped at 25 mm, with up to 2 passes.
  The *aim point* may be up to 25 mm outside the box; the claw itself ends inside it.
- **Post-move check:** a move that still ends more than 15 mm from its target raises an error, which says the arm is safe where it is.
- **Limitations:**
  - Only the claw tip is bounded, not the elbow or forearm, so draw the box smaller than the real free space.
  - The first move after connecting starts from the boot pose, which may be outside the box.
  - The AI cannot be stopped mid-move except with the power switch.

## Phase 1: what was done and how it went

1. **The plan was too big.** The first plan had a mock arm, a YAML config, a separate safety layer, a LangGraph agent and a calibration script.
   It was built and mock-tested before the arm arrived. Once the hardware was on the desk, the user replaced it all with one hand-tested
   `arm.py` REPL: straight-line moves, a hand-drawn bounds box, and record/replay. Lesson: with real hardware in hand, small and measured
   beats large and speculative.
2. **Split human/AI.** `arm.py` became a silent library. The REPL moved to `cli.py`, the bounds and paths were saved to JSON,
   and `mcp_server.py` was added. Making Claude the agent through MCP removed the whole agent layer.
3. **Tested on the real arm**, first with the library and then through a real MCP stdio client.
   - Every bounds, reach, NaN and keep-out refusal moved the arm 0.0 mm.
   - One bug was found. A single T:106 gripper command starts late, and the settle check read "no change yet" as "done".
     Fix: stream the gripper like everything else.
4. **Speed tuning.** The first speeds (20 mm/s default, 40 for the AI) were far too slow. A measured sweep over 370 mm moves gave this:

   | Commanded (mm/s) | Result |
   |---|---|
   | 30 | **Visibly jerky.** 50 Hz micro-steps are too small, so the servos stop and start. 60 already shows an uneven start in the samples. Hence the 100 mm/s floor |
   | 100-500 | Smooth, tracks the command, about 15 mm off the line (that is the sag, not the speed). 370 mm takes 1.2 s at 500 |
   | 600 | Borderline. The peak falls a little short of the command |
   | 800 | Servos saturate. The peak is about 700, 26 mm off the line. It looks smooth but breaks the straight-line guarantee |

   Servo limits sent via T:122 went from 90°/s with acceleration 2 to 360°/s with acceleration 20. That is about 20 % less lag at speed and no worse at slow speed.
5. **Additions:** `go_home`, and LED control (`light`).
6. **Accuracy.**
   - **The sag:** under its own weight the arm stops 10-20 mm short, worst moving *up* to high, stretched poses such as home.
   - **A worse stall, seen twice:** the arm sometimes stops 25-35 mm low near home. Both times it was right after connecting or during
     a long session. It could not be reproduced on demand, and the cause is unknown: possibly unsettled boot, possibly warm servos.
   - **`move_by` drift:** in another chat's gesture demo, building each `move_by` on the measured pose made the sag add up
     (43 mm of drift when the demo asked for 20).
   - **The fixes:** sag correction, and `move_by` building on the last target. Typical final error is now **0.2-4 mm**, against 5-13 mm before.
7. **Gripper speed.** The full range takes about 1.6 s with acceleration 50. Below that the servo itself is the limit, and stream speed doesn't help.

Other lessons:
- **Opening the port resets the ESP32**, and the arm moves. That's why the server opens the port lazily and only once.
  Set `dtr = rts = False` *before* `open()`, or the ESP32 stays in reset.
- **The first feedback after boot can arrive mid-motion.** It has been seen at (320, 4, 213) with the gripper half open.
  Don't treat the first reading as the boot pose.
- **mcp 2.x renamed FastMCP to `MCPServer`** (`mcp.server.mcpserver`).
  - Sync tools run in worker threads, so the `Arm` holds an `RLock` for physical actions.
  - Uncaught exceptions reach the client as a bare "Error executing tool", so raise `ToolError` with the real message.
- **Error wording steers the model.** "Unreachable or blocked?" sent the AI looking for an obstacle that wasn't there.
  Results should say what happened and whether it's safe.
- **Fast testing without moving the arm:** a ~20-line fake serial class can patch `arm.serial.Serial`. It answers T:105 and applies T:1041/T:106.
  It isn't in the repo.

## Known gaps carried into Phase 2

- **No tests or run logs in the repo.** The planned JSONL log of every tool call was never built, and the v1 tests were deleted with the v1 code.
- **The cause of the intermittent 25-35 mm stall is unknown.** If it recurs, log `torS`/`torE` and servo temperature (if the firmware exposes it).
- **`play` stops at every waypoint.** Each segment eases to zero speed, so multi-point gestures look segmented, not flowing.
- **`grip` units differ:** the CLI takes radians and the MCP tool takes an opening from 0 to 1.
- **The box never expires.** If the arm or the table moves, the box is stale; only `bounds_drawn` hints at its age.

## Phase 2: catalog of known movements

What Phase 1 already gives:
- A skill can be a named path in `arm_state.json`, taught by hand (`record NAME`) and run with `play(name)`.
- Every point is bounds-checked before the arm moves.

What Phase 2 should add, roughly in order:

1. **Smooth multi-point motion.** Blend through waypoints instead of stopping at each one, for example a Catmull-Rom spline
   streamed at 50 Hz with easing only at the ends. Gestures like wave and nod depend on this.
   Each blended segment must still be bounds-safe. Spline overshoot can leave the box, so check the sampled curve, not only the points.
2. **Skill metadata.** Give each path a plain-English `description` (which becomes what the model reads), a `status`
   (`draft`/`approved`) and a version. Keep them in `arm_state.json`, or in a `skills/` folder if it grows.
3. **Per-skill tools, or a better `play` description.** For a handful of skills, list each name and description in `play`'s
   tool description. Register separate tools only if that helps the model. Past a few dozen, retrieve by embedding.
4. **Relative / parameterized skills in code.** `wave`, `nod` and `point_at(x, y, z)` should run around the *current* pose
   (built on `go_to` / `move_by`), not at fixed recorded coordinates. Timing-critical skills such as rock-paper-scissors
   (gripper open, half, shut) must be deterministic code; the LLM only chooses *when* to run them. The LED fits here as a signal.
5. **LLM-proposed skills.** A `save_movement(name, points, description)` tool saves as `draft`. A human dry-runs it and approves it in
   `cli.py`, and only `approved` skills are playable. The server would then need to reload state, not read it once.
6. **Handshake (later).** Try force adaptation (`T:112`, unverified) at low speed, with an object first, never a hand first.
7. **Tooling worth adding along the way:** the fake serial class as `tests/`, and a JSONL log of tool calls and results.

Rules that carry over:
- Skills only call primitives, so every move still goes through `go_to`'s checks.
- Use measured results, not assumptions: test every new motion pattern on the arm and record the numbers here.

## Phase 3: camera and perception (do not build yet)

- A camera mounted on the arm. A perception worker (a thread or process) owns the camera, runs open-vocabulary detectors
  (YOLO-World, OWL-ViT, Grounding DINO), keeps the latest scene and emits debounced events.
- Tool results gain an observation package: an optional downscaled image, plus a JSON scene of entities with labels and world positions in mm.
- Depth comes from a table-plane assumption first (cheapest), via a one-time hand-eye calibration that uses the T:105 pose.
- The tool interface stays stable: Phase 3 only enriches results and adds a `look()` tool.

## Constraints

- Python 3.10+ (the venv runs 3.14). The only dependencies are `pyserial` and `mcp` (2.x).
- Keep each layer small and readable. The user prefers the least code that does the job.
- Keep secrets out of the code. Don't print to stdout in anything the MCP server imports.
