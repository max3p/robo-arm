# Arm Brain V2: LLM tool-calling for a Waveshare RoArm-M2-S

## Vision

An LLM controls a desk robot arm through tool calls. The tools act in the physical world. The project has three phases:

1. **Phase 1 (done, 2026-11-07):** safe motion primitives (move, nudge, home, gripper, LED, replay a taught path),
   exposed to Claude as an MCP server. A human draws the workspace bounds by hand, and the AI cannot cross them.
2. **Phase 2 (done, 2026-11-07):** a **catalog of known movements** that make the arm read as a clever companion or pet
   with a personality: body language (nod, shake, curious, happy, excited, sad, startle, sleep, wake_up, look_at),
   exposed to the LLM as tools.
   - **Phase 2.5 (built, 2026-11-07):** `arm.py`, our own terminal chat on the Claude API. You talk to the arm as
     a pet and it answers with movement. Input, brain and body all run at once, and any of them can interrupt the others.
   - **Phase 2.6 (built, 2026-11-07):** speech in. `ears.py` is always listening on the Blue Snowball and
     transcribes locally on the GPU. The arm answers only with movement.
   - **Phase 2.7 (built, 2026-11-07):** idle life. Between inputs the body fidgets by itself, without the LLM:
     content, then bored, then sleepy, then asleep.
   - **Phase 2.8 (built, 2026-11-07):** its name is Robert. Asleep, it ignores everything said except "Robert" or
     "Rob", which wakes it.
3. **Phase 3 (later):** a camera on the arm and a perception worker that feeds scene information back to the LLM.

## Quick start

```
python cli.py      # human control: draw the bounds, teach paths, test moves (see the docstring for commands)
python arm.py      # talk to the arm as a pet (or type; API key in .env). Ctrl-C stops it at once and quits
claude mcp add robo-arm -- $PWD/.venv/bin/python $PWD/mcp_server.py     # already registered locally for this project
```

- The CLI, the companion and the MCP server cannot run at the same time, because the serial port is exclusive.
  Close the Claude Code chat (or disconnect the server with `/mcp`) before running the companion.
- The server reads `arm_state.json` only when it starts. After you change the bounds or paths, run `/mcp` to reconnect it.
- Opening the port resets the board. The arm then moves to its boot pose, which takes up to about 10 s, and from there
  to the resting pose `START`. Keep the workspace clear every time.
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
- The boot pose (`HOME`) is about (309, 4, 228) mm with the gripper shut. Joints read b≈0°, s≈0°, e≈91°, h≈180°.
  It is an upside-down L stretched forward, the worst pose for sag, and the firmware always boots to it.
- **The resting pose `START` is (161.5, -2.0, 213.6) mm, gripper shut:** the arm folded back over the base, set by hand
  by the user on 2026-11-07 and checked on the arm. `Arm(bounds)` waits for the boot motion to settle, then moves there
  at 200 mm/s (skipped if `START` is outside the box). `go_home`, the CLI's `home` and the end of `wake_up` all go there.

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
Claude (Code / Desktop) ──MCP tool call──► mcp_server.py ──► motion.py (Arm) ──serial──► arm
         ▲                                       │                  │
         └──── result: measured pose / error ◄───┘◄── T:105 feedback┘

Human ──► cli.py ──► motion.py ──► arm          cli.py writes arm_state.json (bounds + taught paths)
                                             mcp_server.py reads it once at startup, never writes it

Human ──► arm.py: input ─► brain (Claude API, streaming) ─► body thread ─► motion.py ──► arm      (see Phase 2.5)
         (voice ─► ears.py, or typed)
```

Through MCP, the client (Claude Code) is the agent, with no loop of our own. The pet chat `arm.py` (Phase 2.5) is our own
loop on the Claude API instead, with its API key in `.env`; both drive the same `motion.py`. The design is split by who is
in control:
- **The human side** (`cli.py`) does everything that needs a person holding the arm: drawing bounds and teaching paths.
- **The AI side** (`mcp_server.py`, `arm.py`) can only move inside the bounds.

| File | Role |
|---|---|
| `motion.py` | Motion API: `Arm.go_to` (straight-line, safety-checked, sag-corrected), `gripper`, `play(points)`, `led`, `read`, `free`, `lock`; `perform(keys, size)` plays a gesture (see below); `abort` (a `threading.Event`: set from any thread, the move in progress holds the claw where it is and raises `Interrupted` within 20 ms); `check_target`; `describe_limits`; `load_state`/`save_state`; all tuned constants with the measurements behind them. **Never prints or prompts**: stdout carries the MCP protocol |
| `arm.py` | The pet chat (Phase 2.5): prompt, brain loop, body thread, the tools and the system prompt. Knobs at the top: `MODEL`, `EFFORT`, `IDLE_GAP_S`, `HUSH_S`, `MOODS` (idle, Phase 2.7), `PERSON` |
| `ears.py` | Speech in (Phase 2.6): mic thread, Silero voice detector, faster-whisper on the GPU. Knobs at the top: `MIC`, `STT_MODEL`, `SILENCE_S`, `MIN_SPEECH_S`, `MIN_LOGPROB` |
| `cli.py` | Human REPL. Commands: `read`, `move`, `grip` (radians), `home`, `led`, `bounds` (free the arm and sweep the claw; samples every 100 ms; the min/max becomes the box), `trace` (show the 6 faces of the box), `record NAME` (Enter per point, q to save; one point = a pose), `play NAME`, `gesture NAME [size]`, `look x y z`, `paths`, `delete NAME`. Ctrl-C stops motion |
| `gestures.py` | The gestures as data: keyframes `K(secs, dr, da, dz, grip, led)` relative to the current pose (dr out from the base axis, da degrees of base turn, dz up), plus functions for the aimed ones (`sleep`, `wake_up`, `look_at`) |
| `mcp_server.py` | AI tools (see below). The bounds are frozen at startup and the port opens lazily on the first tool call. `ValueError`/`RuntimeError` become `ToolError`, so the model sees the real reason plus the current pose |
| `arm_state.json` | `{"bounds": {"x": [lo, hi], "y": .., "z": ..}, "bounds_drawn": iso, "paths": {name: [[x, y, z, t], ..]}}`. Current box: x 117.6-478.4, y -102.4-368.1, z -102-231.4 |

### MCP tools

| Tool | Behaviour |
|---|---|
| `workspace()` | Bounds, reach, keep-out, speed range, saved movements and current pose |
| `get_pose()` | Measured `{x, y, z, gripper_open}`, where `gripper_open` runs from 0 (shut) to 1 (wide) |
| `move_to(x, y, z, speed=300)` | Straight-line move. Returns the measured pose, or an error |
| `move_by(dx, dy, dz, speed=300)` | Relative move from the **last target**, so sag doesn't accumulate. If the arm isn't near that target, it moves from the measured pose |
| `go_home(speed=300)` | Goes to the resting pose `START` (161.5, -2, 213.6) and shuts the gripper |
| `gripper(opening)` | 0..1. It stops short on an object, and the result shows where it ended |
| `light(brightness, blinks=0)` | LED from 0 to 1, with optional blinks first |
| `play(name, speed=300)` | Replays a hand-taught path from `arm_state.json` |
| `gesture(name, size=1)` | Body language, by name (an enum in the schema; the description says what each means). Relative ones play wherever the claw is and end where they started; `size` 0.2-1.5. If one does not fit it first slides toward the middle of the box, then shrinks (down to 0.35x); the result gives `size_used` and `slid_mm` |
| `look_at(x, y, z)` | Turns the base to face a point and moves half way to its height, then a small perk. Stays facing it; turns part way if the full turn does not fit |

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
- **Gestures go through `Arm.perform`**, which samples the whole path at 50 Hz (cylindrical interpolation, so a base turn is an
  arc) and checks every sample against the box, reach and keep-out before anything moves. It keeps `go_to`'s sag correction
  as a constant offset and corrects again at the end, so repeated gestures do not sink. Segments are stretched to stay under 500 mm/s.
- **Speed is 100-500 mm/s peak**, with smoothstep easing at both ends, for both the CLI and the AI.
- **Sag correction:** after settling, `go_to` aims past the target by the miss. The push is capped at 25 mm, with up to 2 passes.
  The *aim point* may be up to 25 mm outside the box; the claw itself ends inside it.
- **Post-move check:** a move that still ends more than 15 mm from its target raises an error, which says the arm is safe where it is.
- **Limitations:**
  - Only the claw tip is bounded, not the elbow or forearm, so draw the box smaller than the real free space.
  - If `START` is outside a redrawn box, the arm stays at the boot pose after connecting, which may be outside the box too.
  - Through MCP, the AI cannot be stopped mid-move except with the power switch. The companion can stop it: Ctrl-C (which also quits),
    the `stop` tool and every newer reply set `Arm.abort`.

## Phase 1: what was done and how it went

1. **The plan was too big.** The first plan had a mock arm, a YAML config, a separate safety layer, a LangGraph agent and a calibration script.
   It was built and mock-tested before the arm arrived. Once the hardware was on the desk, the user replaced it all with one hand-tested
   `arm.py` REPL: straight-line moves, a hand-drawn bounds box, and record/replay. Lesson: with real hardware in hand, small and measured
   beats large and speculative.
2. **Split human/AI.** `arm.py` (now `motion.py`) became a silent library. The REPL moved to `cli.py`, the bounds and paths were saved to JSON,
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

- **No tests in the repo.** The v1 tests were deleted with the v1 code; Phase 2's tests lived in a scratch folder. Run
  logs do exist now, for `arm.py` only: `logs/<start time>/transcript.jsonl` (Phase 2.5). The MCP server logs nothing.
- **The cause of the intermittent 25-35 mm stall is unknown.** If it recurs, log `torS`/`torE` and servo temperature (if the firmware exposes it).
- **`play` stops at every waypoint** (it settles and sag-corrects at each). Gestures do not: they use `perform`.
- **`grip` units differ:** the CLI takes radians and the MCP tool takes an opening from 0 to 1.
- **The box never expires.** If the arm or the table moves, the box is stale; only `bounds_drawn` hints at its age.

## Phase 2: catalog of known movements

What Phase 1 already gives:
- A skill can be a named path in `arm_state.json`, taught by hand (`record NAME`) and run with `play(name)`.
- Every point is bounds-checked before the arm moves.

**Progress (2026-11-07):** items 1 and 4 below are built: `Arm.perform` and `gestures.py`, nine gestures (nod, shake,
curious, happy, excited, sad, startle, sleep, wake_up) plus `look_at`, as the `gesture` and `look_at` MCP tools and CLI
commands. The user reviewed every gesture on the arm; one round of tuning followed (deeper and faster nod, faster shake,
longer curious lean, lower moping sad, sleep folded down and back, higher and straighter wake_up stretch, new excited;
wave and shrug were dropped because a 4-DOF arm with no wrist cannot make them read).
Measured on the arm after tuning (claw sampled at 12.5 Hz during each gesture):
- Every gesture ended within 0.5-3.6 mm of where it started, thanks to the sag correction at the end of `perform`
  (before it, a nod ended 13 mm low).
- The claw lags the stream by up to ~75 mm in fast gestures (98 mm in a fast `look_at` turn) and covers ~85 % of the
  commanded vertical swing in a fast nod (67 of 80 mm); the slow ones track closely (sad and sleep lag 22-28 mm).
  So fast gestures need about 15-20 % extra amplitude to look the size they were designed at.
- Gestures play from any pose. If one does not fit, the claw first slides (same direction) toward the middle of the box,
  then it shrinks. Turned far left (y ~ 300) the curious lean and the turns shrank to 0.7x: that corner is tight.
  Since 2026-11-07 a gesture that slid also slides back at the end. Before that it ended at the slid spot, and in live
  sessions the claw wandered away from rest gesture by gesture.
- **Eleven more gestures (2026-11-07)**, so the pet can say more without words: lean_in, calm, tilt, confused, shy,
  play_bow, beg, proud, laugh, greet, yawn. All 18 relative gestures fit at full size, from rest and from a pose turned
  left; the fake arm ran each one, and each ended where it began. lean_in, calm and yawn are deliberately slow
  (median ~40 mm/s). The user hasn't yet checked by eye whether they look smooth.
- **An interrupted gesture puts the light back** where it was before it began. Before, a flash cut off part way left the
  LED stuck at that level, and later gestures kept returning to it.
- **The resting pose `START`** (see Hardware) replaced the boot pose as the place the arm connects to and rests at.
  The user checked it on the arm. It is only 18 mm below the top of the box, so from there the upward gestures (happy,
  startle, excited) first slide 50-100 mm forward and down to fit; the others play in place.
- Not yet checked by eye: whether the slowest parts (sad's mope at ~70-130 mm/s) look stepped, and LED lag.

What Phase 2 should add, roughly in order:

1. **Smooth multi-point motion (built as `perform`; `play` could use it too).** Blend through waypoints instead of stopping at each one, for example a Catmull-Rom spline
   streamed at 50 Hz with easing only at the ends. Gestures like wave and nod depend on this.
   Each blended segment must still be bounds-safe. Spline overshoot can leave the box, so check the sampled curve, not only the points.
2. **Skill metadata.** Give each path a plain-English `description` (which becomes what the model reads), a `status`
   (`draft`/`approved`) and a version. Keep them in `arm_state.json`, or in a `skills/` folder if it grows.
3. **Per-skill tools, or a better `play` description.** For a handful of skills, list each name and description in `play`'s
   tool description. Register separate tools only if that helps the model. Past a few dozen, retrieve by embedding.
4. **Relative / parameterized skills in code (built: `gestures.py`).** `wave`, `nod` and `point_at(x, y, z)` should run around the *current* pose
   (built on `go_to` / `move_by`), not at fixed recorded coordinates. Timing-critical skills such as rock-paper-scissors
   (gripper open, half, shut) must be deterministic code; the LLM only chooses *when* to run them. The LED fits here as a signal.
5. **LLM-proposed skills.** A `save_movement(name, points, description)` tool saves as `draft`. A human dry-runs it and approves it in
   `cli.py`, and only `approved` skills are playable. The server would then need to reload state, not read it once.
6. **Handshake (later).** Try force adaptation (`T:112`, unverified) at low speed, with an object first, never a hand first.
7. **Tooling worth adding along the way:** the fake serial class as `tests/`, and a JSONL log of tool calls and results.

Rules that carry over:
- Skills only call primitives, so every move still goes through the checks in `go_to` or `perform`.
- Use measured results, not assumptions: test every new motion pattern on the arm and record the numbers here.

## Phase 2.5: the pet chat, `arm.py` (built 2026-11-07)

Naming: the program you run is `arm.py`. The motion library, which was `arm.py` through Phase 2, is now `motion.py`.

Why it exists: Claude Code through MCP is turn-based. Every tool call blocks until the arm has settled, and nothing can
cut in. The companion is our own app on the Claude API, so we control the inputs, and nothing waits for anything else.

**The parts.** One asyncio program; the three parts talk only through queues.
- **Input:** a `prompt_toolkit` prompt. What the arm does scrolls above it while you type. The screen is for debugging
  only: the end product has no visible terminal, and audio and video will be the input.
- **Brain:** Claude, streaming. Only new messages and failures wake it.
  - The model is `claude-haiku-5-5` at effort `low`, because the user put speed first: the delay from a message to the
    first move matters most. At low effort it thinks less, and skips thinking on simple turns.
  - If it seems too dim, raise `EFFORT`, or switch `MODEL` to Sonnet 5.5 or Opus 5.5 (slower).
  - The design started on Opus 5.5.
- **Body:** one thread that owns the `Arm` and plays actions in order.

**Decisions:**
1. **It cannot talk, like a dog: everything it does is a tool call** (`gesture`, `look_at`, `move_to`, `go_home`,
   `gripper`, `light`, `wait`, `stop`), and movement, the claw and the light are its whole language. Decided by the
   user on 2026-11-07.
   - The first version had a `say` tool. It was dropped because the end product has no screen.
   - Any text the model writes is shown dimmed, for debugging only, and never acted on. In the first live session after
     the change it wrote none.
   - Gestures always play at full size; the model can no longer pick a size. It used to shrink most of them to
     0.5-0.7x, which made them hard to read. `perform` still shrinks one that can't fit anywhere.
   - **The system prompt sets its character**: a pet, not an assistant. It's told to:
     - answer yes or no with nod or shake;
     - show confused or tilt for what it can't understand or can't do, and never pretend;
     - match the mood: gentle for sadness, energy for good news, laugh for jokes, proud or shy for praise;
     - let its feelings carry over between messages;
     - vary its answers, using one to three actions;
     - never make up facts about its body.
   - **Settling:** after `SETTLE_S` (20 s) without a movement, the body goes back to rest by itself, at 150 mm/s. It
     doesn't do this after sleep or after it's already home. The prompt also asks it to settle when things calm down.
     (Since Phase 2.7 this is the first step of idling, right after the last movement; `SETTLE_S` is gone.)
2. **Each call runs as soon as its block has streamed** (`eager_input_streaming`, acting at `content_block_stop`), so
   the arm moves while Claude is still writing.
   - Results only say "queued". They go back with the next message, so a reply costs one API call, not one per move.
   - Inputs arrive unvalidated, so the code checks them; a bad one gets an error result and wakes the brain again.
3. **A new message cancels the reply in progress.** Nothing from the cut reply is sent back. Instead, the next request
   carries a system note: "cut off ... you had already queued nod".
4. **The conversation is only ever appended to.** That keeps the prompt cache and the model's preserved thinking valid
   (Haiku 5.5 and Opus 5.5 both reject a history that was edited under their thinking).
   Each turn is a user message (tool results first, then your words), plus a mid-conversation system message for the
   program's notes. Your words and the program's notes are never in the same block.
5. **Body preemption:** the first body action of a newer reply drops what older replies left and stops the motion in
   progress. A reply that makes no movement (e.g. only `stop`) lets nothing new start.
6. **Body to brain:**
   - Finished and cut-short actions become notes for the next turn, with the claw position.
   - A refused or failed action wakes the brain at once, even mid-reply, which cuts that reply off and restarts it
     knowing about the failure. A `move_to` outside the box is refused as soon as the call arrives instead, so the
     reason goes back in its tool result and the reply isn't cut off (added 2026-11-07, after Haiku tried z = 350 and
     260 for "how tall can you reach?"; the top is 231).
   - ~~After `IDLE_S` (120 s) without a message, the brain is woken once, so it can doze off.~~ Replaced in Phase 2.7
     by idling in the body, with no API calls.
7. **Safety stays in code.**
   - Ctrl-C stops the arm, cancels the reply and quits, without an LLM round trip.
   - Every limit is still checked by `motion.py` when an action runs.
   - A stop holds the claw at its *measured* pose. Otherwise the servos carry on to the last streamed target, which
     trails the claw by up to ~75 mm at speed.
8. **Refusals:** Haiku 5.5 has safety classifiers but no server-side fallback. A refused reply is dropped, not resent,
   and a note tells the brain about it. (On Opus 5.5, `fallbacks: "default"` could be turned back on.)
9. **The API key lives in `.env`** (`ANTHROPIC_API_KEY=...`), which is git-ignored and readable only by its owner.
   `arm.py` reads it at start; a key already set in the shell wins.

**Tested (2026-11-07), without the real API or arm:** a scripted fake API (real SDK, `httpx2.MockTransport`, server-sent
events with delays) and the fake serial class were used.
- Checked: cut-off replies and their notes, preemption, `stop`, a refused move waking the brain mid-reply, a bad
  gesture name waking it after the reply, and Ctrl-C mid-reply.
- Every request's history was checked: each `tool_use` is answered first in the next user message, and a system
  message is never followed by a user message.
- The whole UI ran through `main()` with piped keystrokes.
- The fake API answered in 0.3-0.7 s; real latency is still to be measured.

**Stopping on the real arm (2026-11-07)**, via `Body.stop()` part way through excited, nod and a `look_at` turn:
- The body was free about 0.12 s after the stop.
- From then the claw drifted 4-17 mm, against up to ~75 mm if it had carried on to the last streamed target.
- Preemption also worked: `happy` was cut at 0.7 s by `shake`, which played from that mid-gesture pose and ended within
  1 mm of where it started.

**Live sessions (2026-11-07, real API and arm).** Three scripted sessions of about 25 messages each were run by piping
keystrokes into `main()` (scratch script, not in the repo).
- **Latency:** the first action came 0.7-1.6 s after each message (median ~0.95 s). Each reply was 50-240 output
  tokens. The prompt cache was read on every turn, so only a few hundred new tokens were sent per reply.
- **Sessions 1-2 (with `say`) found the issues** fixed above:
  - the stuck LED, and the wander away from rest;
  - speech printed twice (Haiku wrote text *and* called `say`);
  - "look at me" barely moving from rest: `look_at` now leans 35 mm toward the point for a moment;
  - gestures shrunk to 0.5-0.7x;
  - a chatbot tone: a question every turn, explaining itself, once inventing a reason about its body;
  - never going back to rest;
  - the idle note reading "0 minutes".
- **Session 3 (wordless).** Every reply was body language, and none was text. Its choices:

  | Message | It did |
  |---|---|
  | "work was rough" | lean_in, soft light |
  | "that helps" | calm |
  | job news | excited |
  | a joke | laugh |
  | "pick up the pen" | confused |
  | 17 × 23 | tilt |
  | "want to play?" | play_bow |
  | "useless" | sad, dim light |
  | an apology | shy |
  | "goodnight" | sleep |

  - A refused move (z above the box) woke it, and it retried inside the box at once.
  - It went back to rest by itself 23 s after its last move.

**Not yet checked:**
- Whether lean_in, calm and yawn look smooth at their slow speeds.
- `PERSON` (600, 0, 300) is a guess at where the user sits. Measure it.

## Phase 2.6: speech in, `ears.py` (built 2026-11-07)

Always listening: what it hears goes to `Brain.hear` exactly like typed text, so speaking cuts off a reply in progress
the same way typing does. Typing still works, for debugging. The arm still answers only with its body.

```
Blue Snowball (48 kHz, via PipeWire) ─► 16 kHz ─► Silero voice detector (32 ms frames)
   speech starts ─► screen: (listening...)
   0.5 s of quiet ends it ─► faster-whisper large-v3-turbo on the RTX 3060 Ti ─► screen: heard> ... ─► Brain.hear
```

**Decisions:**
1. **Local, not a cloud service:** no second account or bill, nothing from an always-on mic leaves the PC, and it is
   fast enough. `large-v3-turbo` in float16 loads in ~2.5 s (once warm) and transcribes a short sentence in ~0.25 s.
   It is downloaded once (~1.6 GB) to `~/.cache/huggingface`.
2. **Whisper only ever sees what the voice detector calls speech.** Given 3 s of room silence it wrote "Thank you."
   with a no-speech probability of 0.0, so that probability is useless as a filter. Also dropped:
   - blips with under 0.3 s of speech (a clunk, a cough), without transcribing them;
   - transcripts with a mean log-probability under -1.0 (the invented "Thank you." scored -1.08; real speech scored
     -0.04 to -0.45).
3. **No wake word.** It hears everything, and the system prompt says not all of it is meant for it (TV, phone calls):
   let it go by, or glance over. It's also told words are machine-transcribed and may be misheard: tilt or look
   confused rather than guess. (Asleep, only its name wakes it: Phase 2.8, below.)
4. **The Snowball only opens at 48 kHz through PipeWire,** so every 3 samples are averaged down to 16 kHz.
5. **The CUDA libraries come from pip** (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`); `ears.load_cuda_libs` loads
   them, since they are not on the library path.
6. **If the ears fail** (no mic, no GPU), the screen says so and typing still works.

**The transcript log, for checking test runs.** Each run writes `logs/<start time>/`:
- `transcript.jsonl`: one line per event, `t` in seconds from the start: `heard` (text, speech length, Whisper
  time, log-probability), `dropped` (why), `speech` (started talking), `typed`, `arm` (each action as queued, and how
  far into the reply), `idle` (each fidget, Phase 2.7), `note` (what the program told the brain), `wrote`, `reply`
  (latency, tokens, cache), `cut`, `error`, `start` and `stop`.
- Each utterance was also saved as a .wav at first, for checking. Removed after the first real test (2026-11-07) went
  well, to save disk space.

**Tested (2026-11-07):**
- Synthetic speech (espeak-ng) fed straight to the detector: 4 phrases with noise and a hiss burst between them came
  out word for word, and the hiss was ignored. 12 s of audio took 1.3 s to process.
- End to end with a fake mic stream (the same phrases in real time), the real API and a fake arm:

  | Heard | It did |
  |---|---|
  | "Hey buddy, I'm home." | look_at, then excited |
  | "Do you want to play?" | play_bow |
  | "I had a really rough day at work." | lean_in |
  | "Don't boy." (espeak's "Good boy.", misheard) | tilt |

  - From the end of speech to the first move took ~2 s: 0.5 s of silence wait, 0.25 s of Whisper, and 0.8-1.3 s of
    Haiku.
- The real Snowball opened and ran without a false trigger in a quiet room.

**First real test by the user (2026-11-07, `logs/2026-11-07_20-48-38`):** 13 utterances, all transcribed word for
word (log-probability -0.03 to -0.35), one blip dropped as too short, and no false triggers from the arm's own motors
while it moved. Asked to sleep and then to wake up, it did both.

## Phase 2.7: idle life (built 2026-11-07)

Between inputs the arm used to sit still. Now the body thread idles by itself whenever it has nothing to do:

| Quiet for (since the brain last moved it) | Mood | Fidgets, picked at random 2-5 s apart (`IDLE_GAP_S`) |
|---|---|---|
| 0 s | content | first back to rest, then glance around, tilt, calm |
| 30 s | bored | glance, look at the person, beg, curious, play_bow, back to rest (no yawn: too many, said the user) |
| 2 min | sleepy | yawn, calm, glance, back to rest |
| 4 min | asleep | the `sleep` gesture, then nothing until the brain moves it |

It never just sits there: idling starts right after the brain's last movement, and nothing in the list is "stay
still". (First built with 20 s / 2 / 5 / 8 min, 6-18 s gaps and "stay still" fidgets; the user found that too
static, 2026-11-07.)

**Decisions:**
1. **No LLM while idle.** Fidgets are picked locally from `MOODS`, so idling costs no API calls (the old `IDLE_S`
   wake-up of the brain is gone). The brain gets one note per mood change ("Nothing from your person for 2 minutes:
   you are getting bored, ..."), not one per fidget, so its mood carries into the next reply. The content mood is not
   noted, since it starts after every reply.
2. **The brain always wins.** Idle moves are queued like a cut-off reply (`gen = None`), so the first movement of any
   reply stops them. Only the brain's own movements reset the quiet clock: if it decides speech isn't for it (the
   TV), the boredom carries on.
3. **Someone starting to talk** (the ears' `speech` event) stops an idle move at once. The arm perks up where it
   faces, without turning (the user's call): `gestures.listen` leans out to 50 mm past the resting pose (less near
   the box edge) and lifts to near the top of the box, and holds there. It is a set distance, not a lean from
   wherever the claw is: at first it leaned 50 mm further each time, and in a run of questions (one perk per
   question, no time to go home between) the claw walked out 162, 213, 263, 312 mm. Now it holds 211 mm, and a claw
   left farther out comes back to it. The brain's first movement replaces it ~2 s after the person stops. Idling waits `HUSH_S`
   (8 s) from the start of speech, then starts again by going home, e.g. after a cough the ears drop. No perk while
   asleep, or while the brain's own movements play. Tested on the real arm (2026-11-07): from a low glance it rose
   from z 137 to 223.
4. **Waking is in code:** if it is asleep, the first movement of a reply plays `wake_up` first. In testing, Haiku
   ignored the prompt's "wake_up first" and played `excited` from the low sleep pose. If the brain doesn't move at
   all, it sleeps on.
5. **Glances aim at a random point -30° to 40° around the base.** From rest the claw is only 161 mm from the base
   axis, so wider turns leave the box (x ≥ 117.6, y ≥ -102.4): -60° to 75° would have refused 45% of glances in
   simulation, and -30° to 40° refused none of 3000.

**Tested (2026-11-07):**
- On the real arm, with the moods sped up (rest at 3 s, bored at 25 s, sleepy at 55 s, asleep at 80 s): it went
  home, then tilt, calm, yawn, look, play_bow, yawn, glance, calm, then fell asleep and stayed asleep. A brain `nod`
  mid-idle replaced the fidget and reset the clock. `hush` cut `calm` short.
- End to end with the real API, a fake mic and a fake arm (sped up): mood notes reached the brain; nothing idled
  between speech and the reply; "Hey buddy, wake up!" after it fell asleep played `wake_up`, then the brain's `happy`.

- On the real arm at the real timings: after a `look_at`, idling began ~5 s later; bored at 30 s, sleepy at 2 min,
  asleep at 4 min, and never motionless for more than 5 s before that.

**Not yet checked:** whether it feels right to the user. The timings and fidget lists are knobs at the top of
`arm.py`.

## Phase 2.8: asleep, only its name wakes it (built 2026-11-07)

Its name is **Robert** ("Rob" for short), chosen by the user. Asleep, whether it dozed off from boredom (Phase 2.7)
or was told to sleep, it hears nothing but its name:
- Speech is still cut out by the voice detector and transcribed locally, since that is the only way to hear the name.
  But unless the text has the word "Robert" or "Rob" (`arm.NAME`; not "robot", "problem", "Bob" or "Roberta"), it
  is dropped: not shown, no text logged (only `dropped`, why "asleep"), and no API call. No perk-up either.
- On its name, `Body.wake` plays `wake_up` at once, and the brain gets what was said plus a note that it woke. The
  brain's replies queue after `wake_up` rather than cutting it off (only `stop` and Ctrl-C cut it), and a `wake_up`
  the brain asks for itself is skipped (Haiku asked for one anyway, so it played twice). At first only the first
  reply waited: in the user's test, "Rob" then "wake up" 1.5 s later cut `wake_up` mid-stretch and left the claw
  410 mm out for the rest of the conversation.
- Typed messages still always get through, for debugging.
- `ears.HOTWORDS` hints the name to Whisper, only while asleep. Without it, a short "Rob" in a fast synthetic voice
  came out "Hey mom!" or "Rub.".
- **Made more sensitive (2026-11-07).** The user had to shout the name slowly, several times. The cause: a quick
  "Rob" is only 0.13-0.29 s of voice, under `MIN_SPEECH_S` (0.3 s), so the ears dropped it as a clunk before
  transcribing it. The voice detector caught every one, even at 3% volume, and Whisper never misheard one. Now
  the ears transcribe short sounds too (still dropped when awake), and asleep, the name in *anything* transcribed
  wakes it, short or unsure. The asleep log line now keeps why the ears would have dropped it, and how long it was
  (still no text).
  - Measured on synthetic voices (4 voices, 3 speeds, 2 volumes): the hint catches 95% of short names (74%
    without), but it also turns 14% of isolated single words ("Rub it.", "Robot.", "Problem.") into "Rob", and
    for a faint word it can write the hint itself ("Robert, Rob"). Whole sentences never woke it, and no clunk did.
    So the hint is on only while asleep: awake, chat is transcribed without the bias.
  - End to end: asleep, a 0.26 s "Rob!" at 300 words per minute woke it; two sentences without the name did not.

**Tested (2026-11-07):**
- Synthetic speech (espeak, 3 voices) through the ears: 13 phrases each, 7 with the name. With the hint, every
  "Robert" and "Rob" was caught, and "robot", "problem", "Bob" and others weren't; one "Roberta" (British voice)
  came out "Robert" and would wake it. Without the hint, 2 of the "Rob"s were missed.
- End to end (real API and Whisper, fake mic and arm): "it is late, go to sleep" gave yawn then sleep. "Did you see
  the game last night?", "Hey Bob, how was work?" and "The robot is asleep." were ignored, with no API call (4
  replies in the whole run). "Robert, wake up!" gave `wake_up`, then the brain's `happy`.

- The user's own voice (2026-11-07): "go to sleep", then three sentences ignored, then "Rob" woke it. Robert also
  nodded to "Do you have a name?" and got excited at "is it Rob?".

**Not checked:** "Rob" from across the room.

**Still possible later:** a dedicated wake-word model (e.g. openWakeWord, trained on "Robert") would let Whisper rest
while it sleeps. For now Whisper runs only on speech (~0.25 s each) and the cost is negligible.

## Phase 2.9: dancing to music, `dance.py` (built 2026-11-08)

Ask it to dance ("Rob, dance to this song!") and the brain calls the `dance` tool. For 30 s the mic goes to a beat
tracker instead of Whisper (`Ears.tap`): nothing is transcribed and the brain sits it out. Then everything goes back to
normal, and the brain is told how it went. With no steady beat it does not dance; the failure wakes the brain, which
can look confused. There is no way to stop it early except typing (a new reply) or Ctrl-C.

- **Beat tracking, numpy only.**
  - Onsets: the rise in loudness from one 16 ms slice to the next, summed per octave.
  - Tempo: the autocorrelation of the onsets, scored at 1, 2, 3 and 4 beats, within 60-180 BPM, favouring tempos
    near 120.
  - Phase: where a comb of beats lines up best.
  - Fixes found on synthetic songs (rock, four-on-the-floor, drumless; 72-174 BPM):
    - Summed flat over frequencies, off-beat hi-hats beat the kick, so it often locked on the off-beat.
    - Without scoring the multiples, it often picked 1.5 or 2/3 beats.
    - Without blurring the onsets, beats that fall between two slices lined up badly.
  - With all three fixes, every tempo and phase was right, to within 15 ms.
  - Above 140 BPM it bobs on every other beat, on the stronger ones.
- **Staying in sync.** The dance keeps its own beat clock, re-finds the beat every 0.5 s in the last 8 s, and nudges the
  clock's speed by up to 10% toward it, so the claw never jumps. Estimates are ignored when they jump to double or half
  the tempo, or by over a quarter beat (snare instead of kick); without that, songs heard at half tempo drifted
  ±150 ms. On the fake arm the bottoms of the bobs landed within ±25 ms of each other.
- **What tells music from everything else** is how well the beat repeats (`MIN_STRENGTH` 0.25):
  - Music through the real mic: 0.48-0.55.
  - Speech: up to 0.17. Noise and random knocks: up to 0.12.
  - Loudness tells nothing: the room alone is about as loud at the mic as music at half volume.
  - At 20% system volume the song did not get through.
- **The bob swings the whole arm with the claw level,** as the user asked. They posed it by hand at rest, low (79, 122)
  and high (214, 230), and the claw is level in all three. The arm's two links were fitted to joint readings at 8 poses
  (238.9 and 279.9 mm, 0.14 mm rms).
  - With the forearm held level, the claw moves like the elbow: on a circle around a point a forearm's length ahead of
    the shoulder. So the dance swings the upper arm from -16° (the high pose) toward -57° (the low pose).
  - The bottom is as low as the drawn box allows: about -42°, at x 120 (the box starts at x 117.6). The user's low pose
    is outside the box.
  - The bob is shallower at faster tempos, to stay under 90% of MAX_SPEED.
  - On the real arm the claw stayed at 88-92° (90 = level). The bob runs diagonally, down and back: at 70 BPM, 45 mm
    in height and 83 mm in reach.
  - The earlier straight-down bob was mostly the elbow, and read as a nod.
- **Lag.** The claw follows its targets 250-370 ms late (about 330 ms at 100 BPM). It behaves like a fixed delay:
  servo acceleration from 20 up to 250 changed nothing. So the dance aims `LAG_S` 0.3 s ahead.
- **First real-song test** (the user, Dancing Queen, ~101 BPM): it found 99 BPM, but the bobs looked slow and late. The
  smooth wave had no moment to see as the beat, and the claw only swung 14-26 mm of the 37 mm asked.
  - The bob became a bouncing ball, `1 - |sin|`: a sharp turn at the bottom on the beat, with the same peak speed.
  - On the real arm, with the song fed straight in (no mic), the bottoms then landed ~65 ms early (spread -127..-3 ms),
    and the claw swung 27-38 mm.
  - The mic's delay (10-50 ms, not measured: PortAudio reports 10-30) should bring that to on the beat or just before.
- **`Arm.follow(pose)`** streams targets made on the fly at 50 Hz, each checked against the box just before it is sent.
- **Real test (arm, mic, a synthetic 116 BPM song through the speakers at 50%):**
  - It found 116 BPM and danced 20 s.
  - All 40 re-checks agreed, despite its own motor noise.
  - Nothing was transcribed.

- **Dancing while listening (2026-11-08).** Real songs from the user's phone, held to the mic, mostly failed the old
  test: listen 6 s, then dance only at strength 0.25 or more. Their strengths were 0.11-0.22, yet the guessed tempos
  were often right (101 BPM for Dancing Queen at 0.11). Phone-like synthetic audio (band-passed 350 Hz-7 kHz,
  squashed) still scored 0.3-0.7, so the synthetic songs are too clean to tune the threshold with. So, as the user
  suggested, it no longer judges one guess:
  - It listens 3 s, then dances on its first guess, and guesses again every 0.5 s from the last 6 s.
  - The tempo is the one the last 12 guesses vote for, weighted by strength. Double or half tempo count as the same
    tempo, since they make the same bobs.
  - It changes tempo only when outvoted 1.5:1. Without that, a drumless song flipped between 119 and 80 BPM.
  - The phase is the agreeing guesses' beats averaged on a circle, so a kick and a snare cannot pull it both ways.
  - The bob's depth follows how much the guesses agree: from 30% (speech or noise, so the bob is shallow and wanders)
    to full at a confidence of 0.15.
  - It no longer refuses at all, except with the ears off.
  - A crash in one real dance: an interpolation step could throw the beat period negative. It is now clamped.
  - Fake arm, synthetic audio:
    - Phone-like songs at 92-158 BPM locked within 1-3 bobs, landing 20-50 ms early (as aimed).
    - 5 s of talking and then music: it started at a wrong 97 BPM, and was on the beat within 4 bobs.
    - Speech only: it bobbed 10-18 mm, wandering.

## Phase 3: camera and perception (do not build yet)

- A camera mounted on the arm. A perception worker (a thread or process) owns the camera, runs open-vocabulary detectors
  (YOLO-World, OWL-ViT, Grounding DINO), keeps the latest scene and emits debounced events.
- Tool results gain an observation package: an optional downscaled image, plus a JSON scene of entities with labels and world positions in mm.
- Depth comes from a table-plane assumption first (cheapest), via a one-time hand-eye calibration that uses the T:105 pose.
- The tool interface stays stable: Phase 3 only enriches results and adds a `look()` tool.

## Constraints

- Python 3.10+ (the venv runs 3.14). The dependencies are `pyserial`, `mcp` (2.x), `anthropic` (2.x),
  `prompt_toolkit`, `faster-whisper`, `sounddevice` and the CUDA 12 libraries from pip (see `requirements.txt`).
  PortAudio comes from the system (Fedora: `portaudio`). faster-whisper's own `decode_audio` breaks with PyAV 19,
  but `ears.py` never uses it.
- Keep each layer small and readable. The user prefers the least code that does the job.
- Keep secrets out of the code. Don't print to stdout in anything the MCP server imports.
