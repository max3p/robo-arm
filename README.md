# Robert, a robot arm pet

A Waveshare RoArm-M2-S desk robot arm that acts like a clever pet dog. You talk to it, Claude decides how it
reacts, and it answers only with body language: nods, head tilts, a play bow, leaning in, the claw and its light.

- **Listens all the time:** speech is turned into text locally on the GPU (faster-whisper). Nothing is recorded.
- **Moves while it thinks:** each movement starts as soon as Claude has written it, and speaking cuts off a reply
  in progress.
- **Has a life of its own:** between conversations it glances around, gets bored, then sleepy, and falls asleep.
  This runs locally, with no API calls.
- **Asleep, it ignores everything except its name:** "Robert" or "Rob" wakes it.
- **Stays in a safe box:** a person draws the space the arm may use by hand, and every move is checked against it
  in code before the arm moves.

The full design notes, decisions and test results are in [project-v2.md](project-v2.md).

## What you need

- A Waveshare RoArm-M2-S on USB (`/dev/ttyUSB0`) with its 12 V supply. Your user must be in the `dialout` group.
- Linux with an NVIDIA GPU (CUDA 12) for speech to text. The first run downloads the Whisper model (~1.6 GB).
- A microphone. It is set up for a Blue Snowball; change `MIC` at the top of [ears.py](ears.py).
- PortAudio from your system packages (Fedora: `portaudio`).
- A Claude API key.

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env && chmod 600 .env    # .env is git-ignored
```

Then draw the box the arm may move in. Run `python cli.py`, free the arm and sweep the claw around the space it may
use. The bounds are saved to `arm_state.json`.

## Running it

```
.venv/bin/python arm.py     # talk to Robert (or type). Ctrl-C stops the arm at once and quits
.venv/bin/python cli.py     # human control: draw the bounds, teach paths, test moves and gestures
```

- **Opening the serial port resets the board.** The arm moves to its boot pose and then to its resting pose
  (about 10 s), so keep the space clear each time.
- **Only one program can use the arm at a time.** If Claude Code has the MCP server connected, disconnect it first.
- **What appears in the terminal is for debugging only.** The arm can't talk, so the screen shows what it heard
  and what it did.
- **Every run is logged** to `logs/<start time>/transcript.jsonl`. This holds what it heard, what it did, its moods
  and each reply's timing and token use.

## Files

| File | What it does |
|---|---|
| [arm.py](arm.py) | The pet. The brain (Claude, streaming tool calls) and the body thread (actions, idling, moods, sleep). Settings at the top: `MODEL`, `MOODS`, `NAME`, `PERSON` |
| [ears.py](ears.py) | Speech in. The mic, a voice detector and Whisper on the GPU |
| [gestures.py](gestures.py) | Body language as keyframes: nod, shake, happy, lean_in, sleep, wake_up, look_at and more |
| [motion.py](motion.py) | Safe motion on the arm: straight-line moves, sag correction, gestures, stopping within 20 ms. All limits are checked here |
| [cli.py](cli.py) | Human control: draw the bounds, record and play paths, try gestures |
| [mcp_server.py](mcp_server.py) | The same arm as MCP tools, for Claude Code or Claude Desktop (see below) |
| `arm_state.json` | The drawn bounds and taught paths. Only `cli.py` writes it |

## Using it from Claude Code (optional)

```
claude mcp add robo-arm -- $PWD/.venv/bin/python $PWD/mcp_server.py
```

Claude then gets tools to move the arm, play gestures, look at points and play taught paths. It's limited to the
same box.

## Safety

- **The box is enforced in code, never only in a prompt.** Moves outside it are refused before anything moves.
  So are moves out of reach and paths that pass too close to the base.
- **The box limits the claw tip only, not the elbow or forearm,** so draw it smaller than the real free space.
- **Speed is capped** at 500 mm/s.
- **Ctrl-C stops the arm without asking the LLM.** The arm holds where it is.
- **The power switch is the real emergency stop.**

## Next

Phase 3 adds a camera on the arm, so Robert can see the room and its person.
