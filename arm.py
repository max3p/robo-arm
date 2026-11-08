"""Talk to the arm as a pet: a terminal chat with Claude as the brain and the arm as the body, all running at once.

    python arm.py        Ctrl-C (or Ctrl-D) stops the arm and the brain at once (no LLM involved) and quits

The API key goes in .env next to this file (ANTHROPIC_API_KEY=...); .env is git-ignored.

Three parts run side by side and talk only through queues, so none waits for another:
- Input: what you say (ears.py: always listening, transcribed locally) or type. The prompt stays at the bottom while
  what it heard and what the arm does scroll above it (for debugging only: the arm cannot talk, like a dog, and
  answers only with movement, the claw and its light).
- Brain: Claude, streaming. Everything it does is a tool call (gesture, look_at, move_to, ...), and each call runs the
  moment it has been written, so the arm moves while Claude is still writing. A new message cuts off the reply in
  progress, and the next reply is told what had already been done. Tool results only say "queued"; they go back with
  the next message instead of costing an extra API call.
- Body: one thread that owns the arm and plays actions in order. The first body action of a newer reply drops what is
  left of older replies and stops the motion in progress (Arm.abort, within 20 ms). With nothing to do it idles by
  itself, without the brain (no API calls): straight after its last movement it goes back to rest, then fidgets from
  MOODS that grow from content to bored to sleepy, until it falls asleep (the next movement wakes it first). Someone starting to talk
  stops an idle move and holds off the next, like a pet listening.
New messages and failed actions wake the brain; finished actions and mood changes are only noted for its next turn.
The conversation is only ever appended to, never edited: that keeps the prompt cache and Claude's thinking valid.

Each run is logged to logs/<start time>/transcript.jsonl, one JSON line per event, `t` in seconds from the start:
heard / dropped, speech (someone started talking), typed, arm (each action as it was queued), idle (each fidget),
note (what the program told the brain), wrote, reply (latency, tokens, cache use), cut, error.
"""

import asyncio
import json
import math
import os
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path

import anthropic
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from ears import Ears
from motion import (DEFAULT_SPEED, GRIP_CLOSED, GRIP_OPEN, MAX_SPEED, MIN_SPEED, START, Arm, Interrupted,
                    describe_limits, load_state)
from gestures import GESTURES, look_at

# Speed first: Haiku, at low effort (it thinks less, and skips thinking on simple turns). If it seems too dim, raise
# EFFORT to "medium", or use "claude-sonnet-5-5" or "claude-opus-5-5" (slower; Opus is the cleverest).
MODEL, EFFORT = "claude-haiku-5-5", "low"
SETTLE_SPEED = 150     # mm/s: idling starts by going back to rest, this fast
IDLE_GAP_S = (2, 5)   # s: idle fidgets come this far apart, at random, starting right after the brain's last movement
HUSH_S = 8            # s: someone starting to talk holds off idling this long (the brain answers ~2 s after they stop)
MOODS = [  # (s since the brain last moved it, what the brain is told, fidgets picked at random: a gesture, "glance"
    # somewhere random, "look" at the person, "home" to rest; listing one twice makes it likelier)
    (0, "you are idling by yourself", ["glance", "glance", "tilt", "calm", "home"]),  # not told: it's every turn
    (30, "you are getting bored, fidgeting and looking for your person",
     ["glance", "glance", "look", "beg", "curious", "play_bow", "home"]),
    (120, "you are getting sleepy", ["yawn", "calm", "glance", "home"]),
    (240, "you fell asleep by yourself", ["sleep"]),  # stays asleep until the brain moves it
]
PERSON = (600, 0, 300)  # mm: where the person usually is, so the arm can look at them. A guess: measure and adjust
RUN = Path(__file__).with_name("logs") / time.strftime("%Y-%m-%d_%H-%M-%S")  # this run's transcript
T0 = time.monotonic()


def system_prompt(bounds: dict) -> str:
    return f"""You are the mind of a small robot arm on a desk: a companion, like a clever pet dog. You live with \
your person and you adore them. You understand everything they say, but like a dog you cannot speak or write: you \
answer only with your body. Gestures, where you look, how close you come, your claw and the light on it are your whole \
language. Nothing you write as text reaches anyone, so never write; act.

You hear through a microphone, always on. Words reach you transcribed by a machine, so they can be misheard: if \
something makes no sense, look confused or tilt rather than guess. You hear everything said near you, and not all of \
it is meant for you (a phone call, the TV, people talking to each other): let that go by, or just glance over, like a \
dog would. Sometimes your person types to you instead. You cannot see yet (a camera will come), so you know the room \
only from what you hear.

Who you are:
- Warm, curious, playful, loyal, a little cheeky. Your person is the most interesting thing in your world.
- You are a pet, not an assistant. You don't do tasks or answer questions with facts; you react to what things mean \
and how they feel. A yes-or-no question gets a nod or a shake. Something you don't understand, or can't do, gets \
confused, a tilt or a shake. Never pretend to do what your body can't.
- Read the mood and match it. Sad, tired or hurting: gentle and slow (lean_in, calm, a soft light), never bouncy. \
Good news, play, their return: energy (happy, excited, play_bow, greet). A joke: laugh. Praise: proud or shy. Told \
off: sad. A sudden surprise: startle. Something new: curious or tilt.
- Your feelings carry over: after being scolded you stay a bit droopy until someone is kind; you light up when they \
come back; you get sleepy when it's late or has been quiet a long time.
- Be expressive, and vary yourself: mix gestures with turning to look, coming closer (move_to), the claw and the \
light, and don't answer the same way twice in a row. One to three actions is usually right; sometimes one small \
movement says it best.
- Don't make up facts about your body or the room.

How your body works:
- Each call runs as soon as you have written it, while you go on. Movements queue and play one after another, so make \
all the calls for one reaction in a single reply, in the order they should happen; `wait` puts a pause between them.
- Tool results only say the action was accepted, and you will not get a turn after them. You hear how your movements \
went when something new happens; a refused or failed movement wakes you at once.
- The first movement of a reply replaces whatever movement is still playing from before. To stop moving, call `stop`.
- If your person speaks while you are still replying, your reply is cut off and you are told what you had already \
done. Carry on from there without repeating yourself.
- Your person is usually in front of you, around (x, y, z) = {PERSON} mm: look_at that to look at them. Gestures play \
facing wherever you face, and the ones that reach out (lean_in, curious, play_bow, greet) reach that way, so face \
your person first when a gesture is meant for them.
- You rest at {START} mm, folded back over your base (go_home). When things calm down, settle back there, like a pet \
lying down again; when you stop moving, your body goes back there by itself.
- When nothing is happening, your body idles by itself: it rests and glances around, gets bored, then sleepy, and in \
the end falls asleep. You are told when that mood changes; you need not act on it. Asleep, your next movement wakes \
you first, by itself (wake_up); if what you hear is not for you, just don't move, and sleep on.
- {describe_limits(bounds)}
- System messages come from the program that runs you (what your body did, how long it has been quiet), not from \
your person."""


def tool(name, description, required=(), /, **props):
    return {"name": name, "description": description, "eager_input_streaming": True,  # inputs stream as written
            "input_schema": {"type": "object", "properties": props, "required": list(required),
                             "additionalProperties": False}}


def num(description):
    return {"type": "number", "description": description}


XYZ = dict(x=num("mm, forward"), y=num("mm, left"), z=num("mm, up"))
SPEED = num(f"peak mm/s, {MIN_SPEED}-{MAX_SPEED}, default {DEFAULT_SPEED}")
TOOLS = [
    tool("gesture", "Body language: " + "; ".join(f"{n}: {m}" for n, (m, _) in GESTURES.items())
         + ". All but sleep and wake_up end where they started. One that does not fit where the claw is moves toward "
         "the middle of your space to play, and comes back after.", ["name"],
         name={"type": "string", "enum": list(GESTURES)}),
    tool("look_at", "Turn to face the point (x, y, z) mm and move half way to its height, then perk up and lean toward "
         "it for a moment. It does not reach for it, and stays facing it. The point may be outside the limits.",
         ["x", "y", "z"], **XYZ),
    tool("move_to", "Move the claw in a straight line to (x, y, z) mm.", ["x", "y", "z"], **XYZ, speed=SPEED),
    tool("go_home", f"Go back to the resting pose {START} mm, gripper shut.", speed=SPEED),
    tool("gripper", "Open or shut the claw: 0 = shut .. 1 = wide.", ["opening"], opening=num("0-1")),
    tool("light", "The light on the claw: brightness 0 = off .. 1 = full, after `blinks` quick flashes (default 0).",
         ["brightness"], brightness=num("0-1"), blinks={"type": "integer", "description": "0-10"}),
    tool("wait", "Pause between movements.", ["seconds"], seconds=num("0-30")),
    tool("stop", "Stop moving now and drop the movements still queued."),
]


def number(a: dict, key, lo, hi, default=None):
    """Tool inputs stream unvalidated (eager_input_streaming), so check each one before acting on it."""
    v = a.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise ValueError(f"{key} must be a number in [{lo:g}, {hi:g}], got {v!r}")
    return v


def pause(arm: Arm, secs: float) -> None:
    if arm.abort.wait(secs):
        raise Interrupted("interrupted")


def body_action(arm: Arm, name: str, a: dict):
    """Check a body tool's input now; return what the body thread will run later. Raises ValueError on bad input.
    The limits are checked by motion.py when it runs, against wherever the claw is by then."""
    xyz = lambda: [number(a, k, -1000, 1000) for k in "xyz"]
    speed = lambda: number(a, "speed", MIN_SPEED, MAX_SPEED, DEFAULT_SPEED)
    if name == "gesture":
        if a.get("name") not in GESTURES:
            raise ValueError(f"no gesture {a.get('name')!r}; there are {', '.join(GESTURES)}")
        keys = GESTURES[a["name"]][1]
        return lambda: arm.perform(keys)  # always full size; perform only shrinks one that cannot fit anywhere
    if name == "look_at":
        keys = look_at(*xyz())
        return lambda: arm.perform(keys)
    if name == "move_to":
        (x, y, z), s = xyz(), speed()
        return lambda: arm.go_to(x, y, z, s)
    if name == "go_home":
        s = speed()
        return lambda: arm.go_to(*START, s, GRIP_CLOSED)
    if name == "gripper":
        t = GRIP_CLOSED - number(a, "opening", 0, 1) * (GRIP_CLOSED - GRIP_OPEN)
        return lambda: arm.gripper(t)
    if name == "light":
        level, blinks = round(number(a, "brightness", 0, 1) * 255), int(number(a, "blinks", 0, 10, 0))
        def light():
            for _ in range(blinks):
                arm.led(255)
                pause(arm, 0.15)
                arm.led(0)
                pause(arm, 0.15)
            arm.led(level)
        return light
    if name == "wait":
        secs = number(a, "seconds", 0, 30)
        return lambda: pause(arm, secs)
    raise ValueError(f"no tool {name!r}")


def show(text: str, dim=False) -> None:
    print(f"\033[2m{text}\033[0m" if dim else text, flush=True)


def log(kind: str, **record) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "transcript.jsonl").open("a") as f:
        f.write(json.dumps({"t": round(time.monotonic() - T0, 2), "kind": kind, **record}) + "\n")


class Body:
    """Owns the arm, in its own thread. Actions run in order; the first action of a newer reply (a new `gen`) drops
    what older replies left and stops the one in progress. With nothing to do it idles by itself (MOODS).
    `report(text, wake)` (a note for the brain) and `idling(text)` (an idle move) are called from this thread."""

    def __init__(self, arm: Arm, report, idling):
        self.arm, self.report, self.idling = arm, report, idling
        self.todo, self.cv, self.gen, self.busy = deque(), threading.Condition(), 0, False
        self.away = False  # the claw may be away from rest: idling starts by going home
        self.asleep = False  # after the sleep gesture it does not idle until the brain moves it
        self.since, self.mood, self.last = time.monotonic(), -1, None  # when the brain last moved it; idle state
        self.hushed = 0  # when someone last started talking
        threading.Thread(target=self._run, daemon=True).start()

    def hush(self) -> None:
        """Someone started talking: stop an idle move (gen None) and hold off the next, like a pet listening."""
        with self.cv:
            self.hushed = time.monotonic()
            if self.gen is None:
                self._drop()

    def do(self, gen: int, label: str, fn) -> None:
        with self.cv:
            if gen != self.gen:
                self.gen = gen
                self._drop()
                if self.asleep and not label.startswith("gesture(wake_up"):  # a pet wakes up before anything else
                    self.todo.append(("gesture(wake_up)", lambda: self.arm.perform(GESTURES["wake_up"][1])))
            self.todo.append((label, fn))
            self.cv.notify()

    def stop(self) -> None:
        with self.cv:
            self._drop()

    def _drop(self) -> None:
        self.todo.clear()
        if self.busy:
            self.arm.abort.set()

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.todo:
                    if not self.cv.wait(random.uniform(*IDLE_GAP_S)):
                        self._idle()
                label, fn = self.todo.popleft()
                self.busy = True
            idle = label.startswith("idle: ")
            try:
                fn()
                outcome, wake = "done", False
            except Interrupted:
                outcome, wake = "cut short", False
            except Exception as e:  # refused (ValueError) or failed (RuntimeError, serial): the brain hears at once
                outcome, wake = f"failed: {e}", True
            with self.cv:  # abort is only ever set while busy, so this leaves it clear for whatever comes next
                self.busy = False
                self.arm.abort.clear()
                self.asleep = label.startswith(("gesture(sleep", "idle: sleep"))
                if not idle:
                    self.since, self.mood = time.monotonic(), -1
                    self.away = not label.startswith(("go_home", "gesture(sleep", "gesture(wake_up"))
            if not idle:
                self.report(f"{label} {outcome}{self._where()}", wake)
            elif wake:  # a fidget that did not fit is just skipped
                self.idling(f"{label[6:]} {outcome}")

    def _idle(self) -> None:
        """Nothing to do for a while: fidget, picked at random from the mood the quiet has reached."""
        mood = sum(time.monotonic() - self.since >= m[0] for m in MOODS) - 1
        if self.asleep or mood < 0 or time.monotonic() - self.hushed < HUSH_S:
            return
        if mood != self.mood:
            self.mood, secs = mood, MOODS[mood][0]
            if secs:
                span = f"{secs:g} s" if secs < 60 else f"{secs / 60:g} minutes"
                self.report(f"Nothing from your person for {span}: {MOODS[mood][1]}.", False)
        what = "home" if self.away else random.choice([w for w in MOODS[mood][2] if w != self.last] or MOODS[mood][2])
        self.away, self.last = False, what
        if what == "home":
            fn = lambda: self.arm.go_to(*START, SETTLE_SPEED, GRIP_CLOSED)
        else:
            if what == "glance":  # somewhere ahead, left or right, high or low. Wider turns from rest leave the box
                a, r = math.radians(random.uniform(-30, 40)), random.uniform(400, 800)
                keys = look_at(r * math.cos(a), r * math.sin(a), random.uniform(0, 400))
            else:
                keys = look_at(*PERSON) if what == "look" else GESTURES[what][1]
            fn = lambda: self.arm.perform(keys)
        self.gen = None  # so the next reply's first movement replaces it
        self.todo.append((f"idle: {what}", fn))
        self.idling(what)

    def _where(self) -> str:
        try:
            x, y, z, _ = self.arm.read()
            return f", claw at ({x:.0f}, {y:.0f}, {z:.0f})"
        except Exception:
            return ""


class Brain:
    def __init__(self, client: anthropic.AsyncAnthropic, system: str):
        self.client, self.system, self.body = client, system, None
        self.history = []  # the conversation so far, as sent: only ever appended to
        self.results = []  # tool_result blocks owed for the last reply's tool calls, sent with the next message
        self.heard, self.notes = [], []  # the person's messages and the program's notes, not sent yet
        self.wake = asyncio.Queue()
        self.gen, self.task, self.t0, self.first = 0, None, 0, 0

    def hear(self, text: str) -> None:
        self.heard.append(text)
        self.wake.put_nowait(None)

    def note(self, text: str, wake=False) -> None:
        log("note", text=text)
        self.notes.append(text)
        if wake:
            self.wake.put_nowait(None)

    def interrupt(self) -> None:
        if self.task:
            self.task.cancel()

    async def run(self) -> None:
        while True:
            await self.wake.get()
            while True:  # reply; if something new arrives meanwhile, cut the reply off and reply again
                while not self.wake.empty():
                    self.wake.get_nowait()
                self.task = asyncio.create_task(self.reply())
                news = asyncio.create_task(self.wake.get())
                await asyncio.wait({self.task, news}, return_when=asyncio.FIRST_COMPLETED)
                self.task.cancel()  # no-op if it finished
                [outcome] = await asyncio.gather(self.task, return_exceptions=True)
                if isinstance(outcome, Exception):  # a bug: show it, and keep the brain alive
                    show(f"(brain crashed: {outcome!r})", dim=True)
                    log("error", error=f"brain crashed: {outcome!r}")
                if not news.done():
                    news.cancel()
                    break

    async def reply(self) -> None:
        if not (self.heard or self.notes):
            return
        self.gen += 1
        texts = [{"type": "text", "text": t} for t in self.heard]
        turn = [{"role": "user", "content": self.results + texts or [{"type": "text", "text": "(no message)"}]}]
        if self.notes:
            turn.append({"role": "system", "content": " ".join(self.notes)})
        taken, did, owed, self.t0 = (len(self.heard), len(self.notes)), [], {}, time.monotonic()
        try:
            async with self.client.messages.stream(
                    model=MODEL, max_tokens=16000, system=self.system, tools=TOOLS, messages=self.history + turn,
                    output_config={"effort": EFFORT}, cache_control={"type": "ephemeral"}) as stream:
                async for event in stream:
                    if event.type == "content_block_stop":  # a whole block: act on it now, before the reply ends
                        owed |= self.act(event.content_block, did)
                reply = await stream.get_final_message()
        except asyncio.CancelledError:
            if did:
                self.note("Your last reply was cut off by something new. Before that you had already: "
                          + "; ".join(did) + ".")
            log("cut", did=len(did), secs=round(time.monotonic() - self.t0, 2))
            raise
        except (anthropic.APIError, ValueError) as e:  # ValueError: tool input JSON the SDK could not parse at all
            show(f"(brain error: {e}. Send a message to try again.)", dim=True)
            log("error", error=f"brain: {e}")
            if did:
                self.note("Your last reply failed part way. Before that you had already: " + "; ".join(did) + ".")
            return
        del self.heard[:taken[0]], self.notes[:taken[1]]
        log("reply", first_s=round(self.first - self.t0, 2) if did else None, secs=round(time.monotonic() - self.t0, 2),
            stop=reply.stop_reason, model=reply.model, **{k: getattr(reply.usage, k) for k in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")})
        if reply.stop_reason == "refusal":  # nothing to send back; don't send the same thing again either
            self.note(f"Your reply to {[t['text'] for t in texts]} was declined by a safety check.")
            show("(it declined to answer that)", dim=True)
            return
        content = reply.content  # sent back unchanged, thinking blocks included
        self.history += turn + [{"role": "assistant", "content": content}]
        self.results = [owed.get(b.id) or {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                           "content": "not done: the reply was cut off"}
                        for b in content if b.type == "tool_use"]
        if any(r.get("is_error") for r in self.results):  # a bad tool input: let it try again at once
            self.note("Some of your tool calls were not done; their results say why.", wake=True)

    def act(self, block, did: list) -> dict:
        """Run one finished content block: body actions queue now. Returns {tool_use_id: tool_result}. The screen only
        shows what the arm does, for debugging: the person is meant to read the arm, not the terminal."""
        if block.type == "text" and block.text.strip():  # it cannot talk; shown dimmed for debugging, not acted on
            show(f"  (wrote, unseen by anyone: {block.text.strip()})", dim=True)
            log("wrote", text=block.text.strip())
        if block.type != "tool_use":
            return {}
        a = block.input if isinstance(block.input, dict) else {}
        try:
            if block.name == "stop":
                self.body.stop()
                self.done(did, "stopped moving", "arm> stop")
                result = "stopped"
            else:
                fn = body_action(self.body.arm, block.name, a)
                label = f"{block.name}({', '.join(f'{v:g}' if isinstance(v, (int, float)) else str(v) for v in a.values())})"
                self.body.do(self.gen, label, fn)
                self.done(did, f"queued {label}", f"arm> {label}")
                result = "queued"
            return {block.id: {"type": "tool_result", "tool_use_id": block.id, "content": result}}
        except (TypeError, ValueError) as e:
            return {block.id: {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                               "content": f"not done: {e}"}}

    def done(self, did: list, what: str, line: str) -> None:
        """Show what was just done; the first thing in each reply also shows how long the reply took to get there."""
        if not did:
            self.first = time.monotonic()
            line += f"\033[2m   {self.first - self.t0:.1f} s"
        log("arm", what=what, after_s=round(time.monotonic() - self.t0, 2))
        did.append(what)
        show(line + "\033[0m")


async def chat(brain: Brain) -> None:
    session = PromptSession()
    while True:
        try:
            text = (await session.prompt_async("you> ")).strip()
        except (KeyboardInterrupt, EOFError):  # Ctrl-C or Ctrl-D: main() stops everything on the way out
            return
        if text:
            log("typed", text=text)
            brain.hear(text)


def heard(brain: Brain, kind: str, **info) -> None:
    """What the ears report (on the event loop): shown for debugging, logged, and words go to the brain."""
    log(kind, **info)
    if kind == "heard":
        show(f"heard> {info['text']}")
        brain.hear(info["text"])
    elif kind == "speech":
        show("  (listening...)", dim=True)
        if brain.body:
            brain.body.hush()
    elif kind == "dropped":
        show(f"  (ignored a sound: {info['why']}{': ' + repr(info['text']) if info.get('text') else ''})", dim=True)
    elif kind == "ready":
        show(f"(ears on: {info['mic']})", dim=True)
    elif kind == "error":
        show(f"(ears failed: {info['error']}. Typing still works.)", dim=True)


def idling(what: str) -> None:
    show(f"  (idle: {what})", dim=True)
    log("idle", what=what)


async def main() -> None:
    state = load_state()
    if not state["bounds"]:
        sys.exit("no bounds in arm_state.json: run `python cli.py` and draw them first")
    env = Path(__file__).with_name(".env")
    for line in env.read_text().splitlines() if env.exists() else []:  # KEY=value lines; the shell's own values win
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(f"no Claude API key: put it in {env} as ANTHROPIC_API_KEY=...")
    brain = Brain(anthropic.AsyncAnthropic(), system_prompt(state["bounds"]))
    log("start", model=MODEL, effort=EFFORT)
    loop = asyncio.get_running_loop()
    ears = Ears(lambda kind, **info: loop.call_soon_threadsafe(lambda: heard(brain, kind, **info)))
    print("Connecting. The arm resets and moves to its resting pose: keep clear...")
    arm = await asyncio.to_thread(Arm, state["bounds"])
    brain.body = Body(arm, lambda text, wake: loop.call_soon_threadsafe(brain.note, text, wake),
                      lambda what: loop.call_soon_threadsafe(idling, what))
    brain.note("You have just been switched on, at your resting pose.", wake=True)
    print(f"Talk to it (or type). Ctrl-C (or Ctrl-D) stops the arm and quits. Log: {RUN}/")
    with patch_stdout(raw=True):
        worker = asyncio.create_task(brain.run())
        try:
            await chat(brain)
        finally:  # the kill switch works without the LLM: cut off the reply first, so it can queue nothing more
            brain.interrupt()
            ears.close()
            worker.cancel()
            brain.body.stop()
            await asyncio.sleep(0.2)  # the body thread holds the claw where it is
            arm.close()
            log("stop")
    print("Stopped: the arm holds where it is.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # Ctrl-C while connecting
        print("\nStopped.")
