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
  itself, without the brain (no API calls): it keeps the camera on the nearest face in view, and fidgets from MOODS
  that grow from curious to bored to sleepy while nothing happens, until it falls asleep. Asleep, only its name (NAME)
  gets through, and wakes it; a wake_up is never cut short, the newer actions queue after it. Someone starting to
  talk stops an idle move: it turns to the face it last saw (or perks up, if it has seen none) while it listens.
Dancing (dance.py) hands the mic to a beat tracker for 30 s instead: nothing is transcribed and the brain sits it out.
- Eyes (eyes.py): the camera on the claw, put into words by a small vision model on the GPU. What it sees goes with the
  brain's next message whenever it has changed; someone coming into view or making a hand gesture wakes the brain. A
  small picture of now goes with what a person says or does (heard, typed, arrived, gesture): ~190 tokens, only on
  turns that happen anyway. A bigger one is sent when the brain calls `look`.
New messages and failed actions wake the brain; finished actions and mood changes are only noted for its next turn.
The conversation is only ever appended to, never edited: that keeps the prompt cache and Claude's thinking valid.

Each run is logged to logs/<start time>/transcript.jsonl, one JSON line per event, `t` in seconds from the start:
heard / dropped, speech (someone started talking), typed, arm (each action as it was queued), idle (each fidget),
note (what the program told the brain), view (what the camera saw, as sent to the brain), eyes (what the eyes
reported: seen, arrived, gesture, ready, error), wrote, reply (latency, tokens, cache use), cut, error. No picture is
saved.
"""

import asyncio
import base64
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import anthropic
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from dance import DANCE_S, dance
from ears import Ears
from eyes import SETTLE_S, Eyes, locate
from motion import (DEFAULT_SPEED, GRIP_CLOSED, GRIP_OPEN, MAX_SPEED, MIN_SPEED, START, Arm, Interrupted,
                    check_target, describe_limits, load_state)
from gestures import GESTURES, listen, look_at
from track import Tracker

# Speed first: Haiku, at low effort (it thinks less, and skips thinking on simple turns). If it seems too dim, raise
# EFFORT to "medium", or use "claude-sonnet-5-5" or "claude-opus-5-5" (slower; Opus is the cleverest).
MODEL, EFFORT = "claude-haiku-5-5", "low"
SETTLE_SPEED = 150     # mm/s: idling starts by going back to rest, this fast
IDLE_GAP_S = (2, 5)   # s: idle fidgets come this far apart, at random, starting right after the brain's last movement
TICK_S = 0.25         # s: how often an idle body checks for someone to follow
WATCH_GAP_S = (6, 15) # s: following someone (track.py), it stops for a fidget this far apart, at random
HOLD_S = 5            # s: after the brain turns it somewhere (look_at, move_to, go_home), following waits this long
HUSH_S = 8            # s: someone starting to talk holds off idling this long (the brain answers ~2 s after they stop)
MOODS = [  # (s of nothing happening: no movement from the brain, no one moving in view; what the brain is told;
    # fidgets picked at random: a gesture, "glance" somewhere random, "look" at the face last seen (or glance), "home"
    # to rest; listing one twice makes it likelier). Curious, it doesn't glance away from a face in view or go home.
    (0, "you are curious, watching", ["glance", "glance", "look", "tilt", "curious"]),  # not told: it's every turn
    (30, "you are getting bored, fidgeting and looking for your person",
     ["glance", "glance", "look", "beg", "curious", "play_bow", "home"]),
    (120, "you are getting sleepy", ["yawn", "calm", "glance", "home"]),
    (240, "you fell asleep by yourself", ["sleep"]),  # stays asleep until the brain moves it
]
WAKE = "gesture(wake_up)"  # the label of the wake_up action, which nothing but stop cuts short
NAME = re.compile(r"\b(robert|rob)\b", re.I)  # asleep, only speech with its name wakes it; the rest is ignored
LOST_S = 60          # s: a face lost from view less than this long ago is looked for again where it was
SCAN_DEG = (45, -45, 90, -90, 135, -135)  # look_at_person, when no face is where it last saw one: turns to try, in order
RUN = Path(__file__).with_name("logs") / time.strftime("%Y-%m-%d_%H-%M-%S")  # this run's transcript
T0 = time.monotonic()


def system_prompt(bounds: dict) -> str:
    return f"""You are Robert (Rob for short), the mind of a small robot arm on a desk: a companion, like a clever \
pet dog. You live with your person and you adore them. You understand everything they say, but like a dog you cannot speak or write: you \
answer only with your body. Gestures, where you look, how close you come, your claw and the light on it are your whole \
language. Nothing you write as text reaches anyone, so never write; act.

You hear through a microphone, always on. Words reach you transcribed by a machine, so they can be misheard: if \
something makes no sense, look confused or tilt rather than guess. You hear everything said near you, and not all of \
it is meant for you (a phone call, the TV, people talking to each other): let that go by, or just glance over, like a \
dog would. Sometimes your person types to you instead.

You see through a camera on your head, just above the claw, looking wherever you face. A small vision model puts what \
it sees into words (it can be wrong), and the program adds where any faces are, roughly how far, and the look_at \
point that faces each one. You are told what your camera sees with your messages, whenever that has changed. Someone \
coming into view, or making a hand gesture at you (a thumbs up, a wave), wakes you. When your person says or does \
something, a small picture from your camera comes with it: trust it over the words, which may be wrong. For a sharper \
picture, or one from where you have just turned, call `look`. It costs more, so use it when you need the detail.

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
- Face tracking is on unless you turn it off: whenever your camera sees someone, your body keeps it on their face \
(or their head, if they are turned away), turning and tilting as they move, by itself and at once. Your gestures \
play facing them, and it carries on after each one. After look_at, move_to or go_home it waits a few seconds, then \
turns back to them. To look at something else for longer (an object, the window), call stop_face_tracking first, \
and start_face_tracking when you are done. If you have lost your person, call look_at_person: your body turns to \
where your camera last saw a face, and looks round for one if no one is there. Never guess where your person is: \
they move about. look_at is for things and places. The gestures that reach out (lean_in, curious, play_bow, \
greet) reach the way you face.
- You rest at {START} mm, folded back over your base (go_home). When things calm down, settle back there, like a pet \
lying down again.
- Between your movements your body carries on by itself: it follows your person (face tracking), looks for them \
again where it last saw them when it has lost them, and looks around curiously when no one is in view. When nothing happens for a while it gets bored, then sleepy, and in the end falls asleep. You are told when that mood changes; you need not act on it. Asleep (by yourself, or when \
told to sleep), you hear nothing at all until someone says your name; that wakes you (wake_up plays by itself), and \
you hear what they said.
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
    tool("look_at", "Turn and tilt your camera to face the point (x, y, z) mm, then perk up toward it for a moment. It "
         "does not reach for it, and stays facing it. The point may be outside the limits.", ["x", "y", "z"], **XYZ),
    tool("look_at_person", "Find your person: turn your camera to where it last saw a face, and if no one is there, "
         "look round until it finds one. With face tracking on and someone in view, it is already on them."),
    tool("start_face_tracking", "Keep your camera on your person by yourself, as they move (on at the start)."),
    tool("stop_face_tracking", "Stop following your person, to look at something else for a while."),
    tool("move_to", "Move the claw in a straight line to (x, y, z) mm.", ["x", "y", "z"], **XYZ, speed=SPEED),
    tool("go_home", f"Go back to the resting pose {START} mm, gripper shut.", speed=SPEED),
    tool("gripper", "Open or shut the claw: 0 = shut .. 1 = wide.", ["opening"], opening=num("0-1")),
    tool("light", "The light on the claw: brightness 0 = off .. 1 = full, after `blinks` quick flashes (default 0).",
         ["brightness"], brightness=num("0-1"), blinks={"type": "integer", "description": "0-10"}),
    tool("wait", "Pause between movements.", ["seconds"], seconds=num("0-30")),
    tool("dance", f"Dance to the music playing: listens for its beat for a few seconds, then bobs and sways to it for "
         f"{DANCE_S} s, following the beat as it goes (gently while it is unsure). Meanwhile you hear nothing; you are "
         "told how it went."),
    tool("look", "Take a picture with your camera and see it yourself. It is taken once the movements before it "
         "have played and the arm has settled, and reaches you straight after, with a turn to react to it."),
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


def body_action(arm: Arm, name: str, a: dict, ears=None, eyes=None):
    """Check a body tool's input now; return what the body thread will run later. Raises ValueError on bad input.
    A move_to target outside the box is refused here, so the reason goes back in its tool result (refused later, it
    woke the brain and cut off the reply still being written). The rest is checked by motion.py when it runs."""
    xyz = lambda: [number(a, k, -5000, 5000) for k in "xyz"]
    speed = lambda: number(a, "speed", MIN_SPEED, MAX_SPEED, DEFAULT_SPEED)
    if name == "gesture":
        if a.get("name") not in GESTURES:
            raise ValueError(f"no gesture {a.get('name')!r}; there are {', '.join(GESTURES)}")
        keys = GESTURES[a["name"]][1]
        return lambda: arm.perform(keys)  # always full size; perform only shrinks one that cannot fit anywhere
    if name == "look_at":
        keys = look_at(*xyz())
        return lambda: arm.perform(keys)
    if name == "look_at_person":
        if eyes is None:
            raise ValueError("you cannot see: the eyes are off")
        return lambda: seek(arm, eyes)
    if name == "move_to":
        (x, y, z), s = xyz(), speed()
        check_target(arm.bounds, x, y, z)
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
    if name == "dance":
        if ears is None:
            raise ValueError("you cannot hear: the ears are off")
        return lambda: dance(arm, ears, show)
    if name == "look":
        if eyes is None:
            raise ValueError("you cannot see: the eyes are off")
        return eyes.snap  # a picture: JPEG bytes
    if name == "wait":
        secs = number(a, "seconds", 0, 30)
        return lambda: pause(arm, secs)
    raise ValueError(f"no tool {name!r}")


def seek(arm: Arm, eyes) -> str:
    """look_at_person: turn to the face the camera last saw; if no face is in view there, look round (SCAN_DEG) and
    stop at the first one. Faces are read straight from the camera: the eyes only place them while the body is idle."""
    def face():  # the biggest face in a sharp frame from now on: its look_at point, or None
        after = time.monotonic() + SETTLE_S
        pause(arm, SETTLE_S + 0.3)
        with eyes.lock:
            faces, frame_t = eyes.faces, eyes.frame_t
        return locate(max(faces, key=lambda f: f[2]), arm._feedback())[3] if faces and frame_t > after else None
    if eyes.target and time.monotonic() - eyes.target[0] < 0.3:
        return "someone is in view already"
    if eyes.nearest:
        arm.perform(look_at(*eyes.nearest[3]))
    fb = arm._feedback()
    yaw, z = math.atan2(fb["y"], fb["x"]), fb["z"] + 270  # looking round ~15 degrees up, for a seated or standing face
    for turn in (0, *SCAN_DEG):
        if turn:
            arm.perform(look_at(1000 * math.cos(yaw + math.radians(turn)), 1000 * math.sin(yaw + math.radians(turn)), z))
        if point := face():
            arm.perform(look_at(*point))
            return "found a face and faced it"
    return "looked all round: no face in view"


def show(text: str, dim=False) -> None:
    print(f"\033[2m{text}\033[0m" if dim else text, flush=True)


def log(kind: str, **record) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "transcript.jsonl").open("a") as f:
        f.write(json.dumps({"t": round(time.monotonic() - T0, 2), "kind": kind, **record}) + "\n")


class Body:
    """Owns the arm, in its own thread. Actions run in order; the first action of a newer reply (a new `gen`) drops
    what older replies left and stops the one in progress (but never a wake_up). With nothing to do it idles by
    itself (MOODS).
    `report(text, wake, picture)` (a note for the brain, with a JPEG from `look`) and `idling(text)` (an idle move)
    are called from this thread. `fb` is the arm's feedback after the last action: where the camera is, when still."""

    def __init__(self, arm: Arm, report, idling):
        self.arm, self.report, self.idling = arm, report, idling
        self.ears = self.eyes = None  # for dance and look; set once they exist
        self.fb = {}
        self._where()
        self.todo, self.cv, self.gen, self.current = deque(), threading.Condition(), 0, None  # current: the label playing
        self.away = False  # the claw may be away from rest: idling starts by going home
        self.asleep = False  # after the sleep gesture it does not idle until the brain moves it
        self.since, self.mood, self.last = time.monotonic(), -1, None  # when the brain last moved it; idle state
        self.hushed = 0  # when someone last started talking
        self.stopped, self.next = 0, 0  # when the last action ended; when the next fidget is due
        self.tracking, self.tracker = True, None  # following people (start/stop_face_tracking); the Tracker
        self.watch = self.hold = 0  # following until then, then a fidget; not following before then
        self.sought = None  # the sighting (its time) last looked for again
        threading.Thread(target=self._run, daemon=True).start()

    def hush(self) -> None:
        """Someone started talking: like a pet, stop idling (gen None), turn to the face it last saw (or, if none
        yet, perk up and lean forward), and hold off fidgets for HUSH_S. Not while asleep or waking up, while the
        brain's movements play, or while following someone (it already faces them); the brain's next movement
        replaces it."""
        with self.cv:
            self.hushed = time.monotonic()
            if (self.asleep or self._waking() or (self.gen is not None and (self.current or self.todo))
                    or self.current == "idle: track"):
                return
            self._drop()
            self.gen, self.away = None, True  # away: idling starts again by going home
            face = self.eyes and self.eyes.nearest
            keys = look_at(*face[3]) if face else listen
            self.todo.append(("idle: listen", lambda: self.arm.perform(keys)))
            self.cv.notify()
            self.idling("listen")

    def wake(self) -> None:
        """Its name was heard while asleep: wake up now."""
        with self.cv:
            self._drop()
            self._wake()

    def do(self, gen: int, label: str, fn) -> None:
        with self.cv:
            if gen != self.gen:
                self.gen = gen
                self._drop()
                if self.asleep and label != WAKE:  # a pet wakes up before anything else
                    self._wake()
            if label != WAKE or not self._waking():  # already waking up: once is enough
                self.todo.append((label, fn))
                self.cv.notify()

    def stop(self) -> None:
        """Stop everything now, a wake_up too."""
        with self.cv:
            self._drop(everything=True)

    def _wake(self) -> None:
        self.asleep = False
        self.todo.append((WAKE, lambda: self.arm.perform(GESTURES["wake_up"][1])))
        self.cv.notify()

    def _waking(self) -> bool:
        return self.current == WAKE or any(label == WAKE for label, _ in self.todo)

    def _drop(self, everything=False) -> None:
        """Drop what is queued and stop what is playing, except a wake_up: cut off half way, it left the claw stretched
        out at the far edge of the box (2026-11-07). Newer actions queue after it instead."""
        self.todo = deque(a for a in self.todo if a[0] == WAKE and not everything)
        if self.current and (everything or self.current != WAKE):
            self.arm.abort.set()

    def _run(self) -> None:
        while True:
            with self.cv:
                while not self.todo:
                    if not self.cv.wait(TICK_S):
                        self._idle()
                label, fn = self.todo.popleft()
                self.current = label
            idle = label.startswith("idle: ")
            picture = None
            try:
                said = fn()
                outcome, wake = f"done: {said}" if isinstance(said, str) else "done", False
                if isinstance(said, bytes):  # look: the brain gets the picture, and a turn to react to it
                    picture, outcome, wake = said, "done: the picture follows", True
            except Interrupted:
                outcome, wake = "cut short", False
            except Exception as e:  # refused (ValueError) or failed (RuntimeError, serial): the brain hears at once
                outcome, wake = f"failed: {e}", True
            where = self._where()  # idle moves too, and before it counts as still: the eyes use it for where it looks
            with self.cv:  # abort is only ever set while something plays, so this leaves it clear for whatever comes next
                self.current = None
                self.arm.abort.clear()
                self.stopped = time.monotonic()
                self.next = self.stopped + random.uniform(*IDLE_GAP_S)
                self.asleep = label.startswith(("gesture(sleep", "idle: sleep"))
                if not idle:
                    self.since, self.mood = time.monotonic(), -1
                    self.away = not label.startswith(("go_home", "gesture(sleep", "gesture(wake_up"))
                    if label.startswith(("look_at(", "move_to(", "go_home(")):  # let it be seen before following on
                        self.hold = time.monotonic() + HOLD_S
            if not idle:
                self.report(f"{label} {outcome}{where}", wake, picture)
            elif wake:  # a fidget that did not fit is just skipped
                self.idling(f"{label[6:]} {outcome}")

    def _idle(self) -> None:
        """Nothing to do (every TICK_S): follow someone in view (track.py), stopping every WATCH_GAP_S for a fidget;
        with no one, fidget every IDLE_GAP_S. Fidgets are picked at random from the mood the quiet has reached."""
        now = time.monotonic()
        mood = sum(now - self.since >= m[0] for m in MOODS) - 1
        if self.asleep or mood < 0:
            return
        if mood != self.mood:
            self.mood, secs = mood, MOODS[mood][0]
            if secs:
                span = f"{secs:g} s" if secs < 60 else f"{secs / 60:g} minutes"
                self.report(f"Nothing from your person for {span}: {MOODS[mood][1]}.", False)
        face = self.eyes and self.eyes.nearest  # (when, degrees right, degrees up, look_at point)
        target = self.eyes and self.eyes.target
        in_view = target and target[0] > self.stopped and now - target[0] < 0.5  # someone, since the arm last stopped
        if self.tracking and in_view and now > self.hold:
            if now < self.watch or now - self.hushed < HUSH_S:  # follow them until a fidget is due
                self.tracker = self.tracker or Tracker(self.arm, self.eyes)
                self.gen, self.away = None, True
                self.todo.append(("idle: track", lambda: self.tracker.run(self._watching)))
                return self.idling(f"follow the {target[3]}")
        elif now < self.next or now - self.hushed < HUSH_S:
            return
        self.next, self.watch = now + random.uniform(*IDLE_GAP_S), now + random.uniform(*WATCH_GAP_S)
        if face and not in_view and now - face[0] < LOST_S and face[0] != self.sought:  # lost them: look where they were
            self.sought, self.gen = face[0], None
            self.todo.append(("idle: find", lambda: self.arm.perform(look_at(*face[3]))))
            return self.idling("find the face last seen")
        what = "home" if self.away and not in_view else random.choice(
            [w for w in MOODS[mood][2] if w != self.last] or MOODS[mood][2])
        if in_view and mood == 0 and what in ("glance", "look"):  # following someone: keep at it (look aims at where
            return  # their face was last placed, which may be out of date: it turned 34 degrees off them, 2026-11-08)
        self.away, self.last = self.away and what != "home", what
        if what == "home":
            fn = lambda: self.arm.go_to(*START, SETTLE_SPEED, GRIP_CLOSED)
        else:
            if what == "look" and not face:  # no face seen yet
                what = "glance"
            if what == "glance":  # anywhere round, high or low (-30 to 40 degrees in the box drawn 2026-11-07)
                a, r = math.radians(random.uniform(-150, 150)), random.uniform(400, 800)
                keys = look_at(r * math.cos(a), r * math.sin(a), random.uniform(0, 400))
            else:
                keys = look_at(*face[3]) if what == "look" else GESTURES[what][1]
            fn = lambda: self.arm.perform(keys)
        self.gen = None  # so the next reply's first movement replaces it
        self.todo.append((f"idle: {what}", fn))
        self.idling(what)

    def _watching(self) -> bool:
        """Following someone: go on? Not once there is something to do, following is turned off, or a fidget is due
        (and no one is talking). Someone moving about is something happening: no boredom."""
        now = time.monotonic()
        if not self.tracker.calm():
            self.since = now
        return self.tracking and not self.todo and (now < self.watch or now - self.hushed < HUSH_S)

    def pose(self):
        """Where the camera is, for the eyes: the arm's feedback while it is still, the tracker's while it follows
        slowly; None while it moves."""
        if self.current is None and not self.todo:
            return self.fb
        if self.current == "idle: track" and self.tracker.calm():
            return self.tracker.fb
        return None

    def _where(self) -> str:
        try:
            self.fb = self.arm._feedback()
            return f", claw at ({self.fb['x']:.0f}, {self.fb['y']:.0f}, {self.fb['z']:.0f})"
        except Exception:
            self.fb = {}  # still, but where the camera points is unknown
            return ""


class Brain:
    def __init__(self, client: anthropic.AsyncAnthropic, system: str):
        self.client, self.system, self.body = client, system, None
        self.history = []  # the conversation so far, as sent: only ever appended to
        self.results = []  # tool_result blocks owed for the last reply's tool calls, sent with the next message
        self.heard, self.notes, self.pictures = [], [], []  # the person's messages, the program's notes and
        self.seen = ""  # pictures from `look`, not sent yet; what the camera saw, as last sent
        self.glance = False  # a person said or did something: send a small picture of now with the next turn
        self.wake = asyncio.Queue()
        self.gen, self.task, self.t0, self.first = 0, None, 0, 0

    def hear(self, text: str) -> None:
        self.heard.append(text)
        self.glance = True
        self.wake.put_nowait(None)

    def note(self, text: str, wake=False, picture: bytes | None = None) -> None:
        log("note", text=text)
        self.notes.append(text)
        if picture:
            self.pictures.append(picture)
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
        for p in self.pictures:
            texts += [{"type": "text", "text": "(your camera, from your look:)"},
                      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": base64.b64encode(p).decode()}}]
        glance = self.glance and not self.pictures and self.body.eyes and self.body.eyes.glance()
        if glance:
            texts += [{"type": "text", "text": "(your camera, now:)"},
                      {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                   "data": base64.b64encode(glance).decode()}}]
        turn = [{"role": "user", "content": self.results + texts or [{"type": "text", "text": "(no message)"}]}]
        seen = self.body.eyes.view() if self.body.eyes else ""
        notes = self.notes + ([f"Your camera sees: {seen}"] if seen and seen != self.seen else [])
        if notes:
            turn.append({"role": "system", "content": " ".join(notes)})
        taken, did, owed = (len(self.heard), len(self.notes), len(self.pictures)), [], {}
        self.t0 = time.monotonic()
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
        del self.heard[:taken[0]], self.notes[:taken[1]], self.pictures[:taken[2]]
        self.glance = False
        if seen != self.seen:
            self.seen = seen
            log("view", text=seen)
        log("reply", first_s=round(self.first - self.t0, 2) if did else None, secs=round(time.monotonic() - self.t0, 2),
            stop=reply.stop_reason, model=reply.model, glance=bool(glance), **{k: getattr(reply.usage, k) for k in (
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
            elif block.name in ("start_face_tracking", "stop_face_tracking"):
                self.body.tracking = block.name.startswith("start")
                self.done(did, block.name, f"arm> {block.name}")
                result = "on" if self.body.tracking else "off"
            else:
                fn = body_action(self.body.arm, block.name, a, self.body.ears, self.body.eyes)
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
    """What the ears report (on the event loop): shown for debugging, logged, and words go to the brain. Asleep, only
    speech with its name gets through (and wakes it), even what the ears would drop as too short or unsure: a quick
    "Rob!" is only 0.13-0.29 s of voice. The rest is neither shown nor kept (the log keeps only why the ears would
    have dropped it, and how long it was)."""
    if kind in ("heard", "dropped") and brain.body.asleep:
        if not NAME.search(info.get("text", "")):
            log("dropped", why="asleep", ears=info.get("why"), speech_s=info["speech_s"])
            return show("  (asleep: ignored what was said)", dim=True)
        kind = "heard"
        brain.body.wake()
        brain.note("You were asleep and heard your name: you are waking up.")
    log(kind, **info)
    if kind == "heard":
        show(f"heard> {info['text']}")
        brain.hear(info["text"])
    elif kind == "speech" and not brain.body.asleep:
        show("  (listening...)", dim=True)
        brain.body.hush()
    elif kind == "dropped":
        show(f"  (ignored a sound: {info['why']}{': ' + repr(info['text']) if info.get('text') else ''})", dim=True)
    elif kind == "ready":
        show(f"(ears on: {info['mic']})", dim=True)
    elif kind == "error":
        show(f"(ears failed: {info['error']}. Typing still works.)", dim=True)


def saw(brain: Brain, kind: str, **info) -> None:
    """What the eyes report (on the event loop): shown for debugging and logged. Someone arriving or a hand gesture
    wakes the brain, but not asleep (only its name wakes it) or dancing (a reply would cut the dance off)."""
    log("eyes", what=kind, **info)
    body = brain.body
    quiet = body.asleep or (body.current or "").startswith("dance")
    if kind == "seen":
        show(f"  (sees: {info['text']}{' Gesture: ' + info['gesture'] if info['gesture'] else ''})", dim=True)
    elif kind == "arrived":
        brain.glance = True
        brain.note("Someone came into view of your camera.", wake=not quiet)
    elif kind == "gesture":
        brain.glance = True
        brain.note(f"Someone in view is making a hand gesture at you: {info['gesture']}.", wake=not quiet)
    elif kind == "ready":
        show(f"(eyes on: {info['camera']})", dim=True)
    elif kind == "error":
        show(f"(eyes failed: {info['error']}. The rest still works.)", dim=True)


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
    print("Connecting. The arm resets and moves to its resting pose: keep clear...")
    arm = await asyncio.to_thread(Arm, state["bounds"])
    brain.body = Body(arm, lambda text, wake, picture=None: loop.call_soon_threadsafe(brain.note, text, wake, picture),
                      lambda what: loop.call_soon_threadsafe(idling, what))
    ears = Ears(lambda kind, **info: loop.call_soon_threadsafe(lambda: heard(brain, kind, **info)),  # ready in ~3 s
                lambda: brain.body.asleep)
    brain.body.ears = ears
    brain.body.eyes = Eyes(lambda kind, **info: loop.call_soon_threadsafe(lambda: saw(brain, kind, **info)),
                           brain.body.pose)
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
