"""Dancing to music: the ears stop transcribing (Ears.tap), the beat is found in what the mic hears, and the whole arm
bobs down on every beat with the claw held level, swaying across each bar of four, the light flashing on the beat. The
brain is not involved.

The bob swings the upper arm (shoulder) while the forearm, and so the claw, stays level: the claw then moves like the
elbow, on a circle of the upper arm's length around a point a forearm's length ahead of the shoulder. Bobbing straight
down instead was mostly the elbow, which read as a nod. The top is the person's "up extra high" pose (2026-11-08); the
bottom goes toward their low pose, as far as the box allows and slow enough for MAX_SPEED.

    listen FIRST_S -> dance DANCE_S on the first guess, guessing again every RESYNC_S from the last WINDOW_S
                      -> each step: the tempo most recent guesses agree on, the bob deeper the more they agree

find_beat (one guess): how much louder each 16 ms slice of sound gets than the one before (drums, notes starting),
summed over octaves; the beat is the gap at which that repeats best, also 2, 3 and 4 gaps on (autocorrelation, BPM
range: without the multiples it often picked 1.5 or 2/3 beats), and its phase the offset where the beats line up best.
Guesses on real songs from a phone held to the mic are weak (strength 0.11-0.22, 2026-11-08; clean synthetic songs
0.4-0.9) but often right, so it never refuses on one guess: it dances, and follows the consensus of the last GUESSES
(strength-weighted votes for a tempo; their beats averaged on a circle). It keeps its own beat clock and nudges its
speed by up to 10% toward that, so a correction never makes the claw jump.
"""

import cmath
import math
import queue
import time
from collections import deque

import numpy as np

from ears import FRAME, RATE
from motion import FOREARM_MM, GRIP_CLOSED, MAX_SPEED, UPPER_ARM_MM, check_target

FIRST_S, DANCE_S, WINDOW_S, RESYNC_S = 3, 30, 6, 0.5  # s
GUESSES = 12            # the last this many guesses (6 s of them) vote on the tempo
AGREE = 0.04            # guesses this close (as bobs, so double or half tempo agree) are for the same tempo
SWITCH = 1.5            # it changes tempo only when the new one has this many times the votes
FULL = 0.15             # confidence (the agreeing guesses' strength, times how well their beats line up, per guess) at
MIN_DEPTH = 0.3         # which it bobs at full depth; below it shallower, down to MIN_DEPTH of it
BPM = (60, 180)         # tempos it looks for. Prefers ones near 120: a beat at 70 could also be counted at 140
FAST_BPM = 140          # faster than this, it bobs on every other beat
TOP_DEG, LOW_DEG = -16, -57  # the upper arm, from vertical (+ forward), at the person's high (214, 230) and low (79, 122)
SWAY_DEG = 8            # the base turns this far each way across a bar
RAMP_S = 2              # it eases into the dance and out of it over this long, so it starts and ends still
LAG_S = 0.3             # s: aim this far ahead. The claw lags its targets ~330 ms (a delay: servo acceleration 20 to 250
# made no difference); with the song fed in without the mic, the bottoms landed ~65 ms early (2026-11-08), which the
# mic's delay (10-50 ms) mostly takes up. Bobs look late: raise it; early: lower it
HOP, WIN = 256, 1024    # samples: one onset value every 16 ms, over 64 ms


def onsets(audio):
    """How much louder each slice of sound got than the one before, summed over octaves (drums, notes starting).
    Value i belongs to the slice centred at (i + 1) * HOP + WIN / 2 samples."""
    slices = np.lib.stride_tricks.sliding_window_view(audio, WIN)[::HOP] * np.hanning(WIN)
    loud = np.log1p(1000 * np.abs(np.fft.rfft(slices)))
    per_octave = 1 / np.maximum(np.arange(loud.shape[1]), 2)  # each octave counts the same: hi-hats span hundreds of
    return np.maximum(np.diff(loud, axis=0), 0) @ per_octave  # bins, a kick a few; summed flat, off-beat hats won


def find_beat(audio):
    """(seconds per beat, seconds from the start of `audio` to a beat in it, strength 0..1): the best guess, however
    weak. Strength through the mic: clean music 0.5, music from a phone 0.11-0.22, speech up to 0.17, noise 0.1.
    None for under 2 s of sound."""
    if len(audio) < RATE * 2:
        return None
    env = onsets(audio)
    e = np.convolve(env - env.mean(), np.hanning(7), "same")  # blurred: a beat between two slices still lines up
    ac = np.correlate(e, e, "full")[len(e) - 1:] / np.arange(len(e), 0, -1)  # per overlapping pair
    ac /= ac[0] or 1
    fps = RATE / HOP
    lags = np.arange(round(fps * 60 / BPM[1]), round(fps * 60 / BPM[0]) + 1)
    ks = [k for k in (1, 2, 3, 4) if lags[-1] * k < len(ac)]  # a beat repeats at 2, 3, 4 beats too; 1.5 beats don't
    strength = np.mean([ac[lags * k] for k in ks], 0)
    lag = lags[np.argmax(strength * np.exp(-0.5 * np.log2(fps * 60 / lags / 120) ** 2))]
    strength = strength[lag - lags[0]]
    a, b, c = ac[lag - 1:lag + 2]
    lag += max(-0.5, min(0.5, 0.5 * (a - c) / (a - 2 * b + c or 1)))  # between lags: the top of a parabola through the
    # peak (unclamped, off a peak it once threw the lag negative)
    step = lag * (2 if fps * 60 / lag > FAST_BPM else 1)  # bobbing on every other beat: on the stronger ones
    ends = len(env) - 1 - np.arange(math.ceil(step))[:, None] - step * np.arange(int(len(env) / step))[None]
    score = np.where(ends >= 0, env[np.clip(np.round(ends).astype(int), 0, None)], 0).sum(1)
    last = len(env) - 1 - np.argmax(score)  # the onset value of the latest (strong) beat
    return lag / fps, ((last + 1) * HOP + WIN / 2) / RATE, strength


def claw(sway, deg):
    """(x, y, z) of the claw with the base turned `sway` degrees and the upper arm `deg` from vertical, the claw level."""
    r, sway, deg = FOREARM_MM + UPPER_ARM_MM * math.sin(math.radians(deg)), math.radians(sway), math.radians(deg)
    return r * math.cos(sway), r * math.sin(sway), UPPER_ARM_MM * math.cos(deg)


class Music:
    """The frames from Ears.tap, with when each was heard. They are contiguous, so sample i was heard at
    start + i / RATE; `start` is the earliest that fits every frame's arrival (a frame can arrive late, never early)."""

    def __init__(self, tapped):
        self.tapped, self.frames, self.start = tapped, [], math.inf

    def take(self, wait=0.0) -> None:
        try:
            while True:
                self.frames.append(self.tapped.get(timeout=wait) if wait else self.tapped.get_nowait())
                self.start = min(self.start, time.monotonic() - len(self.frames) * FRAME / RATE)
                wait = 0
        except queue.Empty:
            pass

    def secs(self) -> float:
        return len(self.frames) * FRAME / RATE

    def beat(self):
        """(seconds per beat, the clock time of a beat, strength) in the last WINDOW_S, or None."""
        n = min(len(self.frames), round(WINDOW_S * RATE / FRAME))
        found = n and find_beat(np.concatenate(self.frames[-n:]))
        if found:
            period, t, strength = found
            return period, self.start + (len(self.frames) - n) * FRAME / RATE + t, strength


def bob(period):
    """Seconds per bob for a beat of `period` seconds: above FAST_BPM, every other beat."""
    return period * (2 if 60 / period > FAST_BPM else 1)


def dance(arm, ears, say=print) -> str:
    """Listen for FIRST_S, then dance DANCE_S to the beat, following it as it listens on. Raises ValueError if the ears
    hear nothing at all (off)."""
    def fits(deg):
        try:
            return [check_target(arm.bounds, *claw(sway, deg)) for sway in (-SWAY_DEG, 0, SWAY_DEG)]
        except ValueError:
            return False

    if not fits(TOP_DEG):
        raise ValueError("the dance does not fit in the box")
    deepest = TOP_DEG
    while deepest - 1 >= LOW_DEG and fits(deepest - 1):
        deepest -= 1
    music, guesses = Music(ears.tap(True)), deque(maxlen=GUESSES)  # (s per beat, clock time of a beat, strength)
    try:
        arm.go_to(*claw(0, TOP_DEG), 200, GRIP_CLOSED)
        say("  (dance: listening for a beat...)")
        deadline = time.monotonic() + FIRST_S + 3  # the ears may not be ready yet
        while music.secs() < FIRST_S:
            if time.monotonic() > deadline:
                raise ValueError("heard nothing at all: the ears are off")
            music.take(0.1)
            arm._check_abort()
        guesses.append(music.beat())
        state = {}

        def vote(now):
            """Follow the tempo most of the last guesses agree on, and the beat they put it at."""
            if new := music.beat():
                guesses.append(new)
            same = lambda p, q: abs(bob(p) / bob(q) - 1) < AGREE
            votes = lambda p: sum(s for q, _, s in guesses if same(q, p))
            p = max((q for q, _, _ in guesses), key=votes)
            if state and votes(state["beat"]) * SWITCH >= votes(p):  # sticks to its tempo unless clearly outvoted: a
                p = state["beat"]  # drumless song flipped between 119 and 80 BPM
            agree = [g for g in guesses if same(g[0], p)]
            beat = float(np.median([q for q, _, _ in agree]))
            v = sum(s * cmath.exp(2j * math.pi * (now + LAG_S - a) / q) for q, a, s in agree)  # where in the beat now
            if not state or abs(bob(beat) / state["bob"] - 1) > AGREE:
                say(f"  (dance: {60 / beat:.0f} BPM{'' if state else f', dancing for {DANCE_S} s'})")
            fast = math.degrees(0.8 * MAX_SPEED * bob(beat) / math.pi / UPPER_ARM_MM)  # peak speed: pi * depth / bob
            want = min(TOP_DEG - deepest, fast) * max(MIN_DEPTH, min(1, abs(v) / len(guesses) / FULL))
            state.update(bob=bob(beat), beat=beat, at=now, phase=cmath.phase(v) / 2 / math.pi, want=want, voted=now)
            state.setdefault("tempos", []).append(60 / beat)

        now = time.monotonic()
        vote(now)
        state.update(b=state["phase"] * state["beat"] / state["bob"], t=now, depth=0.0)  # b counts bobs

        def pose(s):
            if s >= DANCE_S:
                return None
            now = time.monotonic()
            music.take()
            if now - state["voted"] >= RESYNC_S:
                vote(now)
            per = state["bob"] / state["beat"]  # beats per bob; either beat will do for a bob
            err = ((state["phase"] + (now - state["at"]) / state["beat"] - state["b"] * per + 0.5) % 1 - 0.5) / per
            state["b"] += (now - state["t"]) / state["bob"] * (1 + max(-0.1, min(0.1, err)))
            step = 10 * (now - state["t"])  # deg: the depth eases toward what it wants at 10 deg/s
            state["depth"] += max(-step, min(step, state["want"] - state["depth"]))
            state["t"], b = now, state["b"]
            ramp = min(1, s / RAMP_S, (DANCE_S - s) / RAMP_S)
            dip = 1 - abs(math.sin(math.pi * b))  # 1 on the beat, a sharp bounce there like a ball: a smooth wave had no
            # hit to see, and looked late (2026-11-08). The same peak speed as the wave, pi * depth / bob
            x, y, z = claw(SWAY_DEG * ramp * math.sin(math.pi * b / 2), TOP_DEG - state["depth"] * ramp * dip)
            return x, y, z, GRIP_CLOSED, 255 if dip > 0.8 else 30

        arm.follow(pose)
        return f"danced for {DANCE_S} s, mostly at about {np.median(state['tempos']):.0f} BPM"
    finally:
        ears.tap(False)
