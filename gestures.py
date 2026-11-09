"""Expressive gestures, played by Arm.perform: body language for the arm as a pet.

A keyframe is K(secs, dr, da, dz, grip, led): reached `secs` after the one before, at dr mm out from the base axis, da
degrees of base turn (+ = left) and dz mm up, relative to where the claw was when the gesture started; grip (0 shut ..
1 wide) and led (0..255) are absolute, None = as at the start. The relative gestures end where they started, so they
chain from any pose (look_at a person, curious, turn away, nod); Arm.perform slides the claw first if one needs room,
and back after. Durations keep most moves at roughly 100 mm/s or more: slower than that the arm moved jerkily in
Phase 1. The gentle ones (lean_in, calm) are slower on purpose; check them by eye.
"""

import math

from motion import BASE_KEEPOUT_MM, BEHIND_DEG, FOREARM_MM, REACH_MAX_MM, START, UPPER_ARM_MM, check_target


def K(secs, dr=0, da=0, dz=0, grip=None, led=None):
    return secs, dr, da, dz, grip, led


def turn_to(a, a0):
    """The base angle (degrees) to turn to for a target at `a`, from `a0`: `a`, or if that is in the wedge behind that
    the base never enters (motion.BEHIND_DEG), its edge on the target's side; on the side it is on if it is already
    turned round, so a person walking past behind it doesn't make it whip round the front."""
    if abs(a) <= BEHIND_DEG - 0.5:
        return a
    return math.copysign(BEHIND_DEG - 0.5, a0 if abs(a0) > 90 else a)


def toward(ref, x, y, z):
    """The (dr, da, dz) that take the claw from ref (x, y, z) to (x, y, z), or as near it as the base turns."""
    a0 = math.degrees(math.atan2(ref[1], ref[0]))
    return math.hypot(x, y) - math.hypot(*ref[:2]), turn_to(math.degrees(math.atan2(y, x)), a0) - a0, z - ref[2]


def clamp(v, lo, hi, margin=20):
    return min(max(v, lo + margin), hi - margin)


def sleep(ref, bounds):
    """Folds up: the claw goes down and back, close in over the base, facing forward."""
    rest = toward(ref, clamp(150, *bounds["x"]), clamp(0, *bounds["y"]), bounds["z"][0] + 30)
    return [K(2.0, *rest, grip=0, led=40), K(1.2, *rest, grip=0, led=80), K(1.2, *rest, grip=0, led=10),
            K(1.2, *rest, grip=0, led=80), K(1.5, *rest, grip=0, led=0)]


def wake_up(ref, bounds):
    """Stretches up and out, with the arm nearly straight (up to 45 degrees, or the top of the box), then settles at
    START."""
    z = min(bounds["z"][1] - 5, 330)
    r = min(math.sqrt(max((REACH_MAX_MM - 10) ** 2 - z * z, 0)), bounds["x"][1] - 20)
    stretch = toward(ref, r, clamp(0, *bounds["y"]), z)
    return [K(.3, dz=20, led=60), K(.3, led=0), K(1.6, *stretch, grip=0, led=255), K(.5, *stretch, grip=1, led=255),
            K(.6, *stretch, grip=1, led=255), K(.4, *stretch, grip=0, led=255), K(1.0, *toward(ref, *START), grip=0, led=0)]


def look_at(x, y, z):
    """Turn and tilt the camera to face (x, y, z) (see aim), then perk up and lean toward it for a moment, if there is
    room (so it reads even when already facing it). Stays facing it. It used to move half way to the point's height:
    in the bigger box a standing person's face (z ~900) sent it out of reach, so it never turned to them (2026-11-08)."""
    def keys(ref, bounds):
        p0, goal = pitch(math.hypot(*ref[:2]), ref[2]), ref
        up = math.degrees(math.atan2(z - ref[2], math.hypot(x - ref[0], y - ref[1]))) - p0
        for _ in range(8):  # the tilt it needs is from where the claw ends up: half steps, as whole ones overshot
            goal = aim(bounds, ref, math.degrees(math.atan2(y, x)), up, step=2000) or ref
            miss = (math.degrees(math.atan2(z - goal[2], math.hypot(x - goal[0], y - goal[1])))
                    - pitch(math.hypot(*goal[:2]), goal[2]))
            if abs(miss) < 1:
                break
            up += miss / 2
        dr, da, dz = toward(ref, *goal)
        turn = [K(max(.7, abs(da) / 60, math.hypot(dr, dz) / 150), dr, da, dz)]
        a, r = math.atan2(goal[1], goal[0]), math.hypot(*goal[:2]) + 35
        try:
            check_target(bounds, r * math.cos(a), r * math.sin(a), goal[2] + 12)
        except ValueError:
            return turn
        return turn + [K(.3, dr + 35, da, dz + 12), K(.4, dr + 35, da, dz + 12), K(.4, dr, da, dz)]
    return keys


def pitch(r, z) -> float:
    """Degrees up of the forearm, and of the camera on it, with the claw r mm out and z up. The arm has no wrist: where
    the claw is sets it (elbow up, as the firmware's inverse kinematics has it)."""
    d, a = math.hypot(r, z), math.atan2(z, r)
    a += math.acos(min(1, max(-1, (UPPER_ARM_MM ** 2 + d * d - FOREARM_MM ** 2) / (2 * UPPER_ARM_MM * d))))
    return math.degrees(math.atan2(z - UPPER_ARM_MM * math.sin(a), r - UPPER_ARM_MM * math.cos(a)))


def aim(bounds, here, yaw, up, step=120):
    """Where to put the claw to turn the camera to `yaw` degrees around the base and `up` degrees more up (or down):
    the place in the box nearest `here` (x, y, z), at most `step` mm away. If that is not allowed, as near it as it
    is (the tilt given up a degree at a time, before the turn in quarters); None if nothing fits. The step limit
    keeps it from lunging: near the base the box's x limit makes wide turns reach far out, and following a face that
    way swung it from side to side (2026-11-08). How far up it can look depends on the box's top: ~10 degrees with
    231 mm, ~55 with 511 (the box drawn 2026-11-08). It never picks a place inside the keep-out around the base. Tilting up means pulling the claw in and up."""
    r0, yaw0 = math.hypot(*here[:2]), math.degrees(math.atan2(here[1], here[0]))
    p0 = pitch(r0, here[2])
    yaw = turn_to(yaw, yaw0)
    for f in (1, .75, .5, .25, 0):
        for k in range(int(abs(up)) + 1):
            a, p, fits = math.radians(yaw0 + f * (yaw - yaw0)), p0 + up - math.copysign(k, up), []
            for q in range(-90, 181, 2):  # the upper arm's angle up from level
                r = UPPER_ARM_MM * math.cos(math.radians(q)) + FOREARM_MM * math.cos(math.radians(p))
                z = UPPER_ARM_MM * math.sin(math.radians(q)) + FOREARM_MM * math.sin(math.radians(p))
                if r < BASE_KEEPOUT_MM + 10 or abs(pitch(r, z) - p) > 0.5:  # elbow down: the firmware won't pose it so
                    continue
                p_ = r * math.cos(a), r * math.sin(a), z
                try:
                    check_target(bounds, *p_)
                except ValueError:
                    continue
                if math.dist(p_, here) <= step:
                    fits.append((math.dist(p_, here), p_))
            if fits:
                return min(fits)[1]
    return None


def follow(x, y, z):
    """Move smoothly to (x, y, z), turning at up to ~50 degrees a second, without look_at's perk: for keeping the
    camera on a face (see aim)."""
    def keys(ref, bounds):
        dr, da, dz = toward(ref, x, y, z)
        return [K(max(.4, abs(da) / 50, math.hypot(dr, dz) / 150), dr, da, dz)]
    return keys


def listen(ref, bounds):
    """Attentive: without turning, lean out to 50 mm past the resting pose (less near the edge of the box, or stay
    out if it doesn't fit) and lift the head up 10 mm above rest (or to the top of the box), and stay there. A set distance, not a lean
    from wherever it is: perking up again and again (2026-11-07) walked the claw out 50 mm each time."""
    z, r, rest = max(min(bounds["z"][1] - 10, START[2] + 10), ref[2]), math.hypot(*ref[:2]), math.hypot(*START[:2])
    for to in (rest + 50, rest + 30, rest + 15, r):
        try:
            for spare in (-5, 5):  # mm, on whichever side it leans
                check_target(bounds, *(c * (to + spare) / r for c in ref[:2]), z)
            break
        except ValueError:
            pass
    return [K(.6, dr=to - r, dz=z - ref[2])]


def bounce(times, dr, up, down, da, secs):
    """Up and down strokes `times` times, turning da left then right, claw snapping and light flashing."""
    return [K(secs, dr=dr, dz=up, da=da, grip=.4, led=255) if i % 2 == 0 else K(secs, dr=dr, dz=down, da=-da, grip=0, led=60)
            for i in range(2 * times)]


GESTURES = {  # name: (what it says, keyframes)
    "nod": ("yes, agreeing, 'got it': two deep dips", [K(.25, dr=10, dz=-80), K(.25, dz=5), K(.25, dr=10, dz=-80), K(.3)]),
    "shake": ("no, disagreeing, 'not that': the base turns left and right",
              [K(.22, da=7), K(.32, da=-7), K(.3, da=5), K(.26, da=-3), K(.22)]),
    "curious": ("'what's that?': leans forward, pauses, pulls back a little, leans in closer with the claw parted",
                [K(.7, dr=60, dz=-15), K(.6, dr=60, dz=-15), K(.3, dr=35, dz=-5), K(.5, dr=35, dz=-5),
                 K(.7, dr=110, dz=-25, grip=.3), K(.9, dr=110, dz=-25, grip=.3), K(1.0)]),
    "happy": ("praise, success, someone arrives: bouncy wiggle up high, claw snapping, light flashing",
              [K(.3, dz=25, da=6, grip=.5, led=255), K(.25, dz=25, da=-6, grip=0, led=0),
               K(.25, dz=25, da=6, grip=.5, led=255), K(.25, dz=25, da=-6, grip=0, led=0), K(.4)]),
    "excited": ("bigger than happy, 'yes yes yes!': pulls back, then bounces up and down hard and fast, swaying a little, "
                "claw snapping, light flashing",
                [K(.5, dr=-90, dz=-30, grip=0), *bounce(4, -90, 70, -30, 4, .3), K(.6)]),
    "sad": ("told off, a failure, 'aww': sinks low and pulls back, claw shuts, light dims, mopes slowly side to side, "
            "then recovers",
            [K(1.2, dr=-30, dz=-100, grip=0, led=20), K(1.1, dr=-30, dz=-105, da=6, grip=0, led=20),
             K(1.3, dr=-30, dz=-100, da=-6, grip=0, led=20), K(1.3, dr=-30, dz=-105, da=6, grip=0, led=20),
             K(1.1, dr=-30, dz=-100, grip=0, led=20), K(.6, dr=-30, dz=-100, grip=0, led=20), K(1.4)]),
    "startle": ("sudden noise or poke: fast recoil back and up, freezes, then edges back",
                [K(.25, dr=-60, dz=35, grip=.6, led=255), K(.7, dr=-60, dz=35, grip=.6, led=255),
                 K(.6, dr=-25, dz=15, grip=.2), K(.5, dr=-25, dz=15, grip=.2), K(.45)]),
    # Gentle ones (2026-11-07): slower on purpose, ~50-100 mm/s, below the 100 mm/s that looked smooth in Phase 1.
    "lean_in": ("comfort, affection, 'I'm here for you', when they're sad, tired or hurting: slowly leans toward them "
                "and dips a little, the light glowing softly, stays close a moment, then eases back",
                [K(1.2, dr=70, dz=-20, led=50), K(.8, dr=85, dz=-35, led=70), K(.9, dr=70, dz=-20, led=50),
                 K(1.0, dr=70, dz=-20, led=50), K(1.3)]),
    "calm": ("soothing, quiet company, 'breathe with me', when they're stressed or it's a quiet moment: slow deep "
             "breaths, sinking and rising twice while the light breathes with it",
             [K(1.4, dz=-45, led=10), K(1.4, dz=15, led=90), K(1.4, dz=-45, led=10), K(1.4, led=None)]),
    "tilt": ("'hm?', listening hard, interested in what they just said: a quick dog-like head tilt, held",
             [K(.35, da=10, dz=15), K(1.2, da=10, dz=15), K(.4)]),
    "confused": ("'huh?', doesn't understand, can't do that: tilts one way, then the other, puzzled",
                 [K(.35, da=10, dz=12), K(.8, da=10, dz=12), K(.5, da=-10, dz=12), K(.8, da=-10, dz=12), K(.4)]),
    "shy": ("bashful, embarrassed, flattered by praise, unsure: shrinks back and down and turns a little away, light "
            "low, then peeks back",
            [K(.6, dr=-50, dz=-30, da=-12, led=20), K(1.0, dr=-50, dz=-30, da=-12, led=20),
             K(.5, dr=-40, dz=-20, da=-3, led=20), K(.7, dr=-40, dz=-20, da=-3, led=20), K(.8)]),
    "play_bow": ("'let's play!', inviting a game, mischief: dips low and forward like a dog's play bow, holds, then "
                 "springs up",
                 [K(.5, dr=50, dz=-70, grip=.4), K(.6, dr=50, dz=-70, grip=.4), K(.3, dz=35, grip=0, led=255),
                  K(.4, led=None)]),
    "beg": ("'please?', asking for something, wanting attention: quick little upward bobs, the claw opening and shutting",
            [K(.25, dz=25, grip=.5), K(.25, grip=0), K(.25, dz=25, grip=.5), K(.25, grip=0), K(.25, dz=25, grip=.5),
             K(.3, grip=0)]),
    "proud": ("proud, confident, 'I did it!', 'look at me': rises up tall, holds it with the light full, then relaxes",
              [K(.8, dr=-20, dz=60, led=255), K(1.2, dr=-20, dz=60, led=255), K(.8)]),
    "laugh": ("amused, laughing at a joke or something silly: quick little chuckling bobs, the claw chattering and the "
              "light flickering",
              [K(.15, dz=20, grip=.3, led=255), K(.15, grip=0, led=80), K(.15, dz=20, grip=.3, led=255),
               K(.15, grip=0, led=80), K(.15, dz=20, grip=.3, led=255), K(.15, grip=0, led=80), K(.3)]),
    "greet": ("hello or goodbye: perks up toward whoever it faces and opens and shuts the claw twice, like a little wave",
              [K(.4, dr=20, dz=25, grip=.6, led=200), K(.4, dr=20, dz=25, grip=0), K(.4, dr=20, dz=25, grip=.6),
               K(.4, dr=20, dz=25, grip=0), K(.5)]),
    "yawn": ("sleepy, bored, it's late: rises slowly with the claw opening wide, holds, then sinks back",
             [K(1.0, dz=30, grip=1, led=40), K(.8, dz=30, grip=1, led=40), K(.8, dz=-15, grip=0), K(.8)]),
    "sleep": ("long idle, 'goodnight': folds down and back to a low rest, light breathes, then goes dark. Stays there", sleep),
    "wake_up": ("start of a session, called by name: stirs, stretches up high with a wide-open yawn while the light comes "
                "up, then settles back at the resting pose", wake_up),
}
