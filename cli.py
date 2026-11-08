"""Human control of the arm: draw the bounds the AI must stay inside, teach named paths by hand, and test moves.

    python cli.py         uses the bounds saved in arm_state.json (or makes you draw them), then type:
        read                    print the claw position (mm) and gripper angle
        move(x, y, z[, speed])  straight-line move inside the box, speed in mm/s
        grip(t)                 gripper angle in radians (3.14 shut .. 1.08 wide)
        home                    back to the resting pose START (folded back), gripper shut
        led(n)                  gripper LED brightness, 0 (off) .. 255
        bounds                  draw the box again: free the arm and sweep the claw around the space it may use
                                (sampled every 100 ms; the min and max of x, y and z become the box). Saved at once.
        trace[(speed)]          the claw shows the six sides of the box (left right up down forward back)
        record NAME             free the arm; Enter saves a point (must be inside the box), q + Enter stops and saves
                                the path as NAME. One point = a named pose. The AI can play saved paths.
        play NAME[, speed]      replay a path in straight lines, speed in mm/s
        gesture NAME[, size]    body language (nod, shake, curious, happy, excited, sad, startle, sleep, wake_up)
        look(x, y, z)           turn to face a point
        paths                   list the saved paths
        delete NAME             remove a saved path

Ctrl-C is the kill switch: no further motion is sent and the arm holds where it is. The power switch is the real e-stop.
"""

import math
import select
import sys

from arm import GRIP_CLOSED, START, MAX_SPEED, Arm, check_target, load_state, now_iso, save_state
from gestures import GESTURES, look_at

SAMPLE_S = 0.1   # the bounds sweep samples the claw position this often
MIN_BOX_MM = 20  # a bounds sweep that spans less than this on any axis is rejected as a mistake


def show_bounds(state: dict) -> None:
    print(f"Bounds drawn {state['bounds_drawn']} (mm):")
    for axis, (a, b) in state["bounds"].items():
        print(f"  {axis}: {a:7.1f} .. {b:7.1f}   (span {b - a:.0f})")


def draw_bounds(arm: Arm, state: dict) -> None:
    """Free the arm; the person sweeps the claw around the space it may use. The min and max of the sampled
    positions become the box, saved to arm_state.json. The arm is locked again on the way out, even on Ctrl-C."""
    input("Draw the bounds. Torque goes OFF and the arm will sag under its own weight. Support it, then press Enter... ")
    arm.free()
    print("Move the claw around the whole space it may use: corners, top, floor. Press Enter when done.")
    lo = hi = None
    n = 0
    try:
        while not select.select([sys.stdin], [], [], SAMPLE_S)[0]:  # the wait doubles as the sample interval
            p = arm.read()[:3]
            lo = p if lo is None else tuple(map(min, lo, p))
            hi = p if hi is None else tuple(map(max, hi, p))
            n += 1
        sys.stdin.readline()
    finally:
        arm.lock()
    if n < 2 or any(h - l < MIN_BOX_MM for l, h in zip(lo, hi)):
        raise ValueError(f"the sweep spans under {MIN_BOX_MM} mm on some axis ({n} samples); sweep a bigger space")
    state["bounds"], state["bounds_drawn"] = {a: [round(l, 1), round(h, 1)] for a, l, h in zip("xyz", lo, hi)}, now_iso()
    arm.bounds = state["bounds"]
    save_state(state)
    show_bounds(state)
    for name, points in state["paths"].items():
        try:
            for x, y, z, _ in points:
                check_target(arm.bounds, x, y, z)
        except ValueError:
            print(f"  note: path '{name}' now has points outside the box; playing it will be refused")


def trace(arm: Arm, speed: float = MAX_SPEED) -> None:
    """From the centre of the box the claw goes to the middle of each side (x forward, y left, z up), returning to the
    centre in between. A side that go_to refuses (near the base, out of reach) is skipped."""
    (x0, x1), (y0, y1), (z0, z1) = arm.bounds.values()
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    sides = {"left": (c[0], y1, c[2]), "right": (c[0], y0, c[2]), "up": (c[0], c[1], z1),
             "down": (c[0], c[1], z0), "forward": (x1, c[1], c[2]), "back": (x0, c[1], c[2])}
    total = 2 * sum(math.dist(c, p) for p in sides.values())
    print(f"Showing 6 sides, about {total * 1.5 / min(speed, MAX_SPEED):.0f} s plus the move to the centre.")
    arm.go_to(*c, speed)
    for name, p in sides.items():
        print(f"  -> {name}: x={p[0]:.0f} y={p[1]:.0f} z={p[2]:.0f}")
        try:
            arm.go_to(*p, speed)
        except ValueError as e:  # refused before anything was sent, so the arm is still at the centre
            print(f"     skipped: {e}")
            continue
        arm.go_to(*c, speed)


def record(arm: Arm, state: dict, name: str) -> None:
    """Free the arm; Enter stores the current pose (claw xyz + gripper), q + Enter saves the path as `name`.
    Poses outside the bounds are refused. The arm is locked again on the way out, even on Ctrl-C."""
    input("Torque goes OFF and the arm will sag under its own weight. Support it, then press Enter... ")
    points = []
    arm.free()
    print("Move the arm by hand.  Enter = record this pose,  q + Enter = stop and save.")
    try:
        while input(f"rec[{len(points)}]> ").strip().lower() != "q":
            x, y, z, t = arm.read()
            try:
                check_target(arm.bounds, x, y, z)
            except ValueError as e:
                print(f"  NOT saved: {e}")
                continue
            points.append([round(x, 1), round(y, 1), round(z, 1), round(t, 3)])
            print(f"  point {len(points)}: x={x:.1f} y={y:.1f} z={z:.1f} t={t:.2f}")
    finally:
        arm.lock()
        print("Arm locked.")
    if points:
        state["paths"][name] = points
        save_state(state)
    print(f"Saved '{name}' ({len(points)} points)." if points else "Nothing recorded.")


def main() -> None:
    state = load_state()
    arm = Arm(state["bounds"])
    try:
        if state["bounds"]:
            show_bounds(state)
            if input("Redraw them? [y/N] ").strip().lower() == "y":
                state["bounds"] = arm.bounds = None
        while not arm.bounds:
            try:
                draw_bounds(arm, state)
            except ValueError as e:
                print("error:", e)
        print("Commands: read | move(x, y, z[, speed]) | grip(t) | home | led(n) | bounds | trace[(speed)] | record NAME | "
              "play NAME[, speed] | gesture NAME[, size] | look(x, y, z) | paths | delete NAME    Ctrl-C quits and stops any move")
        while True:
            cmd, *args = input("arm> ").replace("(", " ").replace(")", " ").replace(",", " ").split() or [""]
            try:
                if cmd == "read" and not args:
                    print("x=%.1f y=%.1f z=%.1f t=%.2f" % arm.read())
                elif cmd == "move" and len(args) in (3, 4):
                    print("final x=%.1f y=%.1f z=%.1f" % arm.go_to(*map(float, args)))
                elif cmd == "grip" and len(args) == 1:
                    print("gripper t=%.2f" % arm.gripper(float(args[0])))
                elif cmd == "home" and not args:
                    print("final x=%.1f y=%.1f z=%.1f" % arm.go_to(*START, t=GRIP_CLOSED))
                elif cmd == "led" and len(args) == 1:
                    arm.led(int(args[0]))
                elif cmd == "bounds" and not args:
                    draw_bounds(arm, state)
                elif cmd == "trace" and len(args) <= 1:
                    trace(arm, *map(float, args))
                elif cmd == "record" and len(args) == 1:
                    record(arm, state, args[0])
                elif cmd == "play" and len(args) in (1, 2):
                    if args[0] not in state["paths"]:
                        raise ValueError(f"no path '{args[0]}'; saved: {', '.join(state['paths']) or 'none'}")
                    arm.play(state["paths"][args[0]], *map(float, args[1:]))
                    print("done")
                elif cmd == "gesture" and len(args) in (1, 2):
                    if args[0] not in GESTURES:
                        raise ValueError(f"no gesture '{args[0]}'; there are: {', '.join(GESTURES)}")
                    print("final x=%.1f y=%.1f z=%.1f" % arm.perform(GESTURES[args[0]][1], *map(float, args[1:]))[0])
                elif cmd == "look" and len(args) == 3:
                    print("final x=%.1f y=%.1f z=%.1f" % arm.perform(look_at(*map(float, args)))[0])
                elif cmd == "paths" and not args:
                    for name, points in state["paths"].items():
                        print(f"  {name}: {len(points)} points")
                elif cmd == "delete" and len(args) == 1:
                    if state["paths"].pop(args[0], None) is None:
                        raise ValueError(f"no path '{args[0]}'")
                    save_state(state)
                elif cmd:
                    print("unknown command; see the list above")
            except (ValueError, RuntimeError) as e:
                print("error:", e)
    except (KeyboardInterrupt, EOFError):
        print("\nSTOPPED: no further commands sent, the arm holds its position.")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
