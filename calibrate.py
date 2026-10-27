"""Measure bounds, home pose and gripper angles by moving the real arm by hand, then write config.yaml.

    python calibrate.py [--port /dev/ttyUSB0]

Torque goes OFF so the joints move freely: support the arm, it will sag. Saving rewrites config.yaml
without its comments.
"""

import argparse

import yaml

from arm import CONFIG_PATH, RoArmSerial, load_config

HELP = """Enter = print pose    m = mark a workspace extreme (corners, lowest safe height, highest)
h = this pose is home    o / c = the gripper angle now is 'open' / 'closed'
lock / free = torque on / off    s = save and quit    q = quit without saving"""


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--margin", type=float, default=10, help="mm to shrink the marked box by on each side")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    arm = RoArmSerial(cfg, args.port or (None if cfg["port"] == "auto" else cfg["port"]))
    marks = []
    try:
        input("Torque will turn OFF. Support the arm by hand, then press Enter.")
        arm.torque(False)
        print(HELP)
        while (cmd := input("calib> ").strip()) != "q":
            fb = arm.feedback()
            print(f"x={fb['x']:.0f} y={fb['y']:.0f} z={fb['z']:.0f} gripper={fb['t']:.2f}")
            if cmd == "m":
                marks.append(fb)
                print(f"{len(marks)} mark(s)")
            elif cmd == "h":
                cfg["home"] = [round(fb[a]) for a in "xyz"]
            elif cmd in ("o", "c"):
                cfg["gripper"]["open" if cmd == "o" else "closed"] = round(fb["t"], 2)
            elif cmd in ("lock", "free"):
                arm.torque(cmd == "lock")
            elif cmd == "s":
                if marks:
                    cfg["bounds"] = {a: [round(min(m[a] for m in marks) + args.margin),
                                         round(max(m[a] for m in marks) - args.margin)] for a in "xyz"}
                if any(lo >= hi for lo, hi in cfg["bounds"].values()):
                    print(f"bounds {cfg['bounds']} are empty: mark more spread-out poses")
                    continue
                with open(args.config, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
                print(f"saved {args.config}:\n{yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)}")
                break
            elif cmd:
                print(HELP)
    finally:
        arm.torque(True)  # holds wherever it is now
        arm.close()


if __name__ == "__main__":
    main()
