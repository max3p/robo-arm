"""Drive the arm with Claude, or call tools by hand.

    python main.py --goal "move to the centre of the workspace, then wave the gripper"
    python main.py --interactive                     # type tool calls by hand, e.g. `move_to 200 0 150`
    python main.py --arm roarm --goal "..."          # real hardware, only after calibrate.py

Ctrl-C is the emergency stop: no further motion, torque stays on. The power switch is the real e-stop.
"""

import argparse
import json
import time
from pathlib import Path

from arm import CONFIG_PATH, MockArm, RoArmSerial, load_config
from tools import SCHEMAS, Tools

LOG_DIR = Path(__file__).with_name("logs")


def _parse(token: str):
    try:
        return float(token)
    except ValueError:
        return token


def interactive(tools: Tools) -> None:
    params = {s["name"]: list(s["input_schema"]["properties"]) for s in SCHEMAS}
    print("Tools:", ", ".join(f"{n} {' '.join(p)}".strip() for n, p in params.items()), "| Ctrl-D quits")
    while True:
        try:
            name, *values = input("tool> ").split() or [""]
        except EOFError:
            return
        if name in params:
            print(json.dumps(tools.call(name, dict(zip(params[name], map(_parse, values))))))
        elif name:
            print(f"unknown tool {name!r}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=["mock", "roarm"], default="mock")
    p.add_argument("--port", help="serial port (default: config.yaml, then auto-detect)")
    p.add_argument("--goal")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--speed", type=float, help="mm/s, capped at max_speed from the config")
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--mock-drift", type=float, default=0.0, help="mock: final poses are off by this many mm")
    p.add_argument("--mock-reach", type=float, help="mock: targets farther than this from the base stop short")
    args = p.parse_args(argv)
    if not (args.goal or args.interactive):
        p.error("give --goal or --interactive")

    cfg = load_config(args.config)
    if args.arm == "roarm":
        input("REAL ARM. Clamped down, workspace clear, someone at the power switch? Enter to go, Ctrl-C to abort.")
        arm = RoArmSerial(cfg, args.port or (None if cfg["port"] == "auto" else cfg["port"]))
    else:
        arm = MockArm(cfg, args.mock_drift, args.mock_reach)
    LOG_DIR.mkdir(exist_ok=True)
    tools = Tools(arm, cfg, LOG_DIR / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", args.speed)
    try:
        if args.interactive:
            interactive(tools)
        else:
            from agent import run_agent
            print(run_agent(args.goal, tools, args.model, args.max_steps))
    except KeyboardInterrupt:
        print("\nEMERGENCY STOP: no further motion, holding position")
        arm.stop()
    finally:
        arm.close()


if __name__ == "__main__":
    main()
