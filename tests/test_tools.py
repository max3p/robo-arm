import json

import pytest

from arm import CONFIG_PATH, MockArm, load_config
from tools import Tools


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


def test_move_to_reports_measured_pose_and_logs(cfg, tmp_path):
    log = tmp_path / "run.jsonl"
    tools = Tools(MockArm(cfg), cfg, log)
    r = tools.call("move_to", {"x": 200, "y": 50, "z": 100})
    assert r == {"status": "ok", "final_pose": {"x": 200, "y": 50, "z": 100}, "error_mm": 0}
    entry = json.loads(log.read_text())
    assert entry["tool"] == "move_to" and entry["result"] == r and "ms" in entry


def test_long_move_goes_through_waypoints(cfg, capsys):
    tools = Tools(MockArm(cfg), cfg)
    tools.call("move_to", {"x": 150, "y": -120, "z": 80})
    assert capsys.readouterr().out.count("[MOCK ARM] move_to") > 5


def test_rejected_target_does_not_move(cfg):
    arm = MockArm(cfg)
    r = Tools(arm, cfg).call("move_to", {"x": 500, "y": 0, "z": 100})
    assert r == {"status": "rejected", "reason": "x=500 is outside allowed range [150, 300]"}
    assert arm.pose.as_dict() == dict(zip("xyz", cfg["home"]))


@pytest.mark.parametrize("name, args", [
    ("move_to", {"x": 200, "y": 0}), ("move_by", {"dx": "a", "dy": 0, "dz": 0}),
    ("gripper", {"action": "half"}), ("fly", {})])
def test_bad_calls_are_rejected(cfg, name, args):
    assert Tools(MockArm(cfg), cfg).call(name, args)["status"] == "rejected"


def test_move_by_and_home(cfg):
    tools = Tools(MockArm(cfg), cfg)
    assert tools.call("move_by", {"dx": -20, "dy": 10, "dz": -100})["final_pose"] == {"x": 215, "y": 10, "z": 134}
    assert tools.call("go_home", {})["final_pose"] == dict(zip("xyz", cfg["home"]))


def test_error_after_move_stops_the_sequence(cfg, capsys):
    tools = Tools(MockArm(cfg, drift_mm=15), cfg)
    r = tools.call("move_to", {"x": 150, "y": 0, "z": 100})
    assert r["status"] == "error" and r["error_mm"] == 15 and "15 mm from the target" in r["reason"]
    assert capsys.readouterr().out.count("move_to") == 1  # did not continue to later waypoints


def test_unreachable_target_is_reported(cfg):
    r = Tools(MockArm(cfg, reach_mm=250), cfg).call("move_to", {"x": 230, "y": 0, "z": 250})
    assert r["status"] == "error"


def test_gripper_bounds_and_done(cfg):
    tools = Tools(MockArm(cfg), cfg)
    assert tools.call("gripper", {"action": "close"})["measured_angle"] == cfg["gripper"]["closed"]
    assert tools.call("get_bounds", {})["bounds"] == cfg["bounds"]
    assert tools.call("done", {"reason": "ok"}) == {"status": "done", "reason": "ok"}


def test_speed_is_capped(cfg):
    assert Tools(MockArm(cfg), cfg, speed=1000).speed == cfg["max_speed"]


def test_stopped_arm_refuses_motion(cfg):
    arm = MockArm(cfg)
    arm.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        Tools(arm, cfg).call("move_to", {"x": 200, "y": 0, "z": 200})
