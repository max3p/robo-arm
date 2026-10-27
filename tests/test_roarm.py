"""RoArmSerial against a fake that speaks the firmware's JSON lines."""

import json

import pytest

from arm import CONFIG_PATH, Pose, RoArmSerial, load_config


class FakeSerial:
    port = "fake"

    def __init__(self, max_x=1000):
        self.sent, self.buf = [], []
        self.pose, self.t, self.max_x = [235.0, 0.0, 234.0], 3.14, max_x

    def write(self, data):
        cmd = json.loads(data)
        self.sent.append(cmd)
        if cmd["T"] == 1041:
            self.pose = [min(cmd["x"], self.max_x), cmd["y"], cmd["z"]]
        elif cmd["T"] == 106:
            self.t = cmd["cmd"]
        elif cmd["T"] == 105:
            x, y, z = self.pose
            self.buf += [b"some debug text\n", b'{"T":1041}\n',
                         json.dumps({"T": 1051, "x": x, "y": y, "z": z, "t": self.t, "v": 1205}).encode() + b"\n"]

    def readline(self):
        return self.buf.pop(0) if self.buf else b""

    def reset_input_buffer(self):
        self.buf.clear()

    def close(self):
        pass


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


def test_move_streams_small_steps_carrying_gripper_angle(cfg):
    ser = FakeSerial()
    arm = RoArmSerial(cfg, ser=ser)
    r = arm.move_to(Pose(235, 0, 204), speed=1000)
    moves = [c for c in ser.sent if c["T"] == 1041]
    assert r.ok and r.final_pose == Pose(235, 0, 204)
    assert len(moves) == 10 and all(c["t"] == 3.14 for c in moves)


def test_unreachable_target_fails_verification(cfg):
    arm = RoArmSerial(cfg, ser=FakeSerial(max_x=250))
    r = arm.move_to(Pose(270, 0, 234), speed=1000)
    assert not r.ok and r.error_mm == 20 and r.final_pose.x == 250


def test_gripper_angle_is_kept_for_later_moves(cfg):
    ser = FakeSerial()
    arm = RoArmSerial(cfg, ser=ser)
    assert arm.set_gripper(False) == cfg["gripper"]["open"]
    arm.move_to(Pose(240, 0, 234), speed=1000)
    assert ser.sent[-3]["t"] == cfg["gripper"]["open"]  # last 1041, before the settle reads


def test_stop_turns_torque_on_and_blocks_motion(cfg):
    ser = FakeSerial()
    arm = RoArmSerial(cfg, ser=ser)
    arm.stop()
    assert ser.sent[-1] == {"T": 210, "cmd": 1}
    with pytest.raises(RuntimeError):
        arm.move_to(Pose(240, 0, 234), speed=1000)
