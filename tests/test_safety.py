import math

import pytest

from arm import Pose
from safety import Rejected, cap_speed, check_target, waypoints

B = {"x": [150, 300], "y": [-120, 120], "z": [80, 260]}


@pytest.mark.parametrize("values", [
    {"x": 150, "y": -120, "z": 80}, {"x": 300, "y": 120, "z": 260}, {"x": 200.5, "y": 0, "z": 100}])
def test_inside_and_boundary_values_pass(values):
    assert check_target(values, B) == Pose(**values)


@pytest.mark.parametrize("values, msg", [
    ({"x": 500, "y": 0, "z": 100}, "x=500 is outside allowed range [150, 300]"),
    ({"x": 149.9, "y": 0, "z": 100}, "x=149.9 is outside"),
    ({"x": 200, "y": 0, "z": 79}, "z=79 is outside"),  # floor
    ({"x": float("nan"), "y": 0, "z": 100}, "not a finite number"),
    ({"x": 200, "y": float("inf"), "z": 100}, "not a finite number"),
    ({"x": "200", "y": 0, "z": 100}, "not a finite number"),
    ({"x": True, "y": 0, "z": 100}, "not a finite number"),
    ({"x": 200, "y": 0}, "z=None is not a finite number"),
])
def test_rejections(values, msg):
    with pytest.raises(Rejected, match=msg.replace("[", r"\[")):
        check_target(values, B)


def test_long_move_is_split_into_short_segments_ending_at_target():
    start, end = Pose(150, -100, 100), Pose(300, 100, 250)
    path = waypoints(start, end, 30)
    assert path[-1] == end
    assert len(path) == math.ceil(start.dist(end) / 30)
    assert all(a.dist(b) <= 30 + 1e-9 for a, b in zip([start] + path, path))


def test_short_move_is_one_waypoint():
    assert waypoints(Pose(200, 0, 100), Pose(205, 0, 100), 30) == [Pose(205, 0, 100)]


def test_speed_cap():
    assert cap_speed(500, 60) == 60 and cap_speed(20, 60) == 20 and cap_speed(-5, 60) == 1
