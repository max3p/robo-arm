"""The agent loop against a scripted fake client (no API key needed)."""

from types import SimpleNamespace as NS

import pytest

from agent import run_agent
from arm import CONFIG_PATH, MockArm, load_config
from tools import Tools


class FakeClient:
    def __init__(self, calls):
        self.calls, self.requests = list(calls), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        if not self.calls:
            return NS(content=[NS(type="text", text="bye")], stop_reason="end_turn")
        name, args = self.calls.pop(0)
        return NS(content=[NS(type="tool_use", id=f"t{len(self.requests)}", name=name, input=args)],
                  stop_reason="tool_use")


@pytest.fixture
def tools():
    cfg = load_config(CONFIG_PATH)
    return Tools(MockArm(cfg), cfg)


def test_recovers_from_rejection_and_finishes(tools):
    client = FakeClient([("move_to", {"x": 500, "y": 0, "z": 100}), ("move_to", {"x": 250, "y": 0, "z": 100}),
                         ("done", {"reason": "there"})])
    assert run_agent("go", tools, client=client) == "done: there"
    rejected = client.requests[1]["messages"][2]["content"][0]
    assert rejected["is_error"] and "outside allowed range" in rejected["content"]
    assert client.requests[0]["cache_control"] == {"type": "ephemeral"}


def test_repeated_rejection_stops(tools):
    bad = ("move_to", {"x": 500, "y": 0, "z": 100})
    assert "repeated a rejected call" in run_agent("go", tools, client=FakeClient([bad, bad]))


def test_move_error_stops_the_run():
    cfg = load_config(CONFIG_PATH)
    tools = Tools(MockArm(cfg, drift_mm=20), cfg)
    out = run_agent("go", tools, client=FakeClient([("move_to", {"x": 230, "y": 0, "z": 230})] * 3))
    assert out.startswith("stopped: arm ended 20 mm")


def test_step_limit_and_no_tool_call(tools):
    assert "step limit of 2" in run_agent("go", tools, max_steps=2, client=FakeClient([("get_pose", {})] * 5))
    assert "without calling done" in run_agent("go", tools, client=FakeClient([]))
