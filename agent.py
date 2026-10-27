"""Claude tool-use loop: the model proposes one tool call at a time, Tools validates and executes it."""

import json

import anthropic

from tools import SCHEMAS

SYSTEM = """You control a 4-DOF desk robot arm through tools. Coordinates are the gripper tip in mm in \
the arm's base frame: x forward, y left, z up. The tip must stay inside this box: {bounds}. Home is {home}.
Every tool result reports the measured outcome; check it. If a call is rejected, choose a different valid \
target instead of repeating it. Call done when the goal is achieved, or when it cannot be and why."""


def run_agent(goal: str, tools, model: str = "claude-haiku-5-5", max_steps: int = 20, client=None) -> str:
    client = client or anthropic.Anthropic()  # ANTHROPIC_API_KEY from the environment
    system = SYSTEM.format(bounds=json.dumps(tools.cfg["bounds"]), home=tools.cfg["home"])
    messages = [{"role": "user", "content": goal}]
    last_rejected = None
    for _ in range(max_steps):
        resp = client.messages.create(
            model=model, max_tokens=4096, system=system, tools=SCHEMAS, messages=messages,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            output_config={"effort": "low"}, cache_control={"type": "ephemeral"})
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                print(f"  claude: {b.text.strip()}")
        call = next((b for b in resp.content if b.type == "tool_use"), None)
        if call is None:
            return f"model stopped without calling done ({resp.stop_reason})"
        result = tools.call(call.name, call.input)
        print(f"  {call.name}({json.dumps(call.input)}) -> {json.dumps(result)}")
        if result["status"] == "done":
            return f"done: {result['reason']}"
        if result["status"] == "error":  # the arm did not end where it should: stop, don't improvise
            return f"stopped: {result['reason']}"
        if result["status"] == "rejected":
            if (call.name, call.input) == last_rejected:
                return "stopped: the model repeated a rejected call"
            last_rejected = (call.name, call.input)
        messages.append({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call.id, "content": json.dumps(result),
            "is_error": result["status"] != "ok"}]})
    return f"stopped: step limit of {max_steps} reached"
