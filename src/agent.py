"""A tool-using coach: the model plans, the tools compute.

`claude -p` gives text in and text out, with no function-calling interface, so
the loop is written by hand:

1. The model sees the question, the tool list and every earlier step, and replies
   with one JSON object: either ``{"action": "tool", "tool": ..., "args": {...}}``
   or ``{"action": "final", "answer": "..."}``.
2. The loop runs the tool, numbers the result ([T1], [T2], ...; card passages get
   [1], [2], ... in the order they were first retrieved) and feeds it back.
3. A reply that is not valid JSON gets one repair retry with the parse error
   quoted back. A second failure ends the run with the raw text as the answer,
   marked ``parse_failed``.
4. After ``max_steps`` tool calls the model is told to answer with what it has.

The final answer then goes through `numcheck.check`, which flags any number
that is not in the question, a tool result or a cited passage.

When ``ANTHROPIC_API_KEY`` is set, ``ApiToolCoach`` runs the same tools through
the Messages API's native tool use instead. It is optional and is not used in
the evaluation, which runs everything through the CLI.

    python -m src.agent "I went through halfway in 1:42 ... how likely am I to hit the wall?"
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field

from . import numcheck
from .llm import LLM
from .tools import TOOL_SPECS, call_tool

AGENT_SYSTEM = """You are an evidence-minded marathon coach with tools. You answer runners' questions using tool results and knowledge-card passages, never from memory.

Reply with exactly ONE JSON object per turn and nothing else (no prose, no code fences):
  to call a tool:   {"action": "tool", "tool": "<name>", "args": {...}}
  to finish:        {"action": "final", "answer": "<your answer to the runner>"}

Tools:
{tools}

Rules:
- Use predict_finish for anything about splits, projected finish or wall risk; heat_effect for weather; pace_table for goal paces and split targets; retrieve_cards for advice and facts from the knowledge cards.
- Every number in your final answer must appear in a tool result, a passage you cite, or the runner's question. If you need a sum, difference, product or conversion, call calculator first. Do not round to new values in your head; quote the tool's numbers (rounding to whole minutes or seconds is fine).
- Cite passages as [1], [2] (the numbers shown next to them) and tool results as [T1], [T2]. Cite every number.
- Some course-difficulty figures come from a simulation; if you use them, say they are simulated.
- Keep the answer focused: the numbers the runner asked for, what they mean, and one or two concrete actions. If a tool returns an error, fix the arguments or say what you could not compute.
- You may make at most {max_steps} tool calls."""

STEP_PROMPT = """RUNNER'S QUESTION:
{question}

STEPS SO FAR:
{steps}

{instruction}"""


def _tool_lines() -> str:
    lines = []
    for t in TOOL_SPECS:
        props = t["input_schema"]["properties"]
        req = set(t["input_schema"].get("required", []))
        args = ", ".join(f"{k}{'' if k in req else '?'}: {v.get('type', 'any')}"
                         for k, v in props.items())
        detail = "; ".join(f"{k}: {v['description']}" for k, v in props.items()
                           if v.get("description"))
        lines.append(f"- {t['name']}({args}): {t['description']} [{detail}]")
    return "\n".join(lines)


def parse_action(text: str) -> dict:
    """Pull one JSON action out of a model reply; raise ValueError if impossible."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    # Walk to the matching closing brace so trailing chatter is ignored.
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = t[start:i + 1]
                break
    else:
        raise ValueError("unbalanced braces")
    blob = re.sub(r",\s*([}\]])", r"\1", blob)          # trailing commas
    obj = json.loads(blob, strict=False)                # tolerate raw newlines in strings
    if obj.get("action") == "final" and isinstance(obj.get("answer"), str):
        return obj
    if obj.get("action") == "tool" and isinstance(obj.get("tool"), str):
        obj.setdefault("args", {})
        if not isinstance(obj["args"], dict):
            raise ValueError("args must be an object")
        return obj
    raise ValueError('expected {"action": "tool", ...} or {"action": "final", ...}')


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list = field(default_factory=list)
    passages: list = field(default_factory=list)     # global [n] list
    tool_results: list = field(default_factory=list)  # [T1], [T2], ...
    llm_calls: int = 0
    repairs: int = 0
    parse_failed: bool = False
    hit_step_cap: bool = False
    latency_s: float = 0.0
    model: str = ""
    check: dict = field(default_factory=dict)

    def cited_passages(self) -> list[dict]:
        nums = {int(n) for n in re.findall(r"\[(\d+)\]", self.answer)}
        nums |= {int(n) for grp in re.findall(r"\[([\d,\s]+)\]", self.answer)
                 for n in re.findall(r"\d+", grp)}
        return [p for i, p in enumerate(self.passages, 1) if i in nums]

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "question", "answer", "steps", "llm_calls", "repairs", "parse_failed",
            "hit_step_cap", "latency_s", "model", "check")}


class ToolCoach:
    def __init__(self, max_steps: int = 6, llm: LLM | None = None):
        self.max_steps = max_steps
        self.llm = llm or LLM()
        self.system = AGENT_SYSTEM.replace("{tools}", _tool_lines()).replace(
            "{max_steps}", str(max_steps))

    # -- rendering ------------------------------------------------------------
    def _render_steps(self, res: AgentResult) -> str:
        if not res.steps:
            return "(none yet)"
        out = []
        for i, s in enumerate(res.steps, 1):
            out.append(f"Step {i}: {s['tool']}({json.dumps(s['args'], ensure_ascii=False)})")
            out.append(f"Result {s['label']}: {s['shown']}")
        return "\n".join(out)

    def _record_tool(self, res: AgentResult, tool: str, args: dict, result: dict) -> dict:
        label = f"[T{len(res.tool_results) + 1}]"
        res.tool_results.append(result)
        if tool == "retrieve_cards" and "passages" in result:
            shown = []
            for p in result["passages"]:
                key = (p["card"], p["text"])
                idx = next((i for i, q in enumerate(res.passages, 1)
                            if (q["card"], q["text"]) == key), None)
                if idx is None:
                    res.passages.append(p)
                    idx = len(res.passages)
                shown.append(f"\n  [{idx}] ({p['title']}; source: {p['source']}) {p['text']}")
            shown_text = "".join(shown) or " no passages"
        else:
            shown_text = json.dumps(result, ensure_ascii=False)
        return {"tool": tool, "args": args, "label": label, "shown": shown_text,
                "result": result}

    # -- loop -------------------------------------------------------------------
    def _ask(self, res: AgentResult, prompt: str) -> str:
        text = self.llm.complete(prompt, system=self.system)
        res.llm_calls += 1
        res.model = self.llm.last_meta.get("model", res.model)
        return text

    def run(self, question: str) -> AgentResult:
        res = AgentResult(question=question, answer="")
        t0 = time.perf_counter()
        while True:
            at_cap = len(res.steps) >= self.max_steps
            instruction = ("You have used every tool call. Reply now with "
                           '{"action": "final", "answer": "..."} using only the results above.'
                           if at_cap else "Your next action (one JSON object):")
            prompt = STEP_PROMPT.format(question=question, steps=self._render_steps(res),
                                        instruction=instruction)
            raw = self._ask(res, prompt)
            try:
                action = parse_action(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                res.repairs += 1
                raw = self._ask(res, prompt + (
                    f"\n\nYour previous reply could not be parsed ({exc}). It began: "
                    f"{raw[:300]!r}\nReply with only one valid JSON object."))
                try:
                    action = parse_action(raw)
                except (ValueError, json.JSONDecodeError):
                    res.parse_failed = True
                    res.answer = raw
                    break
            if action["action"] == "final":
                res.answer = action["answer"].strip()
                break
            if at_cap:  # asked for a final answer and called a tool anyway
                res.hit_step_cap = True
                res.answer = "(no final answer within the step cap)"
                break
            t1 = time.perf_counter()
            result = call_tool(action["tool"], action["args"])
            step = self._record_tool(res, action["tool"], action["args"], result)
            step["tool_s"] = round(time.perf_counter() - t1, 3)
            res.steps.append(step)
            if res.llm_calls > 3 * self.max_steps:  # belt and braces
                res.answer = "(stopped: too many model calls)"
                break
        res.latency_s = round(time.perf_counter() - t0, 2)
        evidence = [r for s, r in zip(res.steps, res.tool_results) if s["tool"] != "retrieve_cards"]
        evidence += [p["text"] for p in res.cited_passages()]
        res.check = numcheck.check(res.answer, question, evidence)
        return res


class ApiToolCoach:
    """Same tools through the Messages API's native tool use (needs ANTHROPIC_API_KEY)."""

    def __init__(self, max_steps: int = 6, model: str | None = None):
        import anthropic
        from .llm import DEFAULT_API_MODEL

        self.client = anthropic.Anthropic()
        self.model = model or DEFAULT_API_MODEL
        self.max_steps = max_steps
        self.system = AGENT_SYSTEM.split("Reply with exactly ONE JSON")[0] + (
            "Use the tools provided. Every number in your final answer must appear in a "
            "tool result, a cited passage or the question; call calculator for arithmetic. "
            "Cite passages as [card name] and tool results as [T1], [T2] in call order.")

    def run(self, question: str) -> dict:
        messages = [{"role": "user", "content": question}]
        steps, evidence, t0 = [], [], time.perf_counter()
        for _ in range(self.max_steps + 1):
            msg = self.client.messages.create(
                model=self.model, max_tokens=1500, system=self.system,
                tools=TOOL_SPECS, messages=messages)
            uses = [b for b in msg.content if b.type == "tool_use"]
            if msg.stop_reason != "tool_use" or not uses or len(steps) >= self.max_steps:
                answer = "".join(b.text for b in msg.content if b.type == "text").strip()
                break
            messages.append({"role": "assistant", "content": msg.content})
            results = []
            for u in uses:
                out = call_tool(u.name, u.input)
                steps.append({"tool": u.name, "args": u.input, "result": out})
                evidence.append(out)
                results.append({"type": "tool_result", "tool_use_id": u.id,
                                "content": json.dumps(out, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
        return {"question": question, "answer": answer, "steps": steps,
                "latency_s": round(time.perf_counter() - t0, 2), "model": self.model,
                "check": numcheck.check(answer, question, evidence)}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    q = " ".join(sys.argv[1:]) or "What even pace do I need for a 3:30 marathon?"
    res = ToolCoach().run(q)
    for i, s in enumerate(res.steps, 1):
        print(f"step {i}: {s['tool']} {json.dumps(s['args'], ensure_ascii=False)}")
    print("\n" + res.answer + "\n")
    print(f"[{res.model}, {res.llm_calls} model calls, {res.latency_s}s, "
          f"unsourced numbers: {res.check['unsourced'] or 'none'}]")


if __name__ == "__main__":
    main()
