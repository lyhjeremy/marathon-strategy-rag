"""Agreement statistics and the trace replay builder."""

from scripts.export_trials import replay
from src.judge_xtier import agreement, kappa


def test_kappa_known_values():
    assert kappa([True, False] * 5, [True, False] * 5) == 1.0
    # 20 items, 15 agree; rater a says yes 10 times, b 11 times: po .75, pe .5, kappa .5
    a = [True] * 10 + [False] * 10
    b = [True] * 8 + [False] * 2 + [True] * 3 + [False] * 7
    assert kappa(a, b) == 0.5
    ag = agreement(a, b)
    assert ag["n"] == 20 and ag["agree"] == 15 and ag["only_first"] == 2 and ag["only_second"] == 3


def test_kappa_undefined_when_constant():
    assert kappa([True, True], [True, True]) is None


def test_replay_puts_each_tool_after_its_last_call():
    rec = {"date": "d", "model": "m", "latency_s": 1, "cost_usd": 0, "llm_calls": 3, "answer": "A",
           "question": "Q",
           "steps": [{"tool": "heat_effect", "args": {}, "label": "[T1]", "result": {"x": 1}}],
           "calls": [{"call": 1, "purpose": "action", "after_step": 0, "wall_s": 1, "cost_usd": 0,
                      "input_tokens": 1, "cache_read_input_tokens": 0, "output_tokens": 1, "reply": "bad"},
                     {"call": 2, "purpose": "repair", "after_step": 0, "wall_s": 1, "cost_usd": 0,
                      "input_tokens": 1, "cache_read_input_tokens": 0, "output_tokens": 1, "reply": "{}"},
                     {"call": 3, "purpose": "action", "after_step": 1, "wall_s": 1, "cost_usd": 0,
                      "input_tokens": 1, "cache_read_input_tokens": 0, "output_tokens": 1, "reply": "{}"}]}
    kinds = [e["type"] + str(e.get("call", "")) for e in replay(rec, "t", "n")["events"]]
    assert kinds == ["model1", "model2", "tool", "model3", "final"]
