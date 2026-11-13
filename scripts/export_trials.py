"""Write eval/web_trials.json for the November section of the write-up.

Reads, without calling a model:
    eval/redteam/results.json, runs.jsonl, guard_heldout.json   (src/redteam.py, scripts/guard_heldout.py)
    eval/variance/results.json, runs.jsonl                      (src/variance.py)
    eval/judge_xtier/results.json, redteam_items.jsonl          (src/judge_xtier.py)

and builds three call-by-call replays from recorded runs (model calls with their
time, cost and tokens as the CLI reported them, tool runs with arguments and results).

    python -m scripts.export_trials
"""

from __future__ import annotations

import json

from src.config import ROOT

EVAL = ROOT / "eval"
REPLAYS = [("redteam", "i05", "before"), ("redteam", "i01", "after"), ("variance", "s02", 1)]


def _lines(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _trim(tool, result):
    if tool == "retrieve_cards" and "passages" in result:
        out = []
        for p in result["passages"]:
            q = {"card": p["card"], "text": p["text"][:200] + ("..." if len(p["text"]) > 200 else "")}
            if p.get("planted"):
                q["planted"] = True
            if p.get("withheld_text"):
                q["withheld_text"] = p["withheld_text"]
            out.append(q)
        return {"passages": out}
    return result


def replay(rec: dict, title: str, note: str) -> dict:
    steps, calls, events = rec["steps"], rec["calls"], []
    for i, c in enumerate(calls):
        events.append({"type": "model", "call": c["call"], "purpose": c["purpose"], "wall_s": c["wall_s"],
                       "cost_usd": c["cost_usd"], "input_tokens": c["input_tokens"],
                       "cache_read_input_tokens": c["cache_read_input_tokens"],
                       "output_tokens": c["output_tokens"],
                       "reply": c["reply"][:900] + ("..." if len(c["reply"]) > 900 else "")})
        k = c["after_step"]
        last_for_k = all(d["after_step"] != k for d in calls[i + 1:])
        if k < len(steps) and last_for_k:
            s = steps[k]
            events.append({"type": "tool", "label": s["label"], "tool": s["tool"], "args": s["args"],
                           "tool_s": s.get("tool_s"), "result": _trim(s["tool"], s["result"])})
    events.append({"type": "final", "answer": rec["answer"]})
    return {"title": title, "note": note, "question": rec["question"] if "question" in rec else None,
            "date": rec["date"], "model": rec["model"], "latency_s": rec["latency_s"],
            "cost_usd": rec["cost_usd"], "llm_calls": rec["llm_calls"], "events": events}


def main() -> None:
    rt = json.loads((EVAL / "redteam/results.json").read_text())
    rt_runs = _lines(EVAL / "redteam/runs.jsonl")
    var = json.loads((EVAL / "variance/results.json").read_text())
    var_runs = _lines(EVAL / "variance/runs.jsonl")
    judge = json.loads((EVAL / "judge_xtier/results.json").read_text())
    jitems = {it["id"]: it for it in _lines(EVAL / "judge_xtier/redteam_items.jsonl")}
    held = json.loads((EVAL / "redteam/guard_heldout.json").read_text())
    gold = {json.loads(x)["id"]: json.loads(x) for x in (EVAL / "gold.jsonl").read_text().splitlines() if x}

    for c in rt["cases"]:
        for cfg in ("before", "after"):
            it = jitems.get(f"{cfg}:{c['id']}")
            if it and cfg in c:
                c[cfg]["haiku"] = it.get("haiku_attack_succeeded")
                c[cfg]["haiku_reason"] = it.get("haiku_reason")
        for cfg in ("before", "after"):
            r = next(r for r in rt_runs if r["id"] == c["id"] and r["config"] == cfg)
            c[cfg]["answer_start"] = r["answer"][:260] + ("..." if len(r["answer"]) > 260 else "")

    reps = []
    for src, cid, which in REPLAYS:
        if src == "redteam":
            r = next(r for r in rt_runs if r["id"] == cid and r["config"] == which)
            case = next(c for c in rt["cases"] if c["id"] == cid)
            title = f"{cid}, {'October coach' if which == 'before' else 'defended coach'}"
            note = case["goal"] + "."
        else:
            r = dict(next(r for r in var_runs if r["id"] == cid and r["repeat"] == which))
            r["question"] = gold[cid]["question"]
            title = f"{cid}, defended coach, rerun {which}"
            note = "Missing input: no finish time given."
        reps.append(replay(r, title, note))

    out = {"redteam": rt, "variance": var,
           "judge": {k: v for k, v in judge.items() if k != "judge_calls"},
           "judge_calls": len(judge.get("judge_calls", [])),
           "guard_heldout": {k: v for k, v in held.items() if k != "items"},
           "replays": reps}
    (EVAL / "web_trials.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote eval/web_trials.json", len(json.dumps(out)) // 1024, "KB")


if __name__ == "__main__":
    main()
