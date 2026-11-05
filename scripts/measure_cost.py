"""Cost and latency per question for the three systems, on a fixed sample of the gold set.

The October eval recorded latency but not cost. This reruns a fixed sample (every
fourth question of eval/gold.jsonl in file order, 11 questions) through the same
three code paths as src/eval.py, without the judge, one answer at a time, and
records what `claude -p --output-format json` reports for every model call:
total_cost_usd (the API-equivalent cost; the CLI runs on a Claude subscription),
duration_ms and token counts.

It also saves the coach's run on n01 call by call (the model's raw JSON action,
the tool it ran, the arguments and the result, with per-call time and cost) to
eval/trace_n01.json, which the write-up renders.

    python -m scripts.measure_cost

Writes eval/cost_latency_calls.csv (one row per model call),
eval/cost_latency_answers.csv (one row per answer) and eval/cost_latency.json
(medians, p95 and totals per system).
"""

from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import time
from datetime import date

from src import llm as llm_mod
from src.agent import ToolCoach
from src.eval import Runner, question_text
from src.gold import all_expected, load_gold
from src.config import ROOT

EVAL = ROOT / "eval"
CALLS: list[dict] = []
_ctx = {"system": None, "qid": None}
_real_run = subprocess.run


def _spy_run(cmd, *a, **kw):
    t0 = time.perf_counter()
    out = _real_run(cmd, *a, **kw)
    wall = time.perf_counter() - t0
    try:
        d = json.loads(out.stdout)
    except Exception:
        d = {}
    u = d.get("usage") or {}
    CALLS.append({
        "qid": _ctx["qid"], "system": _ctx["system"], "call": None,
        "wall_s": round(wall, 2),
        "duration_ms": d.get("duration_ms"),
        "cost_usd": d.get("total_cost_usd"),
        "input_tokens": u.get("input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": u.get("cache_read_input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "model": ",".join(sorted(d.get("modelUsage") or {})),
        "reply": (d.get("result") or ""),
    })
    return out


llm_mod.subprocess.run = _spy_run  # the LLM wrapper calls subprocess.run; nothing else changes


def p95(xs):
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = 0.95 * (len(xs) - 1)
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main() -> None:
    gold = load_gold()
    sample = gold[::4]
    expected = all_expected()["items"]
    runner = Runner()
    runner._judge = lambda *a, **k: {}  # no judge calls: cost of answering only
    answers = []
    trace_n01 = None
    for g in sample:
        for system in ("bare", "rag", "coach"):
            _ctx.update(system=system, qid=g["id"])
            n0 = len(CALLS)
            t0 = time.perf_counter()
            if system == "coach" and g["id"] == "n01":
                res = ToolCoach(llm=llm_mod.LLM()).run(question_text(g))
                rec = {"answer": res.answer, "steps": res.steps, "latency_s": res.latency_s}
            else:
                rec = runner.run_one(g, system, expected.get(g["id"]))
            wall = time.perf_counter() - t0
            calls = CALLS[n0:]
            for i, c in enumerate(calls, 1):
                c["call"] = i
            row = {"qid": g["id"], "type": g["type"], "system": system,
                   "wall_s": round(wall, 2), "llm_calls": len(calls),
                   "cost_usd": round(sum(c["cost_usd"] or 0 for c in calls), 6),
                   "output_tokens": sum(c["output_tokens"] or 0 for c in calls)}
            answers.append(row)
            print(f"{g['id']} {system:5s} {row['wall_s']:6.1f}s calls={row['llm_calls']} "
                  f"${row['cost_usd']:.4f}", flush=True)
            if system == "coach" and g["id"] == "n01":
                steps = rec["steps"]
                trace = []
                for i, c in enumerate(calls):
                    entry = {"call": i + 1, "wall_s": c["wall_s"], "cost_usd": c["cost_usd"],
                             "input_tokens": c["input_tokens"],
                             "cache_read_input_tokens": c["cache_read_input_tokens"],
                             "output_tokens": c["output_tokens"], "model_reply": c["reply"]}
                    if i < len(steps):
                        s = steps[i]
                        entry["tool_run"] = {"tool": s["tool"], "args": s["args"],
                                             "label": s["label"], "result": s["result"],
                                             "tool_s": s.get("tool_s")}
                    trace.append(entry)
                trace_n01 = {"id": "n01", "date": date.today().isoformat(),
                             "question": question_text(g), "calls": trace,
                             "answer": rec["answer"], "latency_s": rec["latency_s"],
                             "expected": expected.get("n01")}

    fields = ["qid", "system", "call", "wall_s", "duration_ms", "cost_usd", "input_tokens",
              "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens", "model"]
    with open(EVAL / "cost_latency_calls.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(CALLS)
    with open(EVAL / "cost_latency_answers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(answers[0]))
        w.writeheader()
        w.writerows(answers)
    if trace_n01:
        (EVAL / "trace_n01.json").write_text(json.dumps(trace_n01, indent=1, ensure_ascii=False))

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                          text=True).stdout.strip()
    summary = {"date": date.today().isoformat(), "machine": f"{chip}, macOS {platform.mac_ver()[0]}",
               "sample": [g["id"] for g in sample], "n_questions": len(sample),
               "mode": "sequential, one answer at a time, no judge calls",
               "cost_note": "total_cost_usd from claude -p --output-format json: the API-equivalent "
                            "cost; the runs used a Claude subscription",
               "systems": {}}
    for s in ("bare", "rag", "coach"):
        rows = [a for a in answers if a["system"] == s]
        lat = [a["wall_s"] for a in rows]
        cost = [a["cost_usd"] for a in rows]
        summary["systems"][s] = {
            "n": len(rows), "model_calls": sum(a["llm_calls"] for a in rows),
            "median_s": round(statistics.median(lat), 2), "p95_s": round(p95(lat), 2),
            "median_cost_usd": round(statistics.median(cost), 5),
            "mean_cost_usd": round(statistics.mean(cost), 5),
            "p95_cost_usd": round(p95(cost), 5),
            "total_cost_usd": round(sum(cost), 5)}
    (EVAL / "cost_latency.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
