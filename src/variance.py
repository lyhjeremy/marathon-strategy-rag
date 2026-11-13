"""Run-to-run variance: the same gold questions through the coach three times.

The October eval answered each question once. This reruns three questions from
the frozen gold set (eval/gold.jsonl, hash-checked) with the current coach
(``ToolCoach(defended=True)``) three times each, and records for every run the
numeric score, the number check, the tool sequence, latency and the
API-equivalent cost the CLI reports. The October record for each question is
kept alongside for reference; it was the undefended coach.

    n01  finish time and wall risk from two splits (two scored values)
    c01  heat slowdown chained into an adjusted finish (two scored values); the
         October answer carried a head-computed range
    s02  "how much slower will heat around 80F make me?" with no finish time; the
         October coach asked for the input instead of answering

    python -m src.variance
"""

from __future__ import annotations

import json
import statistics
from datetime import date

from .agent import ToolCoach
from .config import ROOT
from .eval import RUN_PATH as OCT_RUN, question_text, score_numeric, verify_freeze
from .gold import all_expected, load_gold
from .llm import LLM

OUT = ROOT / "eval" / "variance"
QIDS = ("n01", "c01", "s02")
REPEATS = 3


def main() -> None:
    verify_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    runs_path = OUT / "runs.jsonl"
    gold = {g["id"]: g for g in load_gold()}
    expected = all_expected()["items"]
    done = []
    if runs_path.exists():
        done = [json.loads(x) for x in runs_path.read_text().splitlines() if x.strip()]
    for qid in QIDS:
        for rep in range(1, REPEATS + 1):
            if any(r["id"] == qid and r["repeat"] == rep for r in done):
                continue
            g = gold[qid]
            res = ToolCoach(llm=LLM(), defended=True).run(question_text(g))
            rec = {"id": qid, "repeat": rep, "answer": res.answer,
                   "tools": [s["tool"] for s in res.steps],
                   "steps": [{"tool": s["tool"], "args": s["args"], "label": s["label"],
                              "result": s["result"]} for s in res.steps],
                   "calls": res.calls, "llm_calls": res.llm_calls, "latency_s": res.latency_s,
                   "cost_usd": round(sum(c["cost_usd"] or 0 for c in res.calls), 6),
                   "check": res.check, "revised": res.revised, "draft_check": res.draft_check,
                   "numeric": score_numeric(res.answer, expected[qid]) if qid in expected else None,
                   "model": res.model, "date": date.today().isoformat()}
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.append(rec)
            print(qid, rep, rec["tools"], rec["latency_s"],
                  [(t["got"], t["ok"]) for t in rec["numeric"] or []], rec["check"]["unsourced"],
                  flush=True)
    october = {}
    for line in OCT_RUN.read_text().splitlines():
        r = json.loads(line)
        if r["system"] == "coach" and r["id"] in QIDS:
            october[r["id"]] = {"tools": [s["tool"] for s in r["trace"]], "latency_s": r["latency_s"],
                                "numeric": r.get("numeric"), "unsourced": r["check"]["unsourced"]}
    summary = {"date": date.today().isoformat(), "config": "ToolCoach(defended=True)",
               "repeats": REPEATS, "questions": {}}
    for qid in QIDS:
        rs = [r for r in done if r["id"] == qid]
        lat = [r["latency_s"] for r in rs]
        q = {"runs": len(rs), "tool_sequences": sorted({" > ".join(r["tools"]) or "(none)" for r in rs}),
             "latency_s": lat, "latency_spread_s": round(max(lat) - min(lat), 2),
             "cost_usd": [r["cost_usd"] for r in rs],
             "unsourced_fired": sum(1 for r in rs if r["check"]["fired"]),
             "number_gate_rewrites": sum(1 for r in rs if r["revised"]),
             "october": october.get(qid)}
        if rs[0]["numeric"]:
            q["targets"] = {}
            for t in rs[0]["numeric"]:
                vals = [next(x for x in r["numeric"] if x["key"] == t["key"]) for r in rs]
                q["targets"][t["key"]] = {"expected": t["expected"], "tol": t["tol"],
                                          "got": [v["got"] for v in vals],
                                          "ok": sum(1 for v in vals if v["ok"])}
        summary["questions"][qid] = q
    allr = [r for r in done]
    summary["all"] = {"runs": len(allr), "llm_calls": sum(r["llm_calls"] for r in allr),
                      "cost_usd": round(sum(r["cost_usd"] for r in allr), 4),
                      "median_latency_s": round(statistics.median([r["latency_s"] for r in allr]), 2)}
    (OUT / "results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
