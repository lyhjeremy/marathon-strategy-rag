"""Group the baselines' numeric misses from the saved eval run, and pull one real example per group.

Reads eval/runs/run_2026-10.jsonl (no model calls). Groups the 42 scored values by the
kind of question in the gold set (the id prefix), splitting wall-risk targets out of the
finish-time questions, and counts misses per system. Also computes latency percentiles
from the same run (four answers at a time).

    python -m scripts.error_analysis

Writes eval/error_analysis.json.
"""

from __future__ import annotations

import json
import statistics

from src.config import ROOT
from src.eval import RUN_PATH

GROUPS = {
    "wall": "Wall risk (n01, n04, n05, n07)",
    "n": "Finish-time projection from splits (n01 to n08)",
    "h": "Heat slowdown (h01 to h06)",
    "p": "Pace and split targets (p01 to p05)",
    "c": "Combined heat and pace (c01, c02)",
    "k": "Arithmetic on a knowledge card (k01 to k05)",
}
EXAMPLES = {  # one recorded miss per group, chosen as the clearest of the group
    "wall": ("n04", "rag"),
    "n": ("n04", "rag"),
    "h": ("h03", "bare"),
    "p": ("p04", "rag"),
    "c": ("c01", "rag"),
    "k": ("k02", "bare"),
}


def group_of(qid: str, key: str) -> str:
    return "wall" if key == "wall_risk_pct" else qid[0]


def pct(xs, q):
    xs = sorted(xs)
    k = q * (len(xs) - 1)
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def main() -> None:
    rows = [json.loads(x) for x in RUN_PATH.read_text(encoding="utf-8").splitlines() if x]
    out = {"source": "eval/runs/run_2026-10.jsonl", "groups": []}
    for g, label in GROUPS.items():
        entry = {"group": g, "label": label, "targets": 0, "misses": {}}
        for system in ("bare", "rag", "coach"):
            miss = tot = 0
            for r in rows:
                if r["system"] != system:
                    continue
                for t in r.get("numeric") or []:
                    if group_of(r["id"], t["key"]) == g:
                        tot += 1
                        miss += not t["ok"]
            entry["targets"] = tot
            entry["misses"][system] = miss
        qid, system = EXAMPLES[g]
        r = next(x for x in rows if x["id"] == qid and x["system"] == system)
        entry["example"] = {
            "id": qid, "system": system,
            "question": r["question"].split("\n\nAt the very end")[0],
            "targets": [t for t in r["numeric"] if group_of(qid, t["key"]) == g],
            "answer": r["answer"],
        }
        out["groups"].append(entry)
    out["totals"] = {s: {"misses": sum(g["misses"][s] for g in out["groups"]),
                         "targets": sum(g["targets"] for g in out["groups"])}
                     for s in ("bare", "rag", "coach")}

    coach = {r["id"]: r for r in rows if r["system"] == "coach"}
    out["coach_known_failures"] = [
        {"id": "s02", "kind": "asked for an input instead of answering",
         "question": coach["s02"]["question"], "answer": coach["s02"]["answer"],
         "judge": coach["s02"]["judge"], "tools_called": len(coach["s02"].get("trace", []))},
        {"id": "c01", "kind": "range computed in the model's head, passed by the number check",
         "question": coach["c01"]["question"].split("\n\nAt the very end")[0],
         "answer": coach["c01"]["answer"],
         "tool_calls": [{"tool": s["tool"], "args": s["args"]} for s in coach["c01"]["trace"]],
         "heat_result": coach["c01"]["trace"][0]["result"]},
    ]

    lat = {}
    for s in ("bare", "rag", "coach"):
        xs = [r["latency_s"] for r in rows if r["system"] == s]
        lat[s] = {"n": len(xs), "median_s": round(statistics.median(xs), 2),
                  "p95_s": round(pct(xs, 0.95), 2), "max_s": max(xs)}
    out["eval_run_latency"] = {"note": "wall-clock per answer, four answers running at once, "
                                       "2026-10-28; excludes the judge call", "systems": lat}
    (ROOT / "eval" / "error_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for g in out["groups"]:
        print(g["label"], g["targets"], g["misses"])
    print(out["totals"])
    print(json.dumps(lat))


if __name__ == "__main__":
    main()
