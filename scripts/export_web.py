"""Write eval/web.json, the data file the write-up page draws from.

Reads the saved eval run (eval/runs/run_2026-10.jsonl), eval/summary.json and
eval/retrieval_compare.json. The traces are the coach's recorded runs from that
eval, copied as they happened (tool calls, arguments, results, the final answer);
long passage text is trimmed for the page.

    python -m scripts.export_web n01 c01 s15
"""

from __future__ import annotations

import json
import sys

from src.config import ROOT
from src.eval import RUN_PATH, SUMMARY_PATH, recheck
from src.gold import load_gold

EVAL = ROOT / "eval"


def _trim_result(tool: str, result: dict) -> dict:
    if tool == "retrieve_cards" and "passages" in result:
        return {"passages": [{"card": p["card"], "text": p["text"][:220] + "..."}
                             for p in result["passages"]]}
    if tool == "pace_table" and "splits" in result:
        r = dict(result)
        r["splits"] = [s for s in r["splits"] if s["km"] in (10, 21.098, 30, 42.195)]
        return r
    return result


def main(trace_ids: list[str]) -> None:
    rows = [json.loads(x) for x in RUN_PATH.read_text(encoding="utf-8").splitlines() if x]
    recheck(rows)
    gold = {g["id"]: g for g in load_gold()}
    summary = json.loads(SUMMARY_PATH.read_text())
    retrieval = json.loads((EVAL / "retrieval_compare.json").read_text())

    per_q = []
    for gid, g in gold.items():
        entry = {"id": gid, "type": g["type"], "q": g["question"]}
        for r in rows:
            if r["id"] != gid:
                continue
            s = r["system"]
            entry[s] = {
                "faith": r["judge"].get("faithfulness"),
                "unsourced": len(r["check"]["unsourced"]),
                "lat": r["latency_s"],
            }
            if r.get("numeric"):
                entry[s]["ok"] = sum(t["ok"] for t in r["numeric"])
                entry[s]["n"] = len(r["numeric"])
        per_q.append(entry)

    traces = []
    for tid in trace_ids:
        r = next(x for x in rows if x["id"] == tid and x["system"] == "coach")
        others = {x["system"]: x for x in rows if x["id"] == tid}
        traces.append({
            "id": tid, "date": r["date"], "model": r["model"], "question": gold[tid]["question"],
            "steps": [{"tool": s["tool"], "args": s["args"], "label": s["label"],
                       "result": _trim_result(s["tool"], s["result"])} for s in r["trace"]],
            "answer": r["answer"], "latency_s": r["latency_s"], "llm_calls": r["llm_calls"],
            "unsourced": r["check"]["unsourced"],
            "numeric": r.get("numeric"),
            "others": {s: {"numeric": o.get("numeric"), "unsourced": o["check"]["unsourced"]}
                       for s, o in others.items() if s != "coach"},
        })

    out = {"summary": summary, "retrieval": {k: v for k, v in retrieval.items()
                                             if k != "per_question"},
           "questions": per_q, "traces": traces}
    path = EVAL / "web.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1:])
