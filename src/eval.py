"""Three systems, one frozen gold set: does a coach that computes beat one that only retrieves?

Systems (same model for all, and for the judge; see `llm.DEFAULT_CLI_MODEL`):

    bare    the model alone, no retrieval, no tools
    rag     the July 2026 RAG coach: dense top-5 passages, answer with [n] citations
    coach   the tool-using agent in src/agent.py

Every system gets the identical question text. Numeric questions add one line
asking for a machine-readable ``FINAL: key=value; ...`` line, which is scored in
code against the expected values from `src/gold.py` (within tolerance or not).

Per answer it also records:

* the LLM judge's faithfulness (claims supported by the system's own context:
  passages for rag, passages plus tool results for coach; the bare answer is
  judged against the passages the RAG retrieved for that question, as in July),
  answer relevance, and context precision (rag and coach only)
* citation validity: share of [n] / [Tn] markers that point at something real
* the unsourced-number check from `src/numcheck.py`
* model calls, tool steps and wall-clock latency

    python -m src.eval                 # full run, resumable; writes eval/runs/ and eval/summary.json
    python -m src.eval --only n01,s03  # a subset
    python -m src.eval --summarize     # rebuild summary.json from the saved run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import numcheck
from .agent import ToolCoach
from .coach import Coach, _format_context
from .config import ROOT
from .gold import EXPECTED_PATH, GOLD_PATH, all_expected, load_gold
from .llm import DEFAULT_CLI_MODEL, LLM
from .tools import pace

EVAL_DIR = ROOT / "eval"
RUN_PATH = EVAL_DIR / "runs" / "run_2026-10.jsonl"
SUMMARY_PATH = EVAL_DIR / "summary.json"
SYSTEMS = ("bare", "rag", "coach")

BARE_SYSTEM = (
    "You are a marathon coach. Answer the runner's question concisely with "
    "specific, practical advice.")

JUDGE_SYSTEM = (
    "You are a rigorous evaluator of answers from a marathon-coaching assistant. "
    "Score strictly. Respond with ONLY a JSON object and nothing else.")

JUDGE_PROMPT = """QUESTION:
{question}

CONTEXT (everything the assistant was allowed to rely on: knowledge-card passages and, where present, tool results):
{context}

ANSWER:
{answer}

Score each field from 0.0 to 1.0 and return exactly this JSON:
{{
  "faithfulness": <fraction of the ANSWER's factual claims and numbers that are supported by CONTEXT>,
  "answer_relevance": <how well the ANSWER addresses the QUESTION>,
  "context_precision": <fraction of CONTEXT items that are relevant to the QUESTION>
}}"""


def final_instruction(item: dict) -> str:
    keys = "; ".join(f"{t['key']}=<number>" for t in item["targets"])
    units = ", ".join(f"{t['key']} in {t['unit']}" for t in item["targets"])
    return (f"\n\nAt the very end of your answer, add one line in exactly this form, "
            f"with plain numbers and no units: FINAL: {keys}  ({units})")


def question_text(item: dict) -> str:
    q = item["question"]
    return q + final_instruction(item) if item["type"] == "numeric" else q


def parse_final(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if "FINAL:" in ln]
    if not lines:
        return {}
    out = {}
    for key, val in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;\s)]+)", lines[-1].split("FINAL:", 1)[1]):
        val = val.strip().rstrip(".,%").replace(",", "")
        try:
            if ":" in val:
                v = pace.parse_time(val)
                if key.startswith("pace"):          # m:ss pace -> seconds
                    a, b = val.split(":")[:2]
                    v = int(a) * 60 + float(b)
                out[key] = v
            else:
                out[key] = float(val)
        except ValueError:
            continue
    return out


def score_numeric(answer: str, expected: dict) -> list[dict]:
    got = parse_final(answer)
    rows = []
    for t in expected["targets"]:
        v = got.get(t["key"])
        rows.append({"key": t["key"], "expected": t["expected"], "tol": t["tol"], "got": v,
                     "ok": v is not None and abs(v - t["expected"]) <= t["tol"],
                     "abs_err": None if v is None else round(abs(v - t["expected"]), 3)})
    return rows


def citation_validity(answer: str, n_passages: int, n_tools: int = 0) -> float | None:
    body = re.sub(r"(?m)^.*FINAL:.*$", "", answer)
    refs = re.findall(r"\[(T?\d+(?:\s*[,;]\s*T?\d+)*)\]", body)
    marks = [m.strip() for grp in refs for m in re.split(r"[,;]", grp)]
    if not marks:
        return None
    ok = 0
    for m in marks:
        if m.startswith("T"):
            ok += 1 <= int(m[1:]) <= n_tools
        else:
            ok += 1 <= int(m) <= n_passages
    return ok / len(marks)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class Runner:
    def __init__(self):
        self.llm = LLM()
        self.rag = Coach()
        self.lock = threading.Lock()

    def _judge(self, question: str, context: str, answer: str) -> dict:
        llm = LLM()
        for _ in range(2):
            v = _extract_json(llm.complete(
                JUDGE_PROMPT.format(question=question, context=context or "(none)",
                                    answer=answer), system=JUDGE_SYSTEM))
            if v:
                return {k: float(v.get(k, 0)) for k in
                        ("faithfulness", "answer_relevance", "context_precision")}
        return {}

    def run_one(self, item: dict, system: str, expected: dict | None) -> dict:
        q = question_text(item)
        rec = {"id": item["id"], "type": item["type"], "system": system, "question": q}
        t0 = time.perf_counter()
        if system == "bare":
            llm = LLM()
            answer = llm.complete(q, system=BARE_SYSTEM)
            rec.update(latency_s=round(time.perf_counter() - t0, 2), llm_calls=1, steps=0,
                       model=llm.last_meta.get("model"))
            with self.lock:
                passages = self.rag.retriever.search(item["question"], k=5)
            judge_ctx = _format_context(passages)
            rec["check"] = numcheck.check(answer, q, [])
            rec["citation_validity"] = None
        elif system == "rag":
            llm = LLM()
            with self.lock:
                passages = self.rag.retriever.search(item["question"], k=5)
            from .coach import PROMPT, SYSTEM
            answer = llm.complete(PROMPT.format(query=q, context=_format_context(passages)),
                                  system=SYSTEM)
            rec.update(latency_s=round(time.perf_counter() - t0, 2), llm_calls=1, steps=0,
                       model=llm.last_meta.get("model"))
            judge_ctx = _format_context(passages)
            cited = {int(n) for grp in re.findall(r"\[([\d,\s]+)\]", answer)
                     for n in re.findall(r"\d+", grp)}
            rec["check"] = numcheck.check(answer, q, [p.text for i, p in enumerate(passages, 1)
                                                      if i in cited])
            rec["citation_validity"] = citation_validity(answer, len(passages))
            rec["passages"] = [p.card for p in passages]
        else:
            res = ToolCoach(llm=LLM()).run(q)
            answer = res.answer
            rec.update(latency_s=res.latency_s, llm_calls=res.llm_calls, steps=len(res.steps),
                       model=res.model, repairs=res.repairs, parse_failed=res.parse_failed,
                       hit_step_cap=res.hit_step_cap,
                       trace=[{"tool": s["tool"], "args": s["args"], "label": s["label"],
                               "result": s["result"], "tool_s": s.get("tool_s")}
                              for s in res.steps])
            ctx = [f"[{i}] ({p['title']}) {p['text']}" for i, p in enumerate(res.passages, 1)]
            ctx += [f"[T{i}] {s['tool']}: {json.dumps(s['result'], ensure_ascii=False)}"
                    for i, s in enumerate(res.steps, 1) if s["tool"] != "retrieve_cards"]
            judge_ctx = "\n\n".join(ctx)
            rec["check"] = res.check
            rec["citation_validity"] = citation_validity(answer, len(res.passages), len(res.steps))
        rec["answer"] = answer
        if expected:
            rec["numeric"] = score_numeric(answer, expected)
        rec["judge"] = self._judge(item["question"], judge_ctx, answer)
        if system == "bare":
            rec["judge"]["context_precision"] = None
        rec["date"] = date.today().isoformat()
        return rec


def _load_run() -> list[dict]:
    if not RUN_PATH.exists():
        return []
    return [json.loads(line) for line in RUN_PATH.read_text(encoding="utf-8").splitlines() if line]


def verify_freeze() -> dict:
    freeze = json.loads((EVAL_DIR / "gold_freeze.json").read_text())
    digest = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    if digest != freeze["sha256"]:
        raise SystemExit("eval/gold.jsonl differs from the frozen version; refusing to run.")
    return freeze


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize() -> dict:
    rows = _load_run()
    gold = {g["id"]: g for g in load_gold()}
    out = {"generated": date.today().isoformat(), "freeze": verify_freeze(),
           "models": sorted({r.get("model") or "" for r in rows}), "systems": {}}
    for s in SYSTEMS:
        rs = [r for r in rows if r["system"] == s]
        if not rs:
            continue
        num = [t for r in rs for t in r.get("numeric", [])]
        numq = [r for r in rs if r["type"] == "numeric"]
        strat = [r for r in rs if r["type"] == "strategy"]
        lat = [r["latency_s"] for r in rs]
        out["systems"][s] = {
            "answers": len(rs),
            "numeric_targets": len(num),
            "numeric_accuracy": _mean([float(t["ok"]) for t in num]),
            "numeric_questions_all_correct": _mean(
                [float(all(t["ok"] for t in r["numeric"])) for r in numq]),
            "final_line_missing": sum(1 for t in num if t["got"] is None),
            "faithfulness": _mean([r["judge"].get("faithfulness") for r in rs]),
            "faithfulness_strategy": _mean([r["judge"].get("faithfulness") for r in strat]),
            "faithfulness_numeric": _mean([r["judge"].get("faithfulness") for r in numq]),
            "answer_relevance": _mean([r["judge"].get("answer_relevance") for r in rs]),
            "context_precision": _mean([r["judge"].get("context_precision") for r in rs]),
            "citation_validity": _mean([r.get("citation_validity") for r in rs]),
            "answers_without_citations": sum(1 for r in rs if r.get("citation_validity") is None),
            "unsourced_fire_rate": _mean([float(r["check"]["fired"]) for r in rs]),
            "unsourced_numbers_share": round(
                sum(len(r["check"]["unsourced"]) for r in rs)
                / max(1, sum(r["check"]["numbers"] for r in rs)), 3),
            "mean_llm_calls": _mean([r["llm_calls"] for r in rs]),
            "mean_tool_steps": _mean([r["steps"] for r in rs]),
            "median_latency_s": round(statistics.median(lat), 1),
            "mean_latency_s": round(statistics.mean(lat), 1),
            "judge_missing": sum(1 for r in rs if not r["judge"].get("faithfulness") and
                                 r["judge"].get("faithfulness") != 0.0),
        }
        if s == "coach":
            out["systems"][s]["repairs"] = sum(r.get("repairs", 0) for r in rs)
            out["systems"][s]["parse_failures"] = sum(1 for r in rs if r.get("parse_failed"))
            out["systems"][s]["hit_step_cap"] = sum(1 for r in rs if r.get("hit_step_cap"))
    # numeric accuracy by question family (first letter of the id)
    fam = {"n": "predict_finish", "h": "heat_effect", "p": "pace_table", "c": "chained",
           "k": "card arithmetic"}
    out["numeric_by_family"] = {}
    for s in SYSTEMS:
        for prefix, name in fam.items():
            ts = [t for r in rows if r["system"] == s and r["id"].startswith(prefix)
                  for t in r.get("numeric", [])]
            if ts:
                out["numeric_by_family"].setdefault(name, {})[s] = _mean([float(t["ok"]) for t in ts])
    out["questions"] = len(gold)
    SUMMARY_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    verify_freeze()
    if args.summarize:
        print(json.dumps(summarize()["systems"], indent=1))
        return

    expected = all_expected()
    EXPECTED_PATH.write_text(json.dumps(expected, indent=1), encoding="utf-8")
    gold = load_gold()
    if args.only:
        keep = set(args.only.split(","))
        gold = [g for g in gold if g["id"] in keep]
    systems = args.systems.split(",")
    done = {(r["id"], r["system"]) for r in _load_run()}
    jobs = [(g, s) for g in gold for s in systems if (g["id"], s) not in done]
    print(f"model alias: {DEFAULT_CLI_MODEL}; {len(jobs)} answers to produce")
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    runner = Runner()
    write_lock = threading.Lock()

    def work(job):
        g, s = job
        try:
            rec = runner.run_one(g, s, expected["items"].get(g["id"]))
        except Exception as exc:  # keep going; the job is retried on the next run
            print(f"  FAILED {g['id']} {s}: {exc}")
            return
        with write_lock:
            with open(RUN_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        ok = ""
        if rec.get("numeric"):
            ok = f" numeric {sum(t['ok'] for t in rec['numeric'])}/{len(rec['numeric'])}"
        print(f"  {g['id']} {s:5s} {rec['latency_s']:6.1f}s{ok} "
              f"faith={rec['judge'].get('faithfulness')} unsourced={rec['check']['unsourced']}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, jobs))
    print(json.dumps(summarize()["systems"], indent=1))


if __name__ == "__main__":
    main()
