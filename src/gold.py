"""Load the gold set and compute each numeric question's expected values.

`eval/gold.jsonl` stores, for every numeric question, the tool calls that answer
it (`let`) and an expression for each target value and its tolerance. Nothing
numeric is typed by hand: expected values come from running the models, the
pace table or the card facts, and tolerances are fixed rules applied to the
models' own outputs:

    finish times         +/- max(1 min, a quarter of the model's 80% interval width)
    heat slowdowns       +/- max(0.5 min, a quarter of the model's lo-hi range)
    wall-risk percent    +/- 5 percentage points
    pace                 +/- 2 s per km or mile (display rounding), widened when
                         the pace depends on a model estimate
    half and 30 km split +/- 0.5 min
    card arithmetic      +/- max(0.5, 1% of the value)

    python -m src.gold          # print every expected value and tolerance
    python -m src.gold --write  # also write eval/gold_expected.json
"""

from __future__ import annotations

import json
import sys

from .config import ROOT
from .tools import call_tool
from .tools.card_facts import value as fact

GOLD_PATH = ROOT / "eval" / "gold.jsonl"
EXPECTED_PATH = ROOT / "eval" / "gold_expected.json"

TOL_PROB = 5.0
TOL_PACE = 2.0
TOL_SPLIT = 0.5
SEC_PER_KM_PER_MIN = 60 / 42.195


def tol_finish(p: dict) -> float:
    return max(1.0, 0.25 * (p["hi80_min"] - p["lo80_min"]))


def tol_heat(h: dict) -> float:
    return max(0.5, 0.25 * (h["hi_min"] - h["lo_min"]))


def tol_card(v: float) -> float:
    return max(0.5, 0.01 * abs(v))


_ENV = {"__builtins__": {}, "max": max, "min": min, "abs": abs, "round": round,
        "fact": fact, "tol_finish": tol_finish, "tol_heat": tol_heat, "tol_card": tol_card,
        "TOL_PROB": TOL_PROB, "TOL_PACE": TOL_PACE, "TOL_SPLIT": TOL_SPLIT,
        "SEC_PER_KM_PER_MIN": SEC_PER_KM_PER_MIN}


def _resolve(x, names: dict):
    if isinstance(x, str) and x.startswith("="):
        return eval(x[1:], dict(_ENV), dict(names))  # expressions are repo-authored
    if isinstance(x, dict):
        return {k: _resolve(v, names) for k, v in x.items() if v is not None}
    return x


def load_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def expected_for(item: dict) -> dict:
    """Run the question's tool calls and evaluate its targets."""
    names, calls = {}, []
    for step in item.get("let", []):
        args = _resolve(step["args"], names)
        out = call_tool(step["tool"], args)
        if "error" in out:
            raise RuntimeError(f"{item['id']}: {step['tool']} failed: {out['error']}")
        names[step["name"]] = out
        calls.append({"tool": step["tool"], "args": args})
    targets = []
    for t in item["targets"]:
        targets.append({"key": t["key"], "unit": t["unit"],
                        "expected": round(float(_resolve(t["value"], names)), 3),
                        "tol": round(float(_resolve(t["tol"], names)), 3)})
    return {"id": item["id"], "calls": calls, "targets": targets}


def all_expected() -> dict:
    from .tools import models

    out = {"model_sources": models.source_info(), "items": {}}
    for item in load_gold():
        if item["type"] == "numeric":
            out["items"][item["id"]] = expected_for(item)
    return out


if __name__ == "__main__":
    exp = all_expected()
    for qid, e in exp["items"].items():
        print(qid, "  ".join(f"{t['key']}={t['expected']}±{t['tol']}" for t in e["targets"]))
    if "--write" in sys.argv:
        EXPECTED_PATH.write_text(json.dumps(exp, indent=1), encoding="utf-8")
        print(f"wrote {EXPECTED_PATH}")
