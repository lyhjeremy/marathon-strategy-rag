"""Does hybrid retrieval or reranking beat plain dense search on these 14 cards?

Relevance labels are the `relevant_cards` field of each gold question (written
when the gold set was frozen, before any retriever was run on it). Scores are
at card level, after collapsing chunks to the card they came from:

    hit@1         the top card is a relevant one
    recall@3      share of a question's relevant cards found in the top 3 cards
    MRR           1 / rank of the first relevant card
    chunk_prec@5  share of the top 5 chunks (what the coach actually reads) that
                  come from a relevant card

A paired bootstrap over questions (2,000 resamples) gives a 95% interval for each
mode's MRR difference from dense.

    python -m src.retrieval_eval      # writes eval/retrieval_compare.json
"""

from __future__ import annotations

import json
import random
import time
from datetime import date

from .config import ROOT
from .gold import load_gold
from .hybrid import MODES, HybridRetriever

OUT = ROOT / "eval" / "retrieval_compare.json"


def _cards(ranking, corpus):
    seen, cards = set(), []
    for cid, _ in ranking:
        c = corpus.passage(cid).card
        if c not in seen:
            seen.add(c)
            cards.append(c)
    return cards


def score(ranking, corpus, relevant: set[str]) -> dict:
    cards = _cards(ranking, corpus)
    first = next((i for i, c in enumerate(cards) if c in relevant), None)
    top5 = [corpus.passage(cid).card for cid, _ in ranking[:5]]
    return {
        "hit1": float(bool(cards) and cards[0] in relevant),
        "recall3": len(relevant & set(cards[:3])) / len(relevant),
        "mrr": 0.0 if first is None else 1.0 / (first + 1),
        "chunk_prec5": sum(c in relevant for c in top5) / max(1, len(top5)),
    }


def main():
    gold = load_gold()
    results, per_q = {}, {}
    for mode in MODES:
        r = HybridRetriever(mode=mode)
        r.ranking("warm up the models")
        rows, t0 = [], time.perf_counter()
        for g in gold:
            rows.append(score(r.ranking(g["question"]), r.corpus, set(g["relevant_cards"])))
        ms = (time.perf_counter() - t0) / len(gold) * 1000
        per_q[mode] = rows
        results[mode] = {k: round(sum(x[k] for x in rows) / len(rows), 3) for k in rows[0]}
        results[mode]["ms_per_query"] = round(ms, 1)
        print(f"{mode:15s} " + "  ".join(f"{k}={v}" for k, v in results[mode].items()))

    rng = random.Random(7)
    n = len(gold)
    for mode in MODES:
        if mode == "dense":
            continue
        diffs = [per_q[mode][i]["mrr"] - per_q["dense"][i]["mrr"] for i in range(n)]
        boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
        results[mode]["mrr_diff_vs_dense"] = round(sum(diffs) / n, 3)
        results[mode]["mrr_diff_ci95"] = [round(boots[49], 3), round(boots[1949], 3)]
        print(f"  {mode}: MRR diff vs dense {sum(diffs)/n:+.3f} "
              f"[{boots[49]:+.3f}, {boots[1949]:+.3f}]")

    by_type = {}
    for mode in MODES:
        by_type[mode] = {}
        for typ in ("numeric", "strategy"):
            idx = [i for i, g in enumerate(gold) if g["type"] == typ]
            by_type[mode][typ] = round(sum(per_q[mode][i]["mrr"] for i in idx) / len(idx), 3)

    OUT.write_text(json.dumps({
        "generated": date.today().isoformat(),
        "questions": n,
        "chunks": len(HybridRetriever(mode="dense").corpus.ids),
        "modes": results,
        "mrr_by_question_type": by_type,
        "per_question": {m: [{"id": g["id"], **row} for g, row in zip(gold, per_q[m])]
                         for m in MODES},
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
