"""Does grounding actually make the coach more trustworthy? Measure it.

This is a compact, RAGAS-style evaluation of the RAG coach, using an LLM as the
judge. For a set of gold questions it scores:

* **faithfulness**   — are the answer's claims supported by the retrieved passages?
* **answer relevance** — does the answer actually address the question?
* **context precision** — are the retrieved passages relevant to the question?
* **citation validity** (deterministic) — do the `[n]` markers point at real
  retrieved passages, not invented indices?

It also runs each question through a **bare LLM with no retrieval** and scores
that answer's faithfulness against the same knowledge base — the gap between the
two is the value RAG adds. Answers and judgements come from Claude; the judge is
asked for strict JSON so scoring is automatic.

Run:  python -m src.eval        # writes eval/results.csv + eval/faithfulness.png
"""

from __future__ import annotations

import csv
import json
import re

from .coach import Coach, _format_context
from .config import ROOT
from .llm import LLM

EVAL_DIR = ROOT / "eval"

# Questions the bundled knowledge base can answer well.
GOLD = [
    "Should I try to negative-split my first marathon?",
    "How much slower will race-day heat around 80°F make me?",
    "How should I pace the final 10K to avoid hitting the wall?",
    "How many grams of carbs per hour should I take during the race?",
    "How should I taper in the last two to three weeks?",
    "Do carbon-plated super-shoes actually make a meaningful difference?",
    "How should I approach a downhill course like Boston?",
]

JUDGE_SYSTEM = (
    "You are a rigorous RAG evaluator. You will be given a QUESTION, the CONTEXT "
    "passages retrieved from a marathon knowledge base, a GROUNDED answer written "
    "from that context, and an UNGROUNDED answer written with no context. Score "
    "strictly. Respond with ONLY a JSON object and nothing else."
)

JUDGE_PROMPT = """QUESTION:
{question}

CONTEXT (the knowledge base passages retrieved for this question):
{context}

GROUNDED answer (should use only the context):
{grounded}

UNGROUNDED answer (written with no context):
{ungrounded}

Score each field from 0.0 to 1.0 and return exactly this JSON:
{{
  "grounded_faithfulness": <fraction of the GROUNDED answer's factual claims supported by CONTEXT>,
  "ungrounded_faithfulness": <fraction of the UNGROUNDED answer's factual claims consistent with CONTEXT>,
  "answer_relevance": <how well the GROUNDED answer addresses the QUESTION>,
  "context_precision": <fraction of CONTEXT passages relevant to the QUESTION>
}}"""

UNGROUNDED_SYSTEM = (
    "You are a marathon coach. Answer the runner's question concisely with "
    "specific, practical advice."
)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _citation_validity(answer_text: str, n_passages: int) -> float | None:
    """Fraction of [n] citations in the answer that map to a real passage."""
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", answer_text)]
    if not cited:
        return 0.0
    valid = sum(1 for n in cited if 1 <= n <= n_passages)
    return valid / len(cited)


def evaluate():
    coach = Coach()
    llm = LLM()
    rows = []

    for q in GOLD:
        grounded = coach.ask(q)
        context = _format_context(grounded.sources)
        ungrounded = llm.complete(q, system=UNGROUNDED_SYSTEM)

        verdict = _extract_json(
            llm.complete(
                JUDGE_PROMPT.format(
                    question=q, context=context,
                    grounded=grounded.text, ungrounded=ungrounded,
                ),
                system=JUDGE_SYSTEM,
            )
        )
        if verdict is None:
            print(f"  (judge returned no JSON for: {q[:40]}… — skipped)")
            continue

        rows.append({
            "question": q,
            "grounded_faithfulness": float(verdict.get("grounded_faithfulness", 0)),
            "ungrounded_faithfulness": float(verdict.get("ungrounded_faithfulness", 0)),
            "answer_relevance": float(verdict.get("answer_relevance", 0)),
            "context_precision": float(verdict.get("context_precision", 0)),
            "citation_validity": _citation_validity(grounded.text, len(grounded.sources)),
        })
        print(f"  scored: {q[:50]}…")

    return rows


def _means(rows) -> dict:
    keys = ["grounded_faithfulness", "ungrounded_faithfulness",
            "answer_relevance", "context_precision", "citation_validity"]
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def _print_table(m):
    print("\nRAG coach evaluation (mean over %d questions)" % len(GOLD))
    print("-" * 46)
    print(f"  faithfulness (grounded)   {m['grounded_faithfulness']:.2f}")
    print(f"  faithfulness (no RAG)     {m['ungrounded_faithfulness']:.2f}")
    print(f"  answer relevance          {m['answer_relevance']:.2f}")
    print(f"  context precision         {m['context_precision']:.2f}")
    print(f"  citation validity         {m['citation_validity']:.2f}\n")


def _write_csv(rows):
    EVAL_DIR.mkdir(exist_ok=True)
    path = EVAL_DIR / "results.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def _write_figure(m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not installed — skipping figure)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})

    # Panel A — the value of grounding
    fa = [m["ungrounded_faithfulness"], m["grounded_faithfulness"]]
    bars = ax1.bar(["no RAG", "grounded RAG"], fa, color=["#B4B0A8", "#2A9D8F"], width=0.6)
    for b, v in zip(bars, fa):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Faithfulness to the\nknowledge base", fontsize=11)
    ax1.set_ylabel("score")
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel B — the grounded coach scorecard
    labels = ["faithfulness", "answer\nrelevance", "context\nprecision", "citation\nvalidity"]
    vals = [m["grounded_faithfulness"], m["answer_relevance"],
            m["context_precision"], m["citation_validity"]]
    bars = ax2.bar(labels, vals, color="#4A7BA6", width=0.62)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Grounded coach — RAGAS-style scorecard", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Retrieval grounding, evaluated with an LLM judge", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    EVAL_DIR.mkdir(exist_ok=True)
    path = EVAL_DIR / "faithfulness.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    rows = evaluate()
    if not rows:
        print("No questions scored — check the LLM backend.")
        return
    m = _means(rows)
    _print_table(m)
    _write_csv(rows)
    _write_figure(m)


if __name__ == "__main__":
    main()
