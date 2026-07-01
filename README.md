# Marathon Strategy RAG

A retrieval-augmented **marathon coach**. Ask a race-strategy question in plain
English — *"should I try to negative-split my first marathon?"*, *"how much
slower will 80°F make me?"* — and get practical, **cited** advice grounded in a
knowledge base built from real marathon science **and my own published data
analyses** of 30,000+ Boston finishers.

> 🌐 **Overview:** https://lyhjeremy.github.io/marathon-strategy-rag/

## Why
Generic running advice is everywhere, and a plain LLM will confidently make up
numbers. This coach only answers from a curated, citable knowledge base — so when
it says *"only 2.5% of runners actually negative-split, so aim for an even split
instead,"* that figure traces back to a real analysis, shown as a `[n]` citation.

Several cards are distilled from my own marathon projects:
- **Negative-Split Myth** — 2.5% of 31,912 Boston finishers ran a true negative split
- **Hitting the Wall** — pace inflects sharply at ~30 km
- **The Heat Tax** — ~1 minute slower per °F of race-day heat
- **Course Difficulty** — Boston & NYC are the hardest Majors
- **Super-Shoes** — ~67 s of the modern elite gain is footwear

…alongside general cards on fueling, tapering, carb-loading, hydration, pacing
and hills.

## How it works
```
question ─▶ local embedding ─▶ Chroma search over knowledge cards ─▶
          (all-MiniLM-L6-v2)      (each chunk keeps its title + source)
                                                    │
                                                    ▼
                    cited coaching answer ◀── Claude ◀── top passages as context
```
Retrieval is fully local and free (`sentence-transformers` + Chroma). Generation
runs on the **Claude CLI** by default (your Claude subscription, no per-token
cost); set `ANTHROPIC_API_KEY` to use the API instead.

## Quick start
```bash
pip install -r requirements.txt

python -m src.ingest                 # index the bundled knowledge base

python -m src.cli ask "how should I pace the last 10K to avoid hitting the wall?"
python -m src.cli chat               # interactive coach
```
The knowledge base ships **with the repo** (`knowledge/*.md`) — no downloads.

## Files
| Path | What it is |
|---|---|
| `knowledge/*.md` | The knowledge base — one citable card per topic |
| `src/ingest.py` | Parse cards → chunk → local embeddings → Chroma index |
| `src/embedder.py` | Local sentence-transformers embedder (free, offline) |
| `src/retriever.py` | Semantic search over the knowledge chunks |
| `src/coach.py` | The RAG chain: retrieve → grounded, cited coaching answer |
| `src/llm.py` | LLM wrapper — Claude CLI (default) or Anthropic API |
| `src/cli.py` | `ask` (one-shot) and `chat` (interactive) commands |

## Extending it
Drop a new `knowledge/<topic>.md` card (with a `# Title` and a `> Source:` line),
re-run `python -m src.ingest`, and the coach can cite it immediately.

## License
[MIT](LICENSE) © 2026 Jeremy Lee
