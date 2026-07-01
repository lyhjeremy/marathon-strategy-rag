<p align="center">
  <img src="assets/banner.png" alt="Marathon Strategy RAG" width="100%">
</p>

# A Marathon Coach That Cites Its Sources

*Turning my own running analyses into a retrieval-augmented coach.*

## The idea

Over several projects I've analysed tens of thousands of marathon finishers — how
rarely anyone negative-splits, where the wall actually hits, what heat costs you,
which Majors run slowest. Those findings are useful *advice*, but they live in
separate write-ups. What if a runner could just **ask a question** and get the
relevant finding back, phrased as coaching — and know it wasn't made up?

That's a textbook case for **RAG**. A general LLM will happily invent a
plausible-sounding "most runners negative-split by 2%." A RAG system instead
retrieves a real knowledge card and answers only from it, citing the source.

## The knowledge base

Rather than an external dataset, this project ships its own corpus: ~14 markdown
**knowledge cards** in `knowledge/`. Each has a title and a `> Source:` line.

Two kinds:
- **Grounded in my analyses** — negative-split reality (2.5% of 31,912 Boston
  finishers), the wall (~30 km inflection), the heat tax (~1 min/°F), Majors
  course difficulty, super-shoes (~67 s). Each states the real figure and cites
  the project it came from.
- **Established science** — carbohydrate fueling, tapering, carb-loading,
  hydration and sodium, even-pacing, the long run, race-week logistics, hill
  pacing.

Shipping the corpus in-repo means the whole thing is self-contained and
reproducible: clone, `pip install`, `python -m src.ingest`, ask.

## The pipeline

<p align="center">
  <img src="assets/architecture.png" alt="Marathon Strategy RAG pipeline" width="760">
</p>

1. **Parse & chunk.** Each card is split into overlapping ~120-word windows that
   respect paragraph boundaries, so a retrieved chunk is coherent on its own. The
   card's title is prepended to every chunk so it stays self-describing.
2. **Embed locally.** `all-MiniLM-L6-v2` (sentence-transformers) on the Mac GPU —
   free, offline, no API key.
3. **Store.** Chroma with cosine similarity; each chunk keeps `title`, `source`
   and `card` as metadata so citations survive retrieval.
4. **Retrieve & answer.** A question is embedded and matched; the top passages go
   to Claude with a strict coaching system prompt: answer only from the passages,
   cite each fact `[n]`, be practical. Generation runs on the Claude CLI (my
   subscription — no per-token cost) or the Anthropic API if a key is set.

## Why the citations matter

The value isn't just fluent advice — it's *auditable* advice. When the coach says
"aim for an even split, because only 2.5% of runners actually negative-split," the
`[1]` points to the exact card, which points to the exact analysis. A runner can
check the reasoning instead of trusting a black box. And because the knowledge is
data, not model weights, extending the coach is just dropping in a new markdown
card and re-indexing.

## Limitations & next steps

- The corpus is deliberately small and curated; it answers strategy questions
  well but isn't a general running encyclopedia.
- Retrieval is single-shot; a query that spans several topics (heat *and* fueling
  *and* pacing) would benefit from multi-query retrieval or a planning step — a
  natural bridge to the agentic [Marathon Training Plan Agent](https://github.com/lyhjeremy)
  companion project.
- No re-ranking beyond vector similarity yet; a cross-encoder would sharpen top-k.

*Code: [github.com/lyhjeremy/marathon-strategy-rag](https://github.com/lyhjeremy/marathon-strategy-rag)*
