# A marathon coach that cites its sources

*Turning my own running analyses into a retrieval-augmented coach.*

## The idea

Over several projects I've analyzed tens of thousands of marathon finishers: how rarely anyone negative-splits, where the wall hits, what heat costs, which Majors run slowest. Those findings are useful advice, but they sit in separate write-ups. I wanted a runner to be able to ask a question, get the relevant finding back phrased as coaching, and be able to check where it came from.

That is a standard case for retrieval-augmented generation. A general LLM will happily produce a plausible statistic such as "most runners negative-split by 2%". A RAG system retrieves a real knowledge card and answers only from it, citing the source.

## The knowledge base

The project ships its own corpus: 14 markdown cards in `knowledge/`, each with a title and a `> Source:` line. There are two kinds.

Six are distilled from my analyses: negative-split reality (2.5% of 31,912 Boston finishers), the wall (an inflection near 30 km), the heat tax (about 1 minute per °F), Majors course difficulty, super-shoes (about 67 seconds of the elite gain) and Boston qualifying standards. Each states the figure and names the project it came from.

The other eight cover established training principles: carbohydrate fueling, tapering, carb-loading, hydration and sodium, even pacing, the long run, race-week logistics and hill pacing.

Because the corpus is in the repo, the whole thing is reproducible: clone, `pip install`, `python -m src.ingest`, ask.

## The pipeline

1. **Parse and chunk.** Each card is split into overlapping windows of about 120 words (30-word overlap) that respect paragraph boundaries, so a retrieved chunk reads on its own. The card title is prepended to every chunk.
2. **Embed locally.** `all-MiniLM-L6-v2` from sentence-transformers, on the Mac GPU through MPS when available. It is free, offline and needs no API key.
3. **Store.** Chroma with cosine similarity. Each chunk keeps `title`, `source` and `card` as metadata so citations survive retrieval.
4. **Retrieve and answer.** The question is embedded and the top five passages go to Claude with a strict system prompt: answer only from the passages, cite each fact as `[n]`, be practical. Generation runs on the Claude CLI (my subscription, no per-token cost) or the Anthropic API if a key is set.

## Why the citations matter

The point is auditable advice. When the coach says "aim for an even split, because only 2.5% of runners actually negative-split", the `[1]` points to the card, and the card names the analysis. A runner can check the reasoning. And because the knowledge is data rather than model weights, extending the coach means adding a markdown card and re-indexing.

## Does grounding help? Measured

`src/eval.py` scores the coach with an LLM judge, RAGAS-style, over seven gold questions. For each question it asks the same model the same question twice, once with retrieval and once without, and asks the judge to score both against the retrieved passages.

| Metric | Grounded coach | Bare model |
|---|---|---|
| Faithfulness to the knowledge base | 0.98 | 0.76 |
| Answer relevance | 0.95 | |
| Context precision | 0.78 | |
| Citation validity | 1.00 | |

Citation validity is computed in code: every `[n]` in every grounded answer pointed at a passage that was actually retrieved. The biggest per-question gap was the heat question, where the bare model scored 0.40 and the grounded coach 0.97; the heat card's per-degree figure comes from my own analysis, and a model without retrieval has no access to it. The weakest retrieval was the carbohydrate question, with a context precision of 0.40.

## The same chain in LangChain

`coach.py` wires retrieval, prompt and LLM by hand so the mechanics stay visible. `chain_lc.py` expresses the same flow as a LangChain LCEL pipeline, `{context: retrieve | format, query: passthrough} | prompt | llm`, so the system also runs on the framework most production RAG code uses (`--engine lc`).

## Limitations and next steps

- Seven questions is a small gold set, and all were written to be answerable from the cards. The judge is an LLM from the same family as the generator.
- Bare faithfulness is measured against the retrieved passages, so a correct claim the cards don't contain counts against the bare model.
- The corpus is small and curated. It handles strategy questions well and is not a general running encyclopedia.
- Retrieval is a single dense query. A question that spans heat, fueling and pacing would benefit from multi-query retrieval or a planning step, which leads to the agentic [marathon training agent](https://github.com/lyhjeremy/marathon-training-agent).
- There is no lexical search or reranking. The sibling [wine sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) adds hybrid BM25 and a cross-encoder, with a measured +21% precision gain.

*Code: [github.com/lyhjeremy/marathon-strategy-rag](https://github.com/lyhjeremy/marathon-strategy-rag)*
