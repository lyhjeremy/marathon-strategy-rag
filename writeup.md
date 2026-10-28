# A marathon coach that cites its sources

*Turning my own running analyses into a retrieval-augmented coach.*

## Where it started

I have run 14 marathons, and over several projects I have also analyzed tens of thousands of other people's: how rarely anyone negative-splits, where the wall hits, what heat costs, which Majors run slowest. Each of those findings ended up in its own write-up. What I wanted was one place a runner could ask a question, get the relevant finding back as coaching, and see which analysis it came from.

Retrieval-augmented generation is built for that. A general LLM will happily make up a plausible statistic ("most runners negative-split by 2%"). A RAG system looks up a knowledge card first, answers from what it found and cites the card, so the number can be checked.

## The knowledge base

The corpus ships with the project as 14 markdown cards in `knowledge/`, each with a title and a `> Source:` line. Six are distilled from my analyses, and each states its figure and names the project behind it: negative-split reality (2.5% of 31,912 Boston finishers), the wall (an inflection near 30 km), the heat tax (about 1 minute per °F), Majors course difficulty, super-shoes (about 67 seconds of the elite gain) and Boston qualifying standards. The other eight cover established training principles: carbohydrate fueling, tapering, carb-loading, hydration and sodium, even pacing, the long run, race-week logistics and hill pacing.

Since the cards live in the repo, anyone can reproduce the whole thing: clone, `pip install`, `python -m src.ingest`, ask.

## The pipeline

Ingestion splits each card into windows of about 120 words with a 30-word overlap, keeping paragraph boundaries intact so a retrieved chunk reads on its own, and prepends the card title to every chunk. The chunks are embedded locally with `all-MiniLM-L6-v2` from sentence-transformers (on the Mac GPU through MPS when it is available), which costs nothing and works offline. They go into Chroma with cosine similarity, and each one keeps `title`, `source` and `card` as metadata so the citation survives retrieval.

At question time the query is embedded, and the top five passages go to Claude with a strict system prompt: answer only from the passages, cite each fact as `[n]`, keep it usable on a training run. Generation runs on the Claude CLI (my subscription, so no per-token cost) or on the Anthropic API if a key is set.

The citations are what make the advice auditable. When the coach says "aim for an even split, because only 2.5% of runners negative-split", the `[1]` points to the card and the card names the analysis, so a runner can follow the reasoning back to the data. The knowledge also sits in files instead of model weights; extending the coach means writing a markdown card and re-indexing.

## Does grounding help?

`src/eval.py` scores the coach with an LLM judge, RAGAS-style, over seven gold questions. Each question goes to the same model twice, once with retrieval and once without, and the judge scores both answers against the retrieved passages.

| Metric | Grounded coach | Bare model |
|---|---|---|
| Faithfulness to the knowledge base | 0.98 | 0.76 |
| Answer relevance | 0.95 | |
| Context precision | 0.78 | |
| Citation validity | 1.00 | |

Citation validity is computed in code: every `[n]` in every grounded answer pointed at a passage that had been retrieved. The widest per-question gap was on heat, where the bare model scored 0.40 and the grounded coach 0.97. That card's per-degree figure comes from my own analysis, and a model without retrieval has no access to it. Retrieval was weakest on the carbohydrate question, with a context precision of 0.40.

## The same chain in LangChain

`coach.py` wires retrieval, prompt and LLM by hand so the mechanics stay visible. `chain_lc.py` builds the same flow as a LangChain LCEL pipeline, `{context: retrieve | format, query: passthrough} | prompt | llm`, so it also runs on the framework most production RAG code uses (`--engine lc`).

## Limitations and next steps

Seven questions is a small gold set, and I wrote all seven to be answerable from the cards; the judge is also an LLM from the same family as the generator. The carbohydrate question's 0.40 context precision has a visible cause: three of its top five passages come from the carb-loading card, which covers the days before the race, because the dense retriever matches on "carbs" and not on "during the race".

One card is out of date. `course-difficulty.md` gives Boston +101 s and New York +78 s as properties of the real courses, but the study behind it used a simulated field with penalties set in code. Faithfulness can't catch this, since a grounded answer that repeats the card scores as faithful. Bare faithfulness is measured against the retrieved passages, which means a correct claim the cards don't contain counts against the bare model.

The corpus is small and curated. It handles strategy questions well, and it is no running encyclopedia. Retrieval is a single dense query, so a question that spans heat, fueling and pacing would benefit from multi-query retrieval or a planning step (that line of thought led to the agentic [marathon training agent](https://github.com/lyhjeremy/marathon-training-agent)). There is also no lexical search or reranking. The sibling [wine sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) adds hybrid BM25 and a cross-encoder, with a measured +21% precision gain.

*Code: [github.com/lyhjeremy/marathon-strategy-rag](https://github.com/lyhjeremy/marathon-strategy-rag)*
