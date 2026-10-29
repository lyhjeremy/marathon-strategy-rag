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

*(July 2026. 2026-10-28: superseded by the three-system evaluation in the October section below, which uses a different judge prompt and a pinned model, so the figures here are not directly comparable.)*

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

The course card's figures are simulated. `course-difficulty.md` gives Boston +101 s and New York +78 s against Berlin, and the study behind it used a simulated field with course penalties planted in code, so those numbers test a method and do not describe the courses. The card labels them as simulated in its title and next to every time figure, so the label travels with every retrieved chunk. Faithfulness would not catch a wrong figure here, since a grounded answer that repeats the card scores as faithful. (2026-10-28: the label was added after the published evaluation; the numbers are unchanged.) Bare faithfulness is measured against the retrieved passages, which means a correct claim the cards don't contain counts against the bare model.

The corpus is small and curated. It handles strategy questions well, and it is no running encyclopedia. Retrieval is a single dense query, so a question that spans heat, fueling and pacing would benefit from multi-query retrieval or a planning step (that line of thought led to the agentic [marathon training agent](https://github.com/lyhjeremy/marathon-training-agent)). There is also no lexical search or reranking. The sibling [wine sommelier RAG](https://github.com/lyhjeremy/wine-sommelier-rag) adds hybrid BM25 and a cross-encoder, with a measured +21% precision gain. (2026-10-28: the October upgrade added both here and measured no gain on this corpus; see below.)

## October 2026 upgrade

The July coach could quote a card's figure but could not compute anything from it. Ask what 75°F costs a 3:30 runner, or what a 1:42 half means for the finish, and the honest answer was "about a minute per degree" and "it depends". Two of my other projects were rebuilt the same week into small models with a Python function each: a finish-time and wall-risk predictor fitted on Boston splits ([marathon-pacing-decay](https://github.com/lyhjeremy/marathon-pacing-decay)), and a heat model fitted on Boston results against race-day weather ([marathon-heat-tax](https://github.com/lyhjeremy/marathon-heat-tax)). The upgrade gives the coach those models as tools, plus a pace table and a calculator, and tests whether that matters.

### Freezing the test first

Before running anything I wrote 43 questions and committed them (`eb413bf`). Twenty-six ask for numbers, 42 scored values in all; 17 are strategy questions, including July's seven. Each numeric question stores the tool calls that answer it, so the expected values are computed from the models rather than typed, and the tolerances are rules on the models' own output: a quarter of the 80% interval for a finish time, a quarter of the range for a heat slowdown, 5 points for a wall-risk percentage, 2 seconds for a pace. The eval checks the file's hash before it runs. The expected values were computed against the models before and after their final release and did not change.

### Retrieval: the reranker I expected to need

The July write-up blamed its weakest score on dense-only retrieval and pointed at the wine project's BM25-plus-cross-encoder setup. I built that here and scored five modes on the frozen questions. None beat plain dense search: mean reciprocal rank 0.911 dense, 0.929 BM25, 0.899 hybrid, 0.907 with reranking, every difference inside its bootstrap interval. With 46 chunks, dense search already puts a relevant card in the top three for 98% of questions. The coach kept dense search.

### The loop

The Claude CLI has no function-calling interface, so the agent loop is about a hundred lines of Python. Each turn the model returns one JSON object, a tool call or a final answer; the loop runs the tool, labels the result `[T1]`, `[T2]` and feeds it back, retries once on invalid JSON, and forces an answer after six tool calls. A check in code then pulls every number out of the answer and looks for it in the question, the tool results and the cited passages. The model results carry the caveats from their own holdout checks (the heat model did not beat a flat no-weather guess on 2015-2019; the pace model's late-race intervals ran narrow on 2024), and the coach is told to pass them on, which it did in the recorded runs. The same four domain tools are served over MCP, so Claude Desktop or Claude Code can call them directly.

One setup detail mattered more than expected: `claude -p` loads the user's settings and CLAUDE.md and has file tools, so a "bare" model run from the repo could read the cards off disk. Every call now runs with tools off and settings off, from a temporary directory.

### Results

Same model for everything (claude-sonnet-5-5), 129 answers:

| | Bare | RAG | Tool coach |
|---|---|---|---|
| Numeric values within tolerance | 14 of 42 | 21 of 42 | 42 of 42 |
| Answers with an unsourced number | 43 of 43 | 28 of 43 | 3 of 43 |
| Faithfulness (judge) | 0.33 | 0.88 | 0.90 |
| Answer relevance (judge) | 0.83 | 0.87 | 0.93 |
| Median seconds | 6.6 | 7.4 | 7.2 |

The perfect numeric score needs a caveat of its own: expected values come from the same models the coach calls, so it shows the loop chose the right tool and copied the result faithfully every time, not that the models are right. The more telling contrasts are elsewhere. Retrieval alone fixed the card-arithmetic questions and little else; on heat the RAG coach fell back on the July card's minute-per-degree rule and missed seven of nine heat values, the worst by 16 minutes. The number check fired on two-thirds of RAG answers, nearly always the model doing its own arithmetic, and on three coach answers. The judge barely separated the two coaches on faithfulness, and it gave a perfect faithfulness score to the coach's one failure, an answer that asked the runner for their finish time instead of answering. That failure is the most useful finding for the next version: when a question leaves out an input, the coach should assume a range and say so.

Full numbers, the per-question records and the limitations are in the README and on the [write-up page](https://lyhjeremy.github.io/marathon-strategy-rag/overview/).

*Code: [github.com/lyhjeremy/marathon-strategy-rag](https://github.com/lyhjeremy/marathon-strategy-rag)*
