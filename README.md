# Marathon strategy coach

A marathon coach that retrieves from 14 knowledge cards and calls tools: a finish-time and wall-risk model fitted on Boston splits, a heat model fitted on Boston results against race-day weather, and a pace table. Every number in an answer is supposed to come from a tool result or a cited card, and a check in code flags the ones that don't. The tools are also served over MCP.

[Live site](https://lyhjeremy.github.io/marathon-strategy-rag/) · [Write-up](https://lyhjeremy.github.io/marathon-strategy-rag/overview/) · [Long-form notes](writeup.md)

Design and copy follow [these standards](https://github.com/lyhjeremy/lyhjeremy/blob/main/DESIGN_STANDARDS.md).

## Results (October 2026)

Three systems answered the same 43 frozen questions with the same model (claude-sonnet-5-5 through the Claude CLI): the model alone, the July RAG coach (dense top-5 passages, `[n]` citations) and the tool coach in `src/agent.py`. 26 questions ask for numbers, 42 scored values in all; 17 are strategy questions.

| Metric | Bare model | RAG coach | Tool coach |
|---|---|---|---|
| Numeric targets within tolerance (of 42) | 14 (33%) | 21 (50%) | 42 (100%) |
| Numeric questions fully right (of 26) | 27% | 50% | 100% |
| Faithfulness (judge) | 0.33 | 0.88 | 0.90 |
| Faithfulness, strategy questions | 0.32 | 0.91 | 0.94 |
| Answer relevance (judge) | 0.83 | 0.87 | 0.93 |
| Context precision (judge) | n/a | 0.70 | 0.84 |
| Citation validity (code) | n/a | 1.00 | 1.00 |
| Answers with an unsourced number | 100% | 65% | 7% |
| Share of all numbers unsourced | 77% | 19% | 0.5% |
| Model calls per answer | 1 | 1 | 2.3 |
| Tool calls per answer | 0 | 0 | 1.3 |
| Median seconds per answer | 6.6 | 7.4 | 7.2 |

Numeric accuracy by question family (bare / RAG / tool coach): finish-time model 50% / 50% / 100%, heat model 0% / 22% / 100%, pace table 57% / 57% / 100%, chained heat-then-pace and finish-then-pace 0% / 0% / 100%, card arithmetic 38% / 100% / 100%.

The tool coach's 42 of 42 is close to circular: expected values come from the same models it calls, so the numeric score measures whether a system reproduces the models. A perfect score means the loop chose the right tool, passed the right arguments and copied the result back on all 26 numeric questions, with no JSON repairs and no step-cap stops. Without tools, retrieval settles the card-arithmetic questions and little else. On heat the RAG coach applied the July card's minute-per-degree rule (about 25 minutes for a 3:30 runner at 75°F, against the model's 16.9); on wall risk the bare model guessed 25% to 65% where the model said 0% to 19%.

The unsourced-number check fired on 28 of 43 RAG answers, mostly the model's own arithmetic ("a 1:42 half is about 4:50/km"), and on 3 of 43 tool-coach answers (a "1 in 6" paraphrase of 17.7%, and two uncited "30 km" remarks). One coach answer failed: asked how much 80°F would cost, it asked for the runner's finish time instead of answering (relevance 0.30; faithfulness 1.00, because an answer with no claims has nothing unsupported). Latencies are wall-clock with four answers running at once.

Every answer, trace and score is in `eval/runs/run_2026-10.jsonl`; the aggregates are in `eval/summary.json`.

### The gold set

`eval/gold.jsonl` was committed at `eb413bf` before any system ran on it. Its SHA-256 is in `eval/gold_freeze.json`, and `src/eval.py` refuses to run if the file changes. A numeric question stores the tool calls that answer it plus an expression for each target and its tolerance; `src/gold.py` runs them, so expected values come from the models, the pace table or card figures (`src/tools/card_facts.py`, each figure checked against its card's text), never from typing. Tolerances are fixed rules on the model output:

| Target | Tolerance |
|---|---|
| Finish time | a quarter of the model's 80% interval width, at least 1 minute |
| Heat slowdown | a quarter of the model's lo-hi range, at least 0.5 minutes |
| Wall-risk percent | 5 percentage points |
| Pace | 2 seconds (wider when it depends on a model estimate) |
| Half or 30 km split | 0.5 minutes |
| Card arithmetic | 1% of the value, at least 0.5 |

The expected values were computed twice: once in phase 1 against the sibling repos' pre-release models, and again against the final vendored models. They came out identical.

### Retrieval: hybrid and reranking made no difference here

`src/hybrid.py` adds BM25, reciprocal rank fusion and the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (the approach from [wine-sommelier-rag](https://github.com/lyhjeremy/wine-sommelier-rag)). `src/retrieval_eval.py` scores each mode on the gold questions, using card-level relevance labels written with the gold set:

| Mode | Hit@1 | Recall@3 | MRR | MRR vs dense, 95% CI | ms/query |
|---|---|---|---|---|---|
| dense | 0.84 | 0.98 | 0.911 | baseline | 23 |
| bm25 | 0.88 | 0.92 | 0.929 | +0.018 [-0.062, +0.094] | 0.1 |
| hybrid (RRF) | 0.81 | 0.97 | 0.899 | -0.012 [-0.054, +0.027] | 8 |
| dense + rerank | 0.84 | 0.98 | 0.907 | -0.004 [-0.070, +0.058] | 115 |
| hybrid + rerank | 0.84 | 0.97 | 0.907 | -0.004 [-0.062, +0.058] | 102 |

With 46 chunks, dense search already puts a relevant card in its top three for 98% of questions, so there is little left to fix. The coach keeps dense search; `MSR_RETRIEVAL=hybrid+rerank` switches it.

## How it works

The knowledge base is 14 markdown cards in `knowledge/`, each with a `# Title` and a `> Source:` line. Eight cover established training principles. Six summarize my published analyses: [negative splits](https://lyhjeremy.github.io/marathon-negative-split-myth/), [the wall](https://lyhjeremy.github.io/marathon-pacing-decay/), [the heat tax](https://lyhjeremy.github.io/marathon-heat-tax/), [Majors course difficulty](https://lyhjeremy.github.io/marathon-majors-course-difficulty/) (simulated field, labeled as such on the card), [super-shoes](https://lyhjeremy.github.io/marathon-shoe-revolution-decomposition/) and [BQ standards fairness](https://lyhjeremy.github.io/boston-marathon-qualifying-fairness/). `src/ingest.py` chunks them (about 120 words, 30-word overlap), embeds with `all-MiniLM-L6-v2` and stores them in Chroma.

The tools, in `src/tools/`:

| Tool | What it does |
|---|---|
| `retrieve_cards(query, k)` | Dense search over the card chunks |
| `predict_finish(splits, gender, age)` | Finish time with 80% and 95% intervals, wall risk and pacing type, from [marathon-pacing-decay](https://github.com/lyhjeremy/marathon-pacing-decay) |
| `heat_effect(finish_min, temp_f, dewpoint_f, headwind_mph)` | Expected slowdown and range from race-day weather, from [marathon-heat-tax](https://github.com/lyhjeremy/marathon-heat-tax) |
| `pace_table(goal_time, strategy)` | Even splits, or a planned fade sized from the negative-split card (+5% under 3:00, +14% median, +22% over 4:00) |
| `calculator(expression)` | Arithmetic, so sums in an answer come from a tool (agent only) |

The two models are vendored in `src/tools/vendor/`; `SOURCE` records the commit each came from. Each result carries a caveat from the model's own holdout check: the pace model has no weather input and its 80% interval at 35 km covered 74% of 2024 holdout runners; the heat model did not beat a flat no-weather guess on its 2015-2019 holdout (2.26% vs 2.15% error) and does not use dew point.

The agent loop (`src/agent.py`) is written by hand because the Claude CLI has no function-calling interface. Each turn the model returns one JSON object, a tool call or a final answer; the loop runs the tool and feeds the result back with a `[T1]`-style label. Invalid JSON gets one repair retry with the parse error quoted; after six tool calls the model must answer. `src/numcheck.py` then extracts every number in the answer and looks for it in the question, the tool results, the tool descriptions and the cited passages, allowing for rounding and for clock times written as minutes or seconds. When `ANTHROPIC_API_KEY` is set, `ApiToolCoach` runs the same tools through the Messages API's native tool use (not used in the evaluation, and untested here for lack of a key).

LLM calls through the CLI run isolated: built-in tools off, no settings or CLAUDE.md loaded (`--setting-sources ""`), the caller's system prompt, a temporary working directory. Without that the "bare" model could have read the cards from disk.

## MCP server

`src/mcp_server.py` serves `retrieve_cards`, `predict_finish`, `heat_effect` and `pace_table` over stdio with the official `mcp` Python SDK (version 2). Point the client at a Python that has this repo's requirements installed.

Claude Code:

```bash
claude mcp add marathon-coach -- ~/venvs/marathon-strategy-rag/bin/python /path/to/marathon-strategy-rag/src/mcp_server.py
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "marathon-coach": {
      "command": "/Users/you/venvs/marathon-strategy-rag/bin/python",
      "args": ["/path/to/marathon-strategy-rag/src/mcp_server.py"]
    }
  }
}
```

`python -m scripts.mcp_smoke` starts the server, lists the tools and calls two of them (`eval/mcp_smoke.json`). `eval/mcp_roundtrip.json` records a real round trip in which Claude Code, given the server through `--mcp-config`, called `pace_table` for a 3:40 goal and quoted 8:23 per mile and a 1:50:00 half.

## How to run

```bash
python3 -m venv ~/venvs/marathon-strategy-rag
~/venvs/marathon-strategy-rag/bin/pip install -r requirements.txt

python -m src.ingest                      # index the cards (to ~/data-cache/marathon-strategy-rag)
python -m src.agent "I went through halfway in 1:42. What finish should I expect?"
python -m src.cli ask "should I negative-split?"   # the July RAG coach

python -m pytest -q tests                 # fast checks, no model calls
python -m src.retrieval_eval              # retrieval modes compared
python -m src.gold                        # expected values for the gold set
python -m src.eval                        # three-system eval (resumable, about an hour)
python -m scripts.export_web n01 c01 s15  # data for the write-up page
python -m scripts.vendor_models           # re-copy the models once their repos mark them ready
```

`CLAUDE_MODEL` sets the model (default: the `sonnet` alias). The index location can be changed with `MSR_DATA_DIR`.

## Repo layout

| Path | What it is |
|---|---|
| `knowledge/*.md` | The knowledge base, one citable card per topic |
| `src/agent.py` | Tool-using coach: JSON action loop, repair retry, step cap, number check |
| `src/tools/` | Tool registry, pace table, card facts, model loader, vendored models |
| `src/mcp_server.py` | The four domain tools over MCP (stdio) |
| `src/numcheck.py` | Unsourced-number post-check |
| `src/gold.py`, `eval/gold.jsonl` | Frozen gold set and its expected values |
| `src/eval.py` | Three-system evaluation |
| `src/hybrid.py`, `src/retrieval_eval.py` | BM25, RRF, cross-encoder reranking, and their comparison |
| `src/coach.py`, `src/chain_lc.py` | The July RAG chain, by hand and in LangChain LCEL |
| `src/eval_v2.py` | The July seven-question evaluation, kept for the record |
| `src/llm.py` | Claude CLI (default) or Anthropic API |
| `eval/` | Run records, summaries, retrieval comparison, MCP records, the July results |

## Limitations

- The numeric score is only as good as the models. The heat model did not beat a flat no-weather guess on its 2015-2019 holdout, and the pace model's late-race intervals ran narrow on 2024. Matching them perfectly is consistency, not correctness. The coach passes both caveats on.
- The judge is the same model as the three systems. Bare faithfulness is judged against the passages RAG retrieved, so a correct claim the cards lack counts against it. It fell from 0.76 (July) to 0.33 because the judge prompt changed (one answer at a time, numbers count as claims); the two are not comparable.
- I wrote all 43 questions, and the numeric ones supply every input. Each system answered once, so run-to-run variation is not measured.
- The number check is rule-based. Its 2.5% rounding allowance let one head-computed range through ("205 to 219 minutes" in c01), and it flags correct paraphrases. Its rules changed twice after the run began (tool descriptions and field-name digits such as the 95 in `lo95_min` count as sources; bold list numbers are skipped); every answer was rescored with the final rules. Under the first version it fired on 11 coach and 29 RAG answers.
- The retrieval comparison uses 43 questions and relevance labels I wrote.
- The course card's figures are simulated, and the card says so next to each one.

## Earlier results (July 2026)

The July version was scored on seven strategy questions (now s01 to s07): faithfulness 0.98 grounded against 0.76 bare, answer relevance 0.95, context precision 0.78, citation validity 1.00 (`eval/results.csv`, `eval/faithfulness.png`). 2026-10-28: superseded by the three-system run above, which uses a different judge prompt (one answer at a time, against that system's own context) and a pinned model, so the two are not directly comparable. The simulated-data label on the course card was added after the July run; the October run used the labeled card.

## License

[MIT](LICENSE) © 2026 Jeremy Lee. Educational tool, not coaching or medical advice.
