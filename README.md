# crossmarket-agent

**English** | [Русский](README.ru.md)

A multi-tool agent for comparing prices across marketplaces (Wildberries ↔ Ozon):
finding a product by description, matching listings between the two marketplaces,
price analytics — plus an eval harness and observability around all of it.

## What already works

- **Corpus snapshot**, collected semi-manually with a browser scraper: prices and
  attributes in ClickHouse, texts in Qdrant, the
  `intfloat/multilingual-e5-large` embedder running locally.
- **Tool A: Retrieval-QA** — an agent on the Claude Agent SDK with a single tool over
  Qdrant + ClickHouse. It answers from the snapshot or says plainly that the product
  is not in the database.
- **Tool C: text2sql** — the agent writes real SQL against the snapshot in ClickHouse
  and repairs it from the error text. Knowledge of the schema, the dialect and the
  data pitfalls lives in a Claude Agent SDK skill rather than being pasted into the
  prompt.
- **Tool B: Wildberries ↔ Ozon matching** — a Wildberries listing → candidates from the
  Ozon collection → a text confirmation model for each one. If none is confirmed, the
  result is empty: a similar product is never passed off as the same one.
- **A + B + C routing** — one configuration with all the tools: the orchestrator
  decides by itself whether to pull the answer from the listing text, look for the
  product on the other marketplace, or compute it over the database.
- **Eval harness**: `make eval` — recall@k and MRR on the golden set, an end-to-end
  agent run with metrics on the final answer, text2sql accuracy on result rows,
  routing accuracy; reports in `evals/reports/`, metrics in MLflow, traces in
  Langfuse. The golden set was split into calibration and held-out parts before the
  judges existed; reported numbers come from the held-out part only.
- **Judges** — three LLM judges: the agent's final answer, the matcher's verdict, the
  agent's SQL. The prompt is tuned on the calibration part, the reported agreement
  with ground truth comes from the held-out part, and it is always shown next to the
  baseline of a judge that gives the same answer to everything and a free
  deterministic check.

## Stack

Claude Agent SDK (orchestrator), Qdrant, ClickHouse, MLflow, Langfuse,
sentence-transformers, FastAPI (later), Grafana + Prometheus (later).

## Stages

- [x] Foundation: data collection, Qdrant and ClickHouse, bare retrieval
- [x] Tool A: Retrieval-QA and an agent on the Claude Agent SDK
- [x] Eval harness and judge scaffolding
- [x] Tool C: text2sql over ClickHouse
- [x] A + C routing
- [x] Tool B: Wildberries ↔ Ozon matching
- [x] Judges and calibration on held-out test sets
- [ ] Observability and cost
- [ ] README and showcase artifacts

## Numbers

Held-out test split, 24 questions out of 61, corpus of 884 listings:

| | opus-5 | haiku-4.5 |
|---|---|---|
| correct outcome | 1.000 | 0.833 |
| relevant listing named in the answer | 1.000 | 0.636 |
| prices and products in the answer backed by the search results | 1.000 | 1.000 |
| cost per question | $0.060 | $0.020 |

Bare retrieval without the agent, same split: recall@1 0.682, recall@5 0.818, MRR 0.783.

The set has 33 negative questions — ones whose answer is not in the corpus. Opus
refused correctly on all of them, including near misses where a similar product
sits right next to it in the corpus.

The second model shows what the cheap one pays with: all of haiku's failures are
"didn't go searching", not a single "searched and picked the wrong thing". Of the 28
questions with an answer in the corpus, it called search on 21 and named every
relevant listing in all 21; it didn't search on the other seven. Neither model
invented prices or products.

### Wildberries ↔ Ozon matching

Held-out test split, 61 labeled pairs, mean of three runs:

| | value |
|---|---|
| F1 | 0.747 |
| precision / recall | 0.862 / 0.660 |
| "no counterpart": share of products where no foreign candidate was confirmed | 1.000 |
| the true counterpart reached the confirmation model | 1.000 |

This is the project's weak spot — and the way the measurement is built shows exactly
where: retrieval brings in candidates without losses, a third of the pairs are lost at
the confirmation step, because the listing often lacks the attribute that would let
the pair be identified. Analysis — [docs/findings.md](docs/findings.md) (in Russian).

text2sql accuracy on the reference cases is 0.947, routing accuracy across the three
tools is 1.000; both numbers come from held-out splits.

### Judges

Judge agreement with ground truth on held-out splits. The positive class is "bad":
precision is the share of real errors among those the judge flagged, recall is the
share caught.

| judge | what it judges | ground truth | test | agreement | baseline | precision | recall |
|---|---|---|---|---|---|---|---|
| QA | the agent's final answer | manual labels | 17 | 0.941 | 0.765 | 0.800 | 1.000 |
| B | the confirmation model's verdict | pair label | 61 | 0.770 | 0.656 | 0.769 | 0.476 |
| C | the agent's SQL and its answer | reference query | 76 | 0.882 | 0.908 | 0.000 | 0.000 |

`baseline` is the agreement of a judge that gives the same answer to everything.
Without this column all three numbers would read as a success; with it, you can see
that judge C loses to the degenerate one.

Breakdown of disagreements:

| judge | where it errs | why |
|---|---|---|
| B | catches none of the 5 false confirmations | a pair with one hidden difference: the judge sees the same listings as the tool and fails to find it in the same place |
| B | misses 6 of 16 false rejections | the scrape lacks the attribute the labeler used to identify the pair; opus instead of sonnet doesn't help |
| C | catches none of the 7 misses | a text2sql miss is a plausible query with the wrong result; without execution it is indistinguishable from a correct one |
| C | 9 false alarms | partly different definitions: the judge grades the answer text, the metric grades the query result |
| QA | 1 false alarm on the test set | nitpicking how the agent read the listing's attributes |

Bottom line: direct metrics come first. Where ground truth exists, the judge is
checked against it — C fails the check, B works on half of the errors, and the answer
judge ties with the keyword check (0.925 each on all 40 labeled). Details —
[docs/findings.md](docs/findings.md) (in Russian), reports — `evals/reports/judge_*.md`.

Short conclusions for every measurement — [docs/findings.md](docs/findings.md), the
failure log — [docs/failures.md](docs/failures.md), run reports —
[evals/reports/](evals/reports/). All of them are in Russian.

**It can't be reproduced from a clone: there is no data in the repository** — neither
the corpus nor the background of 700 distractors. This is deliberate: only the golden
set and the run reports go into git.


MIT.
