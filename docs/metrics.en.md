# Results

**English** | [Русский](metrics.md)

The numbers from the README together with the conditions they were measured under. Every reported
number comes from a held-out test split. Run reports — [evals/reports/](../evals/reports/), short
conclusions from every measurement — [findings.en.md](findings.en.md), the failure log —
[failures.en.md](failures.en.md).

## Search and the agent's answer (A)

Held-out test split: 24 questions out of 61. The corpus is 199 real WB listings plus 700 distractors.

| | opus-5 | haiku-4.5 |
|---|---|---|
| correct outcome | 1.000 | 0.833 |
| the relevant listing named in the answer | 1.000 | 0.636 |
| prices and products in the answer backed by search output | 1.000 | 1.000 |
| cost per question | $0.055 | $0.020 |

Search without the agent, the same split: recall@1 0.682, recall@5 0.818, MRR 0.783.

The set holds 33 negative questions — ones the corpus has no answer to. Opus refused correctly on
all of them, near misses included, where a similar product sits right next to it in the corpus.

The second model shows where a cheaper one gives way: every haiku failure is of one kind — it never
searched. There is not a single case of searching and then picking the wrong listing. Of the 28
questions with an answer in the corpus it called search on 21 and named every relevant listing in
all 21; on the other seven it did not search at all. Neither model invented prices or products.

The text composition in the vector was chosen by measurement, on the corpus without distractors,
28 questions:

| composition | recall@1 | recall@5 | MRR |
|---|---|---|---|
| title + attributes | 0.506 | 0.839 | 0.706 |
| + category | 0.470 | 0.804 | 0.695 |
| + description | 0.649 | 0.929 | 0.857 |
| + category + description | 0.667 | 0.911 | 0.857 |

The description is worth about +0.15 recall@1 and the category nothing, so the category stays out of
the vector. One caveat: the variant with the description has a built-in advantage, because the
golden-set questions were written while looking at the listing descriptions.

## text2sql (C) and routing

**text2sql:** the agent's query result matched the reference one on 0.895 of 19 test questions,
with no query failing to execute. A match is counted after normalization: numbers are rounded, row order is
ignored for unordered questions, and extra columns are allowed. Requiring the exact shape gives
0.789. The schema is delivered by an SDK skill; with the schema inlined in the prompt the score was
0.947 — a difference of one question out of nineteen, within the run-to-run spread. The skill is
twice as expensive, though, $0.057 against $0.027 per question, because its text is written into the
cache again for every question.

**Routing:** on 27 single-hop test questions the right tool was called first in 1.000 of the cases.
Haiku on the same set fails by declining the task rather than by choosing the wrong route: it
answers "household advice is not my area" and calls nothing.

## WB ↔ Ozon matching (B)

Held-out test split, 61 labeled pairs, mean of three runs:

| | value |
|---|---|
| F1 | 0.747 (0.738–0.756) |
| precision / recall | 0.862 / 0.660 |
| "no counterpart": no foreign candidate confirmed | 1.000 (0 of 188) |
| the true counterpart reached the confirmation model | 1.000 |

Retrieval brings in the candidates without losses: recall@5 is 1.000. A third of the pairs is lost at
the confirmation step, because the listing often does not carry the attribute that would identify the
pair. There is no score threshold on candidates: the true counterpart scores 0.831–0.950, while the
top-1 candidate for products that have no counterpart scores 0.800–0.925 — one range sits inside the
other.

The calibration score is 0.892 against 0.747 on the test split. This is not an overfitted prompt: of
the 24 listings whose attributes were filled in by hand, 20 belong to calibration pairs. Test
listings were deliberately left alone — editing them would be tuning against the held-out set.

## Judges

The positive class is "bad": precision is the share of real errors among those the judge flagged,
recall is the share it caught.

| judge | what it judges | ground truth | test | agreement | baseline | precision | recall |
|---|---|---|---|---|---|---|---|
| QA | the agent's final answer | manual labels | 17 | 0.941 | 0.765 | 0.800 | 1.000 |
| B | the confirmation model's verdict | pair label | 61 | 0.770 | 0.656 | 0.769 | 0.476 |
| C | the agent's SQL and its answer | reference query | 76 | 0.882 | 0.908 | 0.000 | 0.000 |

`baseline` is what a judge that gives the same answer to every case would score. Without that column
all three numbers would read as a success; with it, judge C is visibly worse than answering blindly.

| judge | where it errs | why |
|---|---|---|
| B | catches none of the 5 false confirmations | a pair with one hidden difference: the judge sees the same listings as the tool and fails to find it in the same place |
| B | misses 6 of 16 false rejections | the scrape lacks the attribute the labeler used to identify the pair; opus instead of sonnet does not help |
| C | catches none of the 7 misses | a text2sql miss is a plausible query with the wrong result; without execution it is indistinguishable from a correct one |
| C | 9 false alarms | partly differing definitions: the judge grades the answer text, the metric grades the query result |
| QA | 1 false alarm on the test split | nitpicking how the agent read the listing's attributes |

Bottom line: direct metrics come first. Where ground truth exists, the judge is measured against it —
C fails that check, B catches about half of the errors, and the answer judge only matched the
keyword checker (0.925 each on all 40 labeled answers).

**At this data volume the judges did not pay off, and that is a result too.** Judge C failed its
check and is therefore not used at all: it stays out of the reported values and out of the production
path, and is kept as a documented negative result. Judge B works as a filter over the matcher's
rejections but runs into the same ceiling as the tool itself — the incompleteness of the scrape.
Judge QA did not beat a free deterministic checker. Judges pay off where ground truth does not exist
and is expensive to produce; in this project it exists almost everywhere, and the direct metric is
both more accurate and cheaper.

## Cost

Cost in the reports is the Claude Agent SDK's estimate from the price table bundled with its CLI,
not the bill. It was reconciled against the Console bill on 20 routing questions run through an API
key:

| model | role | SDK estimate | Console bill |
|---|---|---|---|
| opus-5 | orchestrator | $0.68 | $0.68 |
| sonnet-5 | B confirmation model | ~$0.34 | $0.23 |
| haiku-4.5 | CLI helper calls | ~$0.06 | $0.06 |
| | total | $1.08 | $0.97 |

The SDK counts tokens exactly: for opus the count matches Console down to the token. The gap on
sonnet comes from a stale price table — CLI 2.1.233, bundled with the previous SDK version, priced
sonnet-5 at $3/$15 instead of $2/$10. The SDK has been upgraded and CLI 2.1.259 uses the correct
prices; matching and judge costs measured before that are overstated by half on their sonnet share.

The authentication mode alone shifts the estimate: the same questions come out 30% more expensive
under a subscription at the same token count, because a subscription writes the prompt cache with a
one-hour lifetime and an API key with a five-minute one, and the longer write costs more. All of the
project's measurements were taken under a subscription, so their cost is an upper bound.

The B confirmation model runs as separate SDK calls from inside the tool, so they never appear in the
orchestrator request's own cost — and in this run they accounted for almost a third of the bill.
Their cost and tokens are added to the outer answer explicitly. Latency on the same 20 questions:
p50 10.9 s, p95 21.1 s.

## Observability

`make observe` brings up Prometheus and Grafana, `make serve` starts FastAPI with `/chat`, `/health`
and `/metrics`. The dashboard shows requests by outcome, p50/p95 latency, cost, tool calls and cache
share, next to the metrics Qdrant and ClickHouse expose themselves. Four alerts: a database is down,
hourly spend, the share of errors and limit cut-offs, p95 latency. Calls to the confirmation model
land in the same Langfuse trace as the orchestrator request.
