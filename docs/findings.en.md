# What the measurements showed

**English** | [Русский](findings.md)

One or two paragraphs per finding. The numbers, the full conditions and the analysis live in
`evals/results.md`; the experiment numbers match.

## Retrieval and the corpus

**1. The description makes the vector work, the category does not.** Four text compositions were run
on one golden set. With the description recall@1 is 0.649, without it 0.506. The category added
nothing, and in the variant without a description it actually hurt. Filtering by category is more
precise through a database field than through the vector.

**2. Obvious distractors do not spoil the corpus.** 700 made-up listings grew the corpus 4.8×, and
recall@10 dropped by only 0.036. That is by design: they are far from every question, so they never
compete for a place in the output. It does not follow that the metric is broken — configurations are
still easy to compare, as long as recall@1 and MRR are used instead of recall@10.

**3–4. A score threshold works nowhere, and this came out three times.** The score distributions of
real answers and of questions with no answer in the database overlap — in tool A, in the same tool
on a set twice the size, and among tool B's candidates. A cut-off either removes nothing or starts
discarding correct answers. The decision is left to the model, which reads titles and descriptions
rather than a number. The constant stays in the config: on a larger corpus a separating point may
appear.

## The agent on top of retrieval

**5. The agent finds what plain search could not.** Both retrieval failures were fixed — "something
to shave with right in the shower" and "something to close an opened bottle of wine". The trace shows
why: what reaches the search is not the user's question but the model's own phrasing ("electric
shaver wet dry shaving"). That is precisely what an agent adds on top of retrieval.

**5. Two tool calls in one turn run in parallel.** Their spans start 0.05–0.9 s apart, so the second
phrasing is not a reaction to the first result but a second hypothesis raised up front. A parallel
pair costs a single turn, which is what keeps a limit of 5–6 turns workable.

**6. A metric stuck at 1.000 measures nothing.** On 61 questions the agent scored 1.000 on every
answer metric. That says the task is easy, not that the agent is good: such a metric will show
neither a regression after a model change nor a gain from a better prompt. Resolving power had to
come from somewhere else — from running a second model.

**7. The cheap model fails by declining the task, not by choosing wrong.** Haiku scored 0.833 against
1.000 on the same set, and all of its failures are of one kind: it never searched. There is not a
single case of searching and then picking the wrong listing, and the same pattern repeated in
routing. Neither model invented prices or products.

## Text2SQL and skills

**9. `lower()` in ClickHouse does not touch Cyrillic.** `lower('Китай')` returns `'Китай'`, and the
filter silently yields zero rows instead of 191. `lowerUTF8()` is needed. Found by running the
skill's examples against the live database — which is why every example in the skill is executed
rather than merely written down.

**9. Any query the agent ran should count, not just the last one.** Having got its number, the agent
often makes one more query for context. Grading the last query measured whether it had stopped
asking, not whether it had found the answer — three correct cases were lost that way on the first
run.

**9. The reference for a case is a query, not the result rows.** The repository is public, and the
rows are prices and titles. The reference SQL is executed at run time against the same snapshot the
agent sees.

**10. The skill cost twice as much as the prompt and added no accuracy.** A schema in the system
prompt sits in the static prefix and is cached once for all questions. The skill puts the same 2.5k
tokens inside each question's own dialogue, so they are written into the cache again every time:
$0.021 → $0.053 per question. Progressive disclosure pays off for large knowledge that is rarely
needed; here the knowledge is small and needed every time.

**11. Routing between two tools turned out to be easy.** 1.000 for opus, including the 24 borderline
questions written precisely to trip it up.

## WB ↔ Ozon matching

**8. The rule about price came from domain knowledge, not from the data.** Sellers raise the price
when a product runs out, so the listing sinks in the marketplace's search results until stock
returns. As long as price counted as an attribute of the product, the model rejected correct pairs
over a price gap. Once the prompt ruled price out, F1 went from 0.846 to 0.899.

**8. A model disagreeing with the labels is a list of cases to review, not a verdict on the labels.**
The first analysis declared the labels noisy; a human review refuted that completely — not a single
labeling error was found. The disagreements were all cases where the label is right but the
listing's fields do not carry the attribute it rests on. The fix there is to add the attribute to
the listing, not to change the label.

**12. The order of the fields in the prompt matters a great deal.** One version asked for the verdict
first and the reason on the next line. On obviously unrelated candidates the model wrote `match`
without thinking and then gave the opposite reason — "completely different products: a cloche hat
versus a plush doll". Swapping the fields, so that the reason comes first and the verdict last,
removed those errors entirely: zero on 188 unrelated candidates against eleven before.

**12. A setting that repairs a metric is a reason to look for a bug, not a setting worth keeping.**
With the broken prompt, a sweep showed that a candidate score cut-off at 0.85 lifted the "no
counterpart" outcome from 0.787 to 0.979 at no cost — it looked like a parameter worth adopting. What
it actually did was remove the most dissimilar candidates, which were exactly the ones the model was
failing on. Once the prompt was fixed, the cut-off gained nothing at any level.

**12. A number taken where the data was repaired measures the data, not the model.** F1 is 0.892 on
calibration and 0.747 on the held-out test, with the prompt unchanged down to the character. The
reason is in the data: of the 24 listings edited by hand, 20 belong to calibration pairs and one to a
test pair, because the attributes were filled in from a disagreement analysis that only covered
calibration. The 0.145 F1 gap is the measured cost of an incomplete scrape.

**12. Here the cheap model is both worse and more expensive.** Haiku loses on precision, recall,
accuracy and F1 while costing $2.16 against $1.64. The one metric it leads on — the share of correct
negatives — follows from its bias toward "no match", not from an ability to tell pairs apart. What it
cannot do is check the details of two listings against the rules in the prompt: it rejects correct
pairs over a packaging-weight difference or an empty brand field, the very things the prompt calls
out as gaps in the listing rather than differences between products.

**12. An empty answer from the model quietly becomes "no match".** One run hit the provider's limits:
89 of 92 answers came back empty, and the metric still looked meaningful — precision 1.000. The only
thing that gave it away was the "unparsed" column of the confusion matrix. The suite now prints a
warning of its own when more than a tenth of the verdicts fail to parse.

**13. Adding a third tool did not disturb the choice between the first two.** After matching was
wired in, routing was re-measured on 75 questions: 1.000 on the held-out test, with all four earlier
strata still at 1.000.

## Judges

**14. Judge agreement means nothing without a baseline next to it.** Judge B scored 0.837 on
calibration and 0.770 on test — the same model with the same prompt. The higher number is the useless
one: a judge that answers "the verdict is correct" to everything scores exactly as much, because the
tool errs on every sixth calibration pair. The lower number beats that blind judge by 0.11, because
the test pairs are harder for the tool. Hence the positive class "bad" for all three judges and the
baseline printed next to every agreement figure.

**15. Judge B catches the matcher's false rejections and misses its false confirmations.** When the
tool called the same product different over an empty field, the judge catches it in 10 cases out of
16. When the tool confirmed a pair that had one hidden difference, the judge missed it in all five —
it fails in exactly the same place the tool did. Useful as a filter over rejections, useless as a
check of confirmations.

**16. A stronger model as the judge did not help.** Opus scores 0.826 on B's calibration against
sonnet's 0.837, at twice the price. The judge sees the same listings as the confirmation model, and
no model can find an attribute that the scrape does not contain — the same ceiling the tool itself
hit at stage 6.

**17. A text2sql judge that does not execute the query is blind.** It caught none of the seven misses
on the test split, with agreement 0.882 against a baseline of 0.908. A text2sql miss is a plausible
query with the wrong result, and there is no way to tell it from a correct one without executing
both. Executing and comparing is exactly what the deterministic metric already does, for free. Such
a judge cannot be trusted on new questions that have no reference either: where a reference exists,
it caught nothing.

**18. The QA judge only matched the keyword check.** On all 40 labeled answers both score 0.925, and
on the test split the checker is even better (1.000 against 0.941). The judge's two wins both landed
on questions whose phrasings are written into its own prompt as examples, so they were won by
construction. The one class the checker cannot express — an absurd question that the agent is right
not to search for — never made it into the held-out split, so there was nothing there for the judge
to prove itself on.

**19. The judge found a defect in the documentation.** The first version of judge C rejected correct
answers because it checked their numbers against the examples in tool C's skill: the skill said 394
listings while the database held 427 after the third batch. The agent reads that same skill, so stale
examples in its knowledge drift away from the data silently. The numbers were recomputed, and they
have to be recomputed after every new batch of listings.

## Cost

**20. The cost the SDK reports is an estimate, and it diverged from the bill.** The Claude Agent SDK
computes `total_cost_usd` itself, from the price table bundled with its CLI — under a subscription
and under an API key alike. Reconciling it with Console on 20 routing questions: opus matched to the
cent and to the token, haiku matched too, while the estimate for sonnet-5 was half again too high,
because CLI 2.1.233 from the previous SDK version priced sonnet-5 as Sonnet 4.6. It also turned out
that the SDK does not run the `claude` from PATH but its own bundled copy, so the CLI version
recorded in the project's documentation had nothing to do with the agent. After the upgrade the new
CLI prices correctly, but it also writes more into the cache, so the run did not get cheaper.

**21. The authentication mode alone changes the cost by 30%.** The same 20 questions are estimated
30% more expensive under a subscription than under an API key, at an identical token count. Under a
subscription the CLI writes the prompt cache with a one-hour lifetime, under a key with a
five-minute one, and the longer write costs more. All of the project's measurements were taken under
a subscription, so their cost is an upper bound on what an API client would pay.

**22. A model call made from inside a tool never lands in the agent's reported cost.** The B
confirmation model runs as separate SDK calls from the tool handler, and the router's
`total_cost_usd` does not see them. In the reconciliation run they made up almost a third of the
bill; before stage 8 the `$/question` figure for routing on B questions was understated several times
over. Their cost and per-model tokens are now added to the outer answer, and Langfuse puts them in
the same trace as the router request.

## Where the effort should go

Tools A and C work well, B clearly underperforms — and that is an honest picture, not a defect of the
measurement:

- retrieval-QA (A): 1.000 correct outcomes on the held-out test;
- text2sql (C): 0.895 with the schema from the skill, 0.947 with the schema in the prompt — a
  difference of one question out of nineteen;
- routing across three tools: 1.000;
- **matching (B): F1 0.747**, of which recall is 0.660 — the model fails to confirm a third of the
  true pairs.

The cause has been established and it is not the model itself: the counterpart reaches the model in
100% of cases, the error happens at the confirmation step, and the misses fall into two groups — the
listing carries no attribute by which the pair could be identified, and the two marketplaces describe
the product differently. The ceiling is therefore set by how complete the scrape is. In a real
project that is where the effort would go: fill in attributes across the whole corpus by a single
rule, blind to the model's verdicts, and measure again. Here the work stops at that point
deliberately.
