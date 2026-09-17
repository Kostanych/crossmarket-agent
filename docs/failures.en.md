# Failure log

**English** | [Русский](failures.md)

Cases where the system silently did the wrong thing. The bar is high: everyday trouble with
collecting data does not belong here. What belongs here is a result that looked right and was not,
or money spent for nothing.

## 1. The Langfuse instrumentation silently did not attach (26.08.2026, stage 2)

A run of 28 questions completed, the metrics landed in MLflow and the script printed "tracing on".
Not a single trace appeared.

The instrumentor patches the `query` attribute of the `claude_agent_sdk.query` module, while the
agent did `from claude_agent_sdk import query` at import time: the reference was frozen before the
patch, so every call went around the wrapper. Nothing reports this — not the instrumentor, which
cannot tell it was bypassed, and not Langfuse, which simply received no data. The check at the time
was "a trace exists", and the trace that existed came from a manual wrapper around the run and held
one empty span.

Cost: a $1.91 run without traces.

Fixed with late binding — the module resolved through `importlib.import_module` and the attribute
read at call time — plus `flush_langfuse()`, because the export is batched and a short-lived process
was exiting before the data went out.

**Conclusion.** For observability, "done" is not "a trace appeared" but "the trace holds what it was
set up for": tool spans, tokens, cost — and that should be checked in code. The same applies to any
tool that works by patching someone else's functions: it stays silent when it is bypassed.

## 2. The refusal checker understated quality, and the number reached the report (28.08.2026, stage 3)

The metric "correct outcome on a negative case" showed 0.867 on far misses and 0.967 over the whole
set. The number went into the report, into the measurement journal and into CLAUDE.md as a property
of the model. In fact the agent had refused correctly on all 33 negative cases — the instrument was
wrong.

A refusal was recognized from a list of phrasings written from imagination rather than collected
from the model's actual answers. Three answers did not match it: "there are no building materials in
the snapshot at all" (the negation sits far from the word "database"), "not found in the Wildberries
database" (that word form was missing from the list), and "equipment (bags, punchbags, pads) not
found" (the enumeration ran past the 80-character window).

A broken instrument is invisible: an understated metric reads as an error of the model, which is
exactly what the measurement was expected to find. The only thing that gave it away was printing the
answers themselves next to the number.

Cost: three re-measurements instead of one, about $5.8.

The check now looks for a source and a negation within one sentence, at any distance from each
other; it was verified on 99 answers from two models and on controls that must not count as a
refusal — an ordinary answer, a substituted product, a refusal by domain, a clarifying question. Raw
runs are now saved into `evals/runs/`, and `python -m evals regrade` recomputes the metrics from them
for free: before that, every change to the parsing meant a new paid run, on different answers than
the ones the problem had been spotted on.

**Conclusion.** An instrument that errs in the same direction as the expected result cannot be seen
in the numbers, only in the raw answers printed beside them. The checker is not part of the system
under measurement, so it has to be fixed even when the breakage surfaces on the test split; but it
is only ever fixed in the direction of a higher score, and the number after the fix comes from a new
run the fix did not touch.

## 3. The diagnosis of the disagreements was written twice, and both times wrong (02–04.09.2026, spike B)

The B spike produced 25 disagreements between the confirmation model and the labels.

**First pass.** I read the "model's reason" column, grouped the phrasings and concluded that the
model applies the rules to individual fields while the labeler applied them to the product, and that
a color mismatch ("aloe" versus "3") was noise in the attributes. That went to the user as the result
of the spike. He asked for examples: "I don't see what you mean about color, is that listing marked
as a match?".

**Second pass.** I opened the listings themselves and wrote the opposite: about 17 of the 25
disagreements are **labeling errors**; the lead example was a garlic press at 214 ₽ marked as a
match to a vegetable cutter at 7539 ₽. That already went into `evals/results.md` and CLAUDE.md as an
established fact, and the conclusion for stage 6 was built on it: "clean the ground truth first,
then measure".

**What was actually true.** On 04.09.2026 the user went through all 25 pairs by hand. Not one
labeling error was among them, and `data/labels.jsonl` did not change. The label had been set by a
person who saw the whole product, while some of the attributes never made it into the scraped
fields: the model refused correctly on what it could see, and the label was correct about the
product. That garlic press matches on color, material, country and a weight of 80 g — only the price
and the seller's wording of the title differ, and price between marketplaces is about markup, not
about the product. The fix is to add the missing attribute to the snapshot, which the user did on 24
listings.

It also came out that the verdict parser matched `match` inside the phrase "this is not a match but
a no_match" and recorded the opposite answer for the model — part of the "false confirmations" was a
parser bug rather than the model's behavior.

Cost: $1.11 on a run that processed 74 pairs instead of 30, because a negative slice in the sampler
(`size - len(negatives)` went below zero, and a slice from the end took nearly the whole corpus);
plus the two days the wrong conclusion stood in the measurement journal and in CLAUDE.md, long
enough to shape a decision about stage 6.

**Conclusion.** A model disagreeing with the ground truth is a list of cases to review, not a
diagnosis. Until the owner of the ground truth has looked at them, the only thing that may be
written is "the model disagreed with the label on N pairs, here they are"; "the labels are noisy" is
already a conclusion, and it was wrong. The order of suspects is not "labels first", as this file
used to say, but: completeness of the fields, then the model, then the labels — and it is a person
who decides. Separately: a disagreement read off a listing whose fields are empty is evidence about
the fields, not about the label. And the model's explanation is its account of what happened, not a
record of it: the diagnosis is built from the source records, and full verdict texts are written to
disk right away, because a report column truncated to 52 characters is useless for analysis.

## 4. The verdict before the reason — the model answered before it thought (07.09.2026, stage 6)

The confirmation model's prompt asked for the answer in the order "first line — match or no_match,
second line — the reason". On obviously unrelated listings the model put the token down without
thinking and then contradicted itself in the next line:

```
match
Completely different products: a cloche hat versus a plush doll
```

The parser read the first line and recorded `match`. What reached the report was that B confirms an
unrelated candidate in 21% of cases — that is, breaks its own rule of never returning the closest
bad match.

The defect is not spread at random: not one such error on real pairs (0 of 35), and all eleven out of
eleven on candidates that were not counterparts. Where the answer is not obvious the model reasons
and the first line comes out right; where it is obvious it answers without thinking. The very outcome
the tool was built for was the one that suffered.

**Worse than the number was the fix the number suggested.** A candidate threshold sweep on the same
run showed that a cut-off at 0.85 lifts "no counterpart" from 0.787 to 0.979 and costs nothing in
found@1. It looked like a parameter worth adopting — while in fact the cut-off removed the most
dissimilar candidates, exactly the ones the model was failing on. Had we taken the threshold, the
defect would have been hidden for good, and the first product whose true counterpart scored 0.84
would have been lost silently.

Cost: $6 on re-measurements, and an unnecessary parameter nearly adopted.

The fix was to swap the fields: the reason first, the verdict on the last line. After it "no
counterpart" became 1.000 (0 false confirmations on 188 candidates), and the sweep showed the
cut-off buys nothing at any level.

**Conclusion.** The format "answer, then reason" lets the model commit to a verdict before it
reasons, and on easy examples it does exactly that. The reason has to come first. The second
conclusion is more general: a setting that repairs a metric is a reason to look for a bug, not a
setting worth keeping. The threshold "worked" here precisely because it correlated with the bug.

## 5. A documentation search launched a paid suite (15.09.2026, stage 8)

While checking an SDK upgrade, the assistant searched `CLAUDE.md` with `grep`, passing the pattern in
double quotes. The pattern picked up a phrase from the document together with its markup:

```
grep -n "...\|`python -m evals qa` жжёт деньги\|..." CLAUDE.md
```

Inside double quotes bash reads backticks as command substitution. Instead of a search, the QA suite
started on 61 questions, and its output went into the `grep` pattern — nothing appeared on screen.

From the outside it looked like a hung search. The command went to the background on the two-minute
timeout, and only a `ps` check revealed `python -m evals qa`; the process was stopped about 140
seconds after it started. A routing smoke run was going on in parallel, so two suites shared the
limits and interleaved their traces in Langfuse.

Cost: about a dozen questions of the suite under the subscription, a few tens of cents by the SDK's
estimate. **The reported artifacts survived by luck:** the suite writes its report and dump at the
end of a run. Had the process finished, it would have silently overwritten `evals/reports/qa_wb.md`
and the dump of the full run and added a run to MLflow — all of it measured on a new CLI with a
different token cost, and all of it passing for the canonical number. Had an API key been enabled in
`.env`, it would have been money off the balance.

**Conclusion.** The project's documentation deliberately writes commands in backticks, and the most
frequent of them cost money — so text taken from it goes into the shell only in single quotes or
through `grep -F -e`. A search command that hangs for more than a couple of seconds is a reason to
check `ps` right away rather than wait for the timeout.
