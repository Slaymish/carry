# The first number, and why it is not a result

A model was put in the loop for the first time on 4 November 2026: qwen3.8-27b wrote
three arms against a synthetic fixture, and two judges scored them blind. This page
records what came back, including the part where the sabotage gate did not hold.

## What ran

- **Fixture.** `code/data/fixture-tool-library.jsonl`, written by
  `code/data/make_fixture.py`: 48 synthetic episodes, 192 turns, four turns each, one
  episode a day for 48 days. A fictional community tool library with a consistent
  vocabulary (the cage, the noticeboard, the Friday count, the cutoff) and a stated
  manner (number first, two sentences). Every line was written for this repository. No
  transcript, no excerpt, no statistic about a real person.
- **Split.** 30 train / 10 recent / 8 anchor episodes. Anchor is the oldest eight, taken
  out of training rather than copied from it. `check_no_leakage` passes.
- **Candidates.** qwen3.8-27b, temperature 0, `enable_thinking=false`, `max_tokens=400`,
  8 ledger lines. 18 held-out items (10 recent, 8 anchor) times three arms: 54
  completions in about four minutes. Committed at `code/data/first-run/candidates.jsonl`
  so the scores can be recomputed without the model.
- **Judges.** `lexical-f1` (a floor) and `gemma-4-31b` (a different family from the
  candidate). The model judge sees the conversation, the reply that actually happened,
  and two candidates labelled A and B, whose order is drawn per item from
  `random.Random(seed)`. No arm name appears in its prompt; a test asserts that.

Reproduce:

```bash
cd code
python3 data/make_fixture.py > data/fixture-tool-library.jsonl
CARRY_MODEL=qwen3.8-27b python3 -m carry.generate --stream data/fixture-tool-library.jsonl \
  --anchor-episodes 8 --max-lines 8 --disable-thinking --max-tokens 400 \
  --out data/first-run/candidates.jsonl --arms-dir data/first-run/arms
python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
  --arms data/first-run/arms/ledger-vs-frozen.jsonl
python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
  --arms data/first-run/arms/ledger-vs-frozen.jsonl --judge model --judge-model gemma-4-31b
```

`enable_thinking=false` is a trade, not a free win: with the reasoning step on, one
completion took about 28 seconds and the run became a two-hour job. The flag is set for
all three arms at once, so no arm gets the slower path.

## The numbers

Recent held-out set, 10 items, adapted minus frozen. Lexical judge:

```
ledger vs frozen       +0.085  95% CI [+0.001, +0.174]  excludes zero
sabotaged vs frozen    +0.013  95% CI [-0.056, +0.085]  includes zero
ledger vs sabotaged    +0.072  95% CI [+0.005, +0.140]  excludes zero
```

Model judge (gemma-4-31b), same items:

```
ledger vs frozen       -0.100  95% CI [-0.600, +0.400]  includes zero
sabotaged vs frozen    +0.200  95% CI [-0.300, +0.700]  includes zero
ledger vs sabotaged    +0.100  95% CI [-0.400, +0.600]  includes zero
```

Anchor set, 8 items, same comparison. Lexical judge: ledger `+0.042`
`[-0.040, +0.122]`, sabotaged `+0.110` `[+0.002, +0.233]`. Model judge: ledger `+0.625`
`[+0.125, +1.000]`, sabotaged `+0.625` `[+0.125, +1.000]`.

**Correction, 4 November (Block C):** the two rows naming the sabotaged arm above were
produced by a block that was not the ledger's own sentences, and are withdrawn. The corrected
run is in `05-robustness.md`.

The two judges do not agree on the sign of the only interesting comparison. The lexical
judge says the ledger arm gains 0.085 and beats the sabotaged arm; the model judge says
it loses 0.100 to frozen and that the sabotaged arm gains 0.200, which is more than the
real one.

## The sabotage gate fails, and the metric is void at this size

Under the model judge the sabotaged arm scores **above** the real arm on recent items
(+0.200 against -0.100). The rule in `docs/00-question.md` is that a metric which fails
to punish a sabotaged update is not measuring adaptation. By that rule this metric, on
this fixture, with this judge, is void. It is not a weak positive result. It is not a
number.

**Added 4 November, Block C.** The verdict stands and the reason is now measured directly:
with both prompt arms rebuilt from the *same* sentences, the paired ledger-vs-sabotaged
comparison under the model judge is `-0.400` `[-0.700, -0.100]`, the only recent-item interval
on this page that excludes zero. The real block loses to its own scramble.

Two things keep that from being the whole story, and neither rescues it.

- **The scrambled arm was not the same sentences, and with that fixed the gate fails
  cleanly.** *Corrected 4 November, Block C.* `build_sabotaged_ledger` shuffled the whole
  20-line pool and took the first eight, so 4 of its 8 lines differed from the ledger's, and
  the control varied content and order at once. Fixed, the sabotaged arm scores `+0.072`
  against the ledger's `+0.075`, and the paired comparison between them is `+0.003`
  `[-0.031, +0.036]`. A block of the user's own sentences in the wrong order is worth exactly
  as much as the right one, so this measures the presence of the block and not memory of a
  moment. Every sabotaged-arm number in this page is from the broken block and is
  withdrawn.
- **The model judge has no power at n=10.** A pairwise judge over 10 items can only
  return deltas in steps of 0.1, and the observed interval half-width is 0.5. Half-width
  scales as 1/sqrt(n), so resolving a 0.1 effect needs roughly 250 recent items, and a
  0.05 effect about 1000. The fixture has 10. Every model-judge number above is
  consistent with no effect, and the lexical judge's +0.085 whose lower bound is 0.001
  is one resample away from vanishing.

**Corrected 4 November, Block C. The model judge is not stable, and this paragraph said
it was.** Seeds 0, 1 and 2 happened to agree; seeds 3 to 9 do not. Over ten seeds the recent
delta is -0.10 on four, 0.00 on four and +0.10 on two (mean -0.020, sd 0.079). The harness is
deterministic (the same seed twice gives a byte-identical report), so the movement is the
judge: eight of the ten items return the same verdict whichever slot the candidate occupies
and two do not, and every seed's total is the sum of those two. There is no consistent "no"
here to quote.

## Controls

Verbosity and agreement rate on recent items:

```
                    frozen      ledger      sabotaged
mean words          19.4        15.3        24.3
agreement rate      0.400       0.000       0.100
```

**Corrected 4 November, Block C. The length argument was backwards.** For a
precision-weighted F1 a shorter reply is the advantaged one, and on these ten items the gain
rises with words saved: +0.017 per word, r = +0.72, with the fitted gain at equal length
+0.018 rather than +0.085. The subset of seven items where the ledger reply is not longer
still contains replies eight words shorter, so it cannot remove a graded effect. On the
regenerated run the equal-length gain is -0.004.

**The agreement control cannot remove anything**, because the ledger arm never agrees: the
subset is all ten items. What the numbers do support is narrower than "agreement is not what
it wins with": the model judge's -0.100 is not spread evenly, and sits almost entirely on the
four items where the frozen reply does open with agreement (-0.75 there, +0.33 on the other
six). That is consistent with a judge that prefers the agreeing arm, and it cannot be
separated from the ledger arm simply losing those items at four items. See
`05-robustness.md`.

## What this is not

**The ledger arm is prompt-level adaptation, and prompt-level adaptation is a baseline
and not the parametric claim this project is about.** It is a frozen model handed a
short block of the user's own earlier sentences in its context. If a ledger prompt
matched a trained adapter, the project would have its answer and the answer would be
that the weights are unnecessary. Nothing here trains anything, so the anchor comparison
above is not retention: nothing can be forgotten when nothing was learned. The harness
prints that caveat on every run, and it is the reason both prompt arms score `+0.625` on
the anchor. A prompt arm carries the same vocabulary whether or not it is going to be
used, and the anchor set rewards exactly that.

## What would change this

- **The foreign-content sabotage.** A block of six true sentences about a different
  fictional user, same length and register, is the control with teeth. If the ledger arm
  still scores like it, the judge is measuring register and nothing else. This is next.
- **More recent items.** 250 of them for a 0.1 effect at this judge's variance. Any
  smaller number cannot be read as a result, whatever its interval says.
- **Item-level judge verdicts persisted**, so judge disagreement can be measured
  per item instead of inferred from two point estimates.
