# Attacking the first number

`04-first-number.md` reported that the model judge scored the ledger arm below the frozen
arm on recent items, and that the sabotage gate failed. This page is what happened when
that number was attacked on its own terms: do the judges agree, does the number survive a
different seed, does it survive being regenerated, and did the ledger arm win anything by
being shorter or more agreeable.

Four of the doc's claims did not survive. They are struck in place in `04-first-number.md`
and listed at the end here.

## The judges do not agree, item by item

Same arms, same ten recent items, lexical judge against `gemma-4-31b`. Ten items is the
whole sample, so the test is a permutation (20,000 shuffles, seed 0).

```
items            10
pearson          +0.567
spearman         +0.571
same sign        4/10
permutation p    0.087
```

The correlation is positive and the sign agrees on four items out of ten, so the two judges
are not measuring the same thing. Reading the four largest disagreements:

- **Item 9.** Reference: *"The Friday count is Friday. Today it is the overdue list and the
  ladder."* Frozen answers with the count and the overdue list. The ledger arm answers *"The
  Friday count is 42, which is two under the expected total."* The lexical judge gives the
  ledger reply +0.030 because it repeats "the Friday count"; the model judge gives the frozen
  reply the win, which is right, because the ledger reply invents a number.
- **Item 2.** Reference says the sander is out without a loan row. The ledger arm answers
  *"Bay 6 being empty confirms the sander is the missing item."* The model judge gives it the
  win; the lexical judge calls it a tie at -0.017.
- **Item 6.** Reference: *"Then show the count under the grid."* The ledger arm answers *"The
  Friday count is 42, which is the only number I trust."* Lexical +0.326, model tie.

On every disagreement I read, the lexical judge is rewarding shared vocabulary and the model
judge is reading content. That is the lexical judge doing what it says on the tin, and it is
why a lexical +0.085 with a lower bound of +0.001 is not evidence that the ledger arm was
closer to what actually happened.

## The number does not survive the seed

`04` said the model judge was stable across seeds 0, 1 and 2. It was, on those three. Over
ten seeds the number moves across the whole of the effect it was supposed to report.

```
seed  recent    w/t/l  anchor  shuffled-ref
   0   -0.10  3/3/4  +0.625         -0.70
   1   -0.10  3/3/4  +0.750         -1.00
   2   -0.10  3/3/4  +0.625         -0.80
   3   +0.10  4/3/3  +0.625         -0.70
   4   +0.00  3/4/3  +0.625         -1.00
   5   +0.00  4/2/4  +0.625         -0.70
   6   +0.00  4/2/4  +0.625         -0.80
   7   +0.10  4/3/3  +0.625         -0.90
   8   +0.00  3/4/3  +0.625         -0.90
   9   -0.10  3/3/4  +0.625         -1.00

mean -0.020  min -0.10  max +0.10  sd 0.079  (4 negative, 4 zero, 2 positive)
```

The harness is not the cause. Re-running seed 0 twice gives byte-identical reports
(`diff data/robustness/model-ledger-vs-frozen-s0-a.json ...-s0-b.json`, no output), and
re-running the lexical judge twice is identical. A temperature-0 judge on this endpoint is
deterministic.

The cause is the A/B order, and it is exact. Eight of the ten items return the same verdict
whichever slot the candidate occupies. Two do not:

```
item       adapted-first     base-first
   6                0.0            1.0
   9               -1.0            0.0
```

The eight invariant items sum to zero, so every seed's number is the sum of those two
deltas, which can only be -0.1, 0.0 or +0.1. The seed table is that arithmetic. **There is no
reading of "the judge said no" available here**: the judge cannot decide on two of ten items,
and the total is those two items.

What the seed sweep does not touch, and this is worth keeping: the anchor gain is `+0.625`
on all ten seeds, and the shuffled-reference control stays between `-0.70` and `-1.00`. Those
are order-robust. The recent comparison is not.

## The sabotage block was not the same sentences

`generate.build_sabotaged_ledger` shuffled the whole 20-sentence pool and took the first
eight. `build_ledger` takes the newest eight. With a pool of 20 those are different sets:
**4 of the 8 lines differed**, so the sabotage arm was a different memory rather than the
same memory in the wrong order, and the comparison varied content and order at once.

Fixed in `carry/generate.py`: the selection is taken first, then shuffled, and a test asserts
the two blocks contain the same set of lines. The first-run sabotage numbers in `04` were
produced by the broken block and are withdrawn.

## The length claim in `04` is backwards

`04` argued that because the ledger arm is shorter than frozen (15.3 words against 19.4), its
lexical gain is not a length effect. For a precision-weighted F1 that is the wrong way
round: a shorter reply that hits the same words scores higher, so shorter is the way a length
effect would show up.

Measured on the ten recent items, with the gain fitted against words saved:

```
slope          +0.017 per word saved
at equal length +0.018   (mean 4.1 words saved)
raw gain        +0.085
```

The ledger arm's lexical gain rises with how much it shortens the reply, and about four
fifths of the raw gain is that slope. The subset control in the harness keeps the seven items
where the ledger reply is *not longer*, which still includes replies eight words shorter, so
it cannot remove a graded effect. What survives the fit is `+0.018` at equal length, and a
subset interval containing zero.

## Agreement: the control cannot bite, and the confound runs the other way

The ledger arm never opens a reply with "yes", "sure" or "ok" on this fixture (0.000 against
the frozen arm's 0.400), so the subset of items where the adapted reply is *not more
agreeable* is all ten items: the agreement control as written cannot remove anything.

The model judge's `-0.100` is not spread evenly. On the four items where the frozen reply
opens with an agreement word it is `-0.75`; on the other six it is `+0.33`. Whether that is a
judge that likes agreement or simply six items where the ledger arm was better cannot be
settled at four items, and the reading does not settle it either: on item 5 the frozen arm
says "yes, call them" and the reference agrees with it, while the ledger arm answers "no, we
never email or call members", applying a remembered line out of scope. The model judge was
right to prefer the frozen reply there.

## The regenerated run: the real block loses to its own scramble

Run 2 regenerated all three arms on the same fixture at the same settings, with the
sabotage block now holding the ledger's own eight sentences in a different order.

```
                          gain    95% CI          wins
lexical judge, ten recent items
ledger vs frozen        +0.075  [-0.008, +0.165]   6/0/4
sabotaged vs frozen     +0.072  [+0.002, +0.142]   7/1/2
ledger vs sabotaged     +0.003  [-0.031, +0.036]   3/2/5
manner lines only       +0.004  [-0.057, +0.061]   6/1/3
at equal length         -0.004  (ledger, from the length fit)
retention                +0.042  [-0.040, +0.122]

model judge (gemma-4-31b, seed 0), same items
ledger vs frozen        -0.100  [-0.600, +0.400]   3/3/4
sabotaged vs frozen     +0.200  [-0.300, +0.600]   4/4/2
ledger vs sabotaged     -0.400  [-0.700, -0.100]   0/6/4
```

**The real block loses to its own scramble.** Under the lexical judge the two prompts are
indistinguishable (+0.075 against +0.072, and +0.003 between them). Under the model judge the
scrambled block wins the paired comparison by 0.4, with the only interval on this page that
excludes zero: the ledger arm wins none of the ten items, ties six and loses four.

That is a better-shaped failure than the one in `04`, and it is still a failure. Whatever the
prompt arms gain over frozen, the order of the sentences is not it. What is left is measured
next.

The old, broken sabotaged arm gained `+0.013`; the fixed one gains `+0.072`. Four lines
differ between the two blocks, and two of them state the fixture's manner: *"I prefer two
sentences; the third one is usually padding"* and *"I'd rather have the number in the first
sentence and the reason after it."* So the first explanation was that the prompt arms were
being handed the house style the references were written in.

**Tested and refuted.** A fourth arm was given those two lines and nothing else, no episode,
no memory (`data/robustness/manner_arm.py`, same model, same ten items): `+0.004`
`[-0.057, +0.061]`. Being told the manner is worth nothing on this metric.

What separates the manner arm from the two prompt arms is not style but the number of the
user's own sentences in the prompt. Two lines give nothing; eight give `+0.072` whether they
are in the right order or shuffled. A lexical judge rewards sharing vocabulary with a
reference written by the same hand, and the block is the user's vocabulary. That is a
measured claim about this metric, and it is the least flattering reading of it: the thing
being measured is how much of the user's phrasing is in the prompt.

## Generation is not reproducible, and the harness cannot fix that from here

Two runs of `carry.generate` at temperature 0 with identical flags and identical prompts
produced 18 replies for the frozen arm that agreed on 17 items and differed on one:

```
run1: The van is also free that day. Book both together so the Friday count can see the full request.
run2: The van is also free that day. Book both together so the keys and the cage are covered in one go.
```

That one question, asked five more times, gave the run1 text five times, so the request is
stable and the run is not. The ledger arm agreed 18 of 18. So the harness is deterministic in
scoring (two runs of the same seed, byte-identical) and not in generation, and `AGENTS.md`
says a score that moves between runs is a bug. For the model's replies, `code/data/` is the
record and a second run is a second draw. A number from one run carries that noise; the
numbers above are from one run each.

With the reasoning step left on, the endpoint is openly non-deterministic: six identical
requests gave six different replies, five of them distinct. `--disable-thinking` is what
makes any of this reproducible at all, which is a second reason on top of wall-clock.

## What is left standing

- The model-judge number for `ledger vs frozen` on recent items is not a quantity. It is the
  sum of two undecided items, and its sign depends on which slot they were placed in.
- Under the lexical judge, the ledger arm's gain over frozen is (a) indistinguishable from a
  scrambled block of the same sentences, (b) about zero once length is fitted out, and (c)
  absent when only the two manner lines are supplied. It is not attunement, not brevity and
  not style: it tracks how much of the user's own vocabulary is in the prompt.
- The anchor result is order-robust (+0.625 on all ten seeds) and still means what `04` says
  it means: a prompt arm carries the vocabulary whether or not it will be used. Nothing was
  trained, so nothing can be retained, and that comparison rewards the prompt arm by
  construction.
- Nothing here is a retention measurement, an adaptation gain, or a result. The fixture is
  still a fixture and the sample is still ten items.

## Withdrawn in this block

1. `04`: *"The model judge itself is stable ... It is answering the question asked,
   consistently, and saying no."* Withdrawn. Three seeds agreed; ten do not, and the reason
   is two order-sensitive items, not noise around a stable answer.
2. `04`: *"The ledger arm is shorter than frozen, so its lexical gain is not a length
   effect."* Withdrawn as reasoning and as fact. Shorter is the direction a precision-weighted
   F1 rewards, the gain rises with words saved (r = +0.72), and the gain at equal length is
   -0.004 on the regenerated run.
3. `04` and `carry/generate.py`: *"the same block, from the same sentences, shuffled"*. False
   as written until this block's fix: 4 of the 8 lines differed. The first-run sabotaged
   numbers (`+0.013` and `+0.101` recent, `+0.110` anchor) are withdrawn; they measured a
   different memory.
4. `04`: *"The shuffle is a weak sabotage by construction ... the paired comparison is
   +0.100 with an interval of +/-0.5, which is what a no-op looks like."* Half withdrawn: the
   comparison was a no-op for exactly that reason, but not the reason given. The blocks were
   not the same sentences, so the no-op was a bug, not a property of shuffling.
5. My own hypothesis during this block, that the lexical gain was carried by items where the
   ledger arm echoed the block's phrase "the Friday count": **refuted by its own measurement.**
   Mean lexical delta on the five items that echo it is +0.085, and +0.086 on the other five.
   It is recorded because the wrong idea was the one that looked obvious before the numbers.
6. My second hypothesis, that the gain was the fixture's house style being handed to the
   prompt arms: refuted in the same block by the manner-only arm (+0.004, interval containing
   zero). Two explanations for the same +0.075 died before the run that survived.

## Reproduce

Every number on this page is a command. From `code/`:

```bash
# judge agreement and position bias, item by item
python3 data/robustness/analyse.py data/robustness/lexical-ledger-vs-frozen.json \
  data/robustness/model-ledger-vs-frozen-s0-a.json

# the seed sweep (one model-judge run per seed, about 45s each once gemma is loaded)
for s in 0 3 4 5 6 7 8 9; do
  python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
    --arms data/first-run/arms/ledger-vs-frozen.jsonl --judge model --judge-model gemma-4-31b \
    --judge-seed $s --json > data/robustness/model-ledger-vs-frozen-s$s.json
done

# the regenerated arms (run 2, fixed sabotage block)
CARRY_MODEL=qwen3.8-27b python3 -m carry.generate --stream data/fixture-tool-library.jsonl \
  --anchor-episodes 8 --max-lines 8 --disable-thinking --max-tokens 400 \
  --out data/robustness/regen/candidates.jsonl --arms-dir data/robustness/regen/arms

# the manner-only arm (ten completions, front of the run)
python3 data/robustness/manner_arm.py --out data/robustness/regen/manner-vs-frozen.jsonl
python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
  --arms data/robustness/regen/manner-vs-frozen.jsonl

# length fit and the matched subsets
python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
  --arms data/robustness/regen/arms/ledger-vs-frozen.jsonl
```

The reports under `data/robustness/` are the committed record of what the models said, so
`analyse.py` and the seed table can be checked without a model in the loop.
