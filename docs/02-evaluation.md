# Evaluation: the actual hard part

Everything below is the reason this repository exists.

## The signal is an absence

A human does not emit labels. What we want to learn from is "that landed right", and
it arrives as: he did not correct me, he did not re-brief me, he came back later.

Two other things produce exactly the same observation.

- The agent has drifted and the user has stopped bothering to correct it.
- The agent has gone silent or dead, and has nothing to say.

So the raw stream is confounded with its own failure modes. Any method that treats
"no correction" as a positive label will, over enough time, train itself into the
first failure while reading it as success.

This is not a hypothetical. In the system this project grew out of, a mail probe failed
on every tick for nine days while its job reported healthy, and a feed layer was found
to be indistinguishable between "quiet" and "dead". Absence of complaint is the least
trustworthy signal on the machine.

## What we do instead

We do not train on the absence. We evaluate against held-out behaviour, and only
update when a held-out number improves.

### Held-out construction

- Split strictly by time. Never by random sample: adjacent turns leak.
- Hold out whole episodes, not turns. A response can be predicted from its own episode
  if half the episode is in training.
- Two held-out sets with different roles: **recent** (just after the training window)
  and **anchor** (an older segment, fixed forever). Recent detects learning. Anchor
  detects forgetting. Anchor must never be rotated, or it stops anchoring anything.
- The anchor is the one number that cannot be gamed by training longer, because the
  model has not seen that period in the last several updates.

### What a held-out item is

A triple: the context available at that moment, the response that actually happened,
and a reference. The reference is the point of contention and has three candidate
sources, in increasing cost:

1. **Self-consistency.** Generate from the frozen base model several times and take
   the modal response. Measures "moved away from generic", which is not the same as
   "moved towards him". Cheap, weak, useful only as a sanity floor.
2. **Preference pairs from history.** Where the user corrected the agent, the pair
   (rejected, accepted) is a genuine preference with a label that arrived naturally.
   These are rare but they are real, and a correction is the only place in the stream
   where the user tells you the truth without being asked.
3. **Judged comparison.** A stronger model or a human compares the adapted arm to the
   frozen arm on held-out context. Expensive, and the judge must be blind to which arm
   it is seeing.

We expect (2) to be the load-bearing one. Corrections are scarce, and that scarcity is
the main practical constraint on the size of any evaluation.

### Metrics

- **Adaptation gain**: held-out win rate of adapted vs frozen, with a bootstrap
  interval. Reported as a paired difference on identical items, never as two separate
  scores.
- **Retention**: the same comparison on the anchor set. A gain that costs retention is
  a different, worse result.
- **Sabotage sensitivity**: repeat the whole evaluation with an adapter trained on
  corrupted data (shuffled turns, or turns from an unrelated period). If this scores
  as well as the real adapter, the metric is measuring fine-tuning volume. This is a
  gate, not a diagnostic.
- **Verbosity, agreement rate and hedging rate**: reported alongside every win rate,
  because a model that says "yes" more often and writes more will win naive
  comparisons. Controlling for these is not optional.

## Ways this evaluation lies

- **Judge contamination.** If the judge is a model, it prefers its own register. A
  judge that is the same family as the candidate is not a judge.
- **Blind-arm leakage.** If the judge can tell which arm produced a response from
  length or formatting, the comparison is over.
- **Unstable anchor.** Rotating the anchor set when scores look bad. This will be
  tempting.
- **Survivorship in the corpus.** The turns that exist are the turns the user chose to
  have. Long gaps, aborted conversations and things never asked are invisible.
- **A single user.** n=1. Every result is a case study, and the harness should say so
  rather than manufacture a significance test that implies generality.

## What we would need to believe a result

A claim of the form "the adapter is better at being with this person" requires, at
minimum: a paired comparison on items the adapter never saw, an anchor that did not
regress, a sabotage control that failed as expected, and verbosity and agreement
controlled. Anything less is a demo.
