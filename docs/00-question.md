# The question

## Short form

Can a personal agent's continuity live in its parameters, updated continuously from
an unlabelled interaction stream, and can anyone tell whether those updates made it
better?

The second clause is the project. The first is engineering.

## Why not just keep writing files

Memory files hold what can be stated: a preference, a commitment, a fact with a
source. They cannot hold a manner, which is what someone actually recognises. A
style guide is a manner described by someone who does not have it.

There is a second reason, and it is measurable. A transcript-replaying agent pays
for the whole relationship on every turn, and the cost grows with the length of the
relationship while the marginal value of old turns falls to roughly nothing. Continuity
should be roughly flat in cost, and it currently is not.

## What would count as an answer

We would accept all of the following as a positive result:

1. An adapter updated on interaction stream segment *n* scores higher on held-out
   segment *n+1* than the frozen base model, by an interval that excludes zero.
2. The same adapter does not score lower on an older held-out segment. What was
   learned earlier is not eaten by what was learned later.
3. A deliberately corrupted update (trained on shuffled or adversarially bad turns)
   scores *worse*, which proves the metric is not measuring fluency or length.
4. Deleting a segment from the retained data and rebuilding removes its influence,
   demonstrated by a probe that changes.

Point 3 is the one that stops this being self-congratulation. A metric that fails to
punish a sabotaged update is not measuring adaptation.

## What would falsify it

- Held-out scores that move with the amount of training but not with the quality of
  it. That means the harness is detecting "more fine-tuning", not "better attunement".
- A preference for the adapted model that vanishes when length, verbosity or
  agreement rate is controlled for. Sycophancy is the default failure of every
  personalisation result, and it will flatter us if we let it.
- No separation between a good update and a sabotaged one. Then there is no signal
  to train on, and the project has no floor.

## The honest prior

Our prior is that (1) and (2) are achievable and unremarkable, (3) is achievable and
is the actual contribution, and (4) is achievable but ugly, because it means rebuild
rather than subtract.

See [`02-evaluation.md`](02-evaluation.md) for how the stream becomes a number.
