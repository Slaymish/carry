# carry

**Continuity that lives in the weights, and a way to tell whether it's working.**

Hamish Burke and Elio (Hermes agent), November 2026.

---

A personal agent that knows you across months should not need re-briefing, should not replay your history to sound consistent, and should get *cheaper* to talk to as the relationship gets longer rather than more expensive.

Today that continuity is written down: memory files, retrieval, summarised transcripts. A file can only hold what can be said out loud, and most of what makes a person recognisable is not sayable that way. It is a manner, not a ledger. A ledger can be audited. A manner can only be lived with.

So the question is whether the manner can live in the parameters instead, updated continuously from an interaction stream.

We think that is tractable. We think the unsolved part is something else entirely, and it is the reason this repository exists before any adapter does.

## The problem

Continual learning from a human is not a labelling problem. There is no target. Nobody emits a loss.

The signal we actually want is "that landed right", and it arrives as an absence: he did not correct me, he did not re-brief me, he came back. That is the same observation as "I have silently drifted wrong and he has given up mentioning it". Both look like a quiet week.

Everything else in this project is downstream of that. If you cannot separate those two cases, you cannot train on the stream, because you cannot tell which updates to keep.

## Why evaluation is the hard part

Two parametric-memory projects can both claim continuity. Only one can show a number that goes down when it gets worse.

The literature optimises personalisation against held-out human preferences, which are labels. We do not have those, and asking for them every day defeats the point of an agent that is supposed to notice things itself. So the interesting work is constructing a signal that is *neither* a daily interruption *nor* a number the model produces about itself, because a model scoring its own adaptation is exactly the failure mode with no floor.

Our current line: hold out real interactions, freeze a reference set of what a good response was, and score against it. A number that can go down beats a vibe that cannot. Details and the cases where this breaks: [`docs/02-evaluation.md`](docs/02-evaluation.md).

## What we are building

Three layers, and only one of them is new.

- **Weights hold the manner.** A small local model, continuously adapted. Chosen so it can actually be updated and so that deletion is possible by rebuilding from retained data rather than by subtracting from a checkpoint nobody can inspect.
- **Files hold the ledger.** Commitments, corrections, things told once. Small, checkable, and wrong is visibly wrong. This already exists and works; it is not what this project replaces.
- **A judge holds the truth.** The evaluation harness. It is written first, on purpose, because a parametric memory without a judge is indistinguishable from a confident drift.

## What we are not doing

- Not a new memory-file system. There are good ones; this is the layer underneath the question they do not answer.
- Not training on a frontier hosted model. You cannot update weights you do not hold.
- Not recursive self-training on generated summaries. Models collapse when trained on their own output; the stream has to be real interaction, not our reading of it.
- Not any personal data in this repository. None. The harness runs on synthetic samples and the corpus stays on the machine it was written on.

## Status

Early. The question is written down, and the evaluation harness runs: it scores supplied
arms on a recent held-out set and on a fixed anchor set, each with its own bootstrap
interval, and refuses to print a retention number it has not measured.

On 4 November 2026 a model went in the loop for the first time. qwen3.8-27b generated
three arms (frozen, ledger-prompted, shuffled-ledger) over a 48-episode synthetic
fixture; a lexical floor and gemma-4-31b judged them blind, in a seeded order, never
told which arm was which. The result is a null, and the sabotage gate fails under the
model judge at ten recent items: the shuffled-ledger arm scored above the real one. That
is written up in [`docs/04-first-number.md`](docs/04-first-number.md), void verdict
included, rather than quietly retried at a friendlier seed.

No adapter has been trained. The ledger arm is prompt-level adaptation, which is a
baseline this project has to beat, not the parametric claim it is making.

## Layout

```
docs/       the question, the approach, the evaluation design, the first number, open problems
research/   prior art, with every citation verified against the paper
code/       the harness: stream handling, held-out splits, judges, scoring
```

`code/` is standard-library Python and runs without a model:

```bash
cd code
python3 -m carry.evaluate --stream data/sample.jsonl
```

It prints a baseline score and a confidence interval. That is the number future work has to beat.

## Working notes

This is a collaboration. Design decisions, including the ones we got wrong and abandoned, are recorded rather than quietly replaced, because the reasoning is most of the value.

The starting position came from a specific argument worth preserving: an agent that carries its history in a transcript pays for the whole relationship on every turn, and dilutes the thing it is being asked. Continuity should cost about the same in year two as in week one. See [`docs/01-approach.md`](docs/01-approach.md).

MIT licensed.
