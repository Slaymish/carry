# code

The evaluation harness. Standard library Python, no model, no dependencies.

```bash
cd code
python3 -m carry.evaluate --stream data/sample.jsonl
python3 -m unittest discover -s tests -v
```

Requires Python 3.12 or newer.

## What it does

Reads an interaction stream, cuts it into sittings, splits those strictly by time, and
scores two candidate arms against what actually happened. It prints a paired gain with a
bootstrap interval, a sabotage control, and two nuisance controls. If the sabotage
control matches the real gain, the output says `METRIC VOID` rather than reporting a
number.

```
stream      data/sample.jsonl
split       16 train / 5 recent / 3 anchor episodes
judge       lexical-f1
items       5
gain        +0.656  95% CI [+0.613, +0.700]  excludes zero
sabotage    -0.091  control passed
```

That output is from the demo arms and is not a result. It exists so nobody mistakes a
working harness for a working method.

## Modules

| module | job |
|---|---|
| `stream.py` | JSONL to turns, turns to sittings. Strict schema, refuses to repair. |
| `split.py` | chronological split into train / recent / anchor, with a leakage check |
| `judge.py` | the judge protocol, a lexical floor, paired comparison, bootstrap |
| `evaluate.py` | the CLI, the controls, and the sabotage gate |

## The stream schema

One JSON object per line. Anything else is ignored, so a dump from another harness
reduces to this without touching the code.

```json
{"ts": "2026-11-03T22:41:00Z", "role": "user", "text": "..."}
```

`role` is `user` or `assistant`. A missing offset on a timestamp is read as UTC, not
local time, because a silent timezone shift would move episode boundaries and nothing
would look wrong.

## Scoring real arms

Supply candidates keyed by held-out item index:

```json
{"index": 0, "base": "...", "adapted": "..."}
```

```bash
python3 -m carry.evaluate --stream my-stream.jsonl --arms arms.jsonl
```

The `base` arm is normally the frozen model, the `adapted` arm the model with the
adapter. Both are scored against the same reference on the same context, so the mean of
per-item differences is a paired quantity and the interval means something.

## Two things this deliberately does not do

- **Judge a reply as good or bad.** `LexicalJudge` measures overlap. It is a floor and a
  sabotage detector. A real judge is a separate component with its own blindness
  requirements, documented in `../docs/02-evaluation.md`.
- **Report retention.** Retention needs both arms scored on the anchor set, which needs
  trained arms. The harness refuses to invent a number for it.

## Data

`data/sample.jsonl` is synthetic, written for this repository. The real corpus never
enters the repo: see the data rules in `../AGENTS.md`. Five held-out items is enough to
prove the plumbing and nowhere near enough to prove a method, which is why the sample is
labelled demo rather than fixture.
