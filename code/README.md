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
bootstrap interval, the same comparison on the anchor set with its own interval
(retention), a sabotage control, and two nuisance controls. If the sabotage control
matches the real gain, the output says `METRIC VOID` rather than reporting a number.

```
stream      data/sample.jsonl
split       16 train / 5 recent / 3 anchor episodes
judge       lexical-f1
items       5
gain        +0.656  95% CI [+0.613, +0.700]  excludes zero
sabotage    -0.091  control passed
retention   not computed: demo arms are synthetic; nothing was trained, so nothing can be retained
anchor      an anchor is only meaningful once something is actually trained
```

That output is from the demo arms and is not a result. It exists so nobody mistakes a
working harness for a working method.

## Modules

| module | job |
|---|---|
| `stream.py` | JSONL to turns, turns to sittings. Strict schema, refuses to repair. |
| `split.py` | chronological split into train / recent / anchor, with a leakage check |
| `judge.py` | the judge protocol, a lexical floor, paired comparison, bootstrap |
| `evaluate.py` | the CLI, the controls, the sabotage gate, and retention |
| `generate.py` | candidate generation for three arms over an OpenAI-compatible endpoint |
| `judge.py` (model) | a pairwise judge from another model family, blind to the arm |

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

Supply candidates keyed by held-out set and item index:

```json
{"split": "recent", "index": 0, "base": "...", "adapted": "..."}
{"split": "anchor", "index": 0, "base": "...", "adapted": "..."}
```

```bash
python3 -m carry.evaluate --stream my-stream.jsonl --arms arms.jsonl
```

`split` defaults to `recent`, so a file written before anchors were scored still loads.
`generate.py` also writes `ledger-vs-sabotaged.jsonl`: the sabotage gate as a paired
interval between the two prompt arms, with the reference held fixed.
The `base` arm is normally the frozen model, the `adapted` arm the model with the
adapter, or with the ledger prompt. Both are scored against the same reference on the
same context, so the mean of per-item differences is a paired quantity and the interval
means something.

### The model judge

The lexical judge measures overlap. A claim about attunement needs a judge that can read,
so `--judge model` swaps in a pairwise judge from another model family:

```bash
python3 -m carry.evaluate --stream data/fixture-tool-library.jsonl --anchor-episodes 8 \
  --arms data/first-run/arms/ledger-vs-frozen.jsonl --judge model --judge-model gemma-4-31b
```

Two candidates, labelled A and B in an order drawn per item from `random.Random(seed)`
(`--judge-seed`), never named, asked which is closer to the reply that actually happened.
A reply that is not exactly one of A, B or TIE raises rather than defaulting to a tie,
because a default would turn a broken judge into a result. The judge client runs at
temperature 0 with the reasoning step off; that is a stated choice, since a judge that
thinks first writes the same labels more slowly.

### Retention

Rows tagged `"split": "anchor"` are scored as the same paired comparison on the oldest
held-out episodes, with their own bootstrap interval. That is the number training longer
cannot buy: recent gain says it learned, retention says it did not lose what it knew.

The harness will not print a retention number it did not measure. In demo mode it says
nothing was trained, so nothing can be retained; with an arms file that has no anchor
rows it says so; with fewer than three anchor items it says the interval would be
meaningless. Every run prints the same line underneath: an anchor is only meaningful
once something is actually trained.

## Generating candidates

`generate.py` talks to a local OpenAI-compatible endpoint over `urllib`, standard library
only, and produces the three arms from one stream:

- **frozen** the base model with no memory.
- **ledger** the same model with a short block of the user's own earlier sentences, quoted
  verbatim from the training episodes.
- **sabotaged** the same block from the same sentences shuffled, so length, vocabulary
  and register are identical and only the pairing is wrong.

Shuffling is the weakest form of this control: every sentence stays true, so the arm is
still informed, merely out of order. It is a floor, and it needs a foreign-content
control beside it before anyone claims a judge is measuring attunement.

```bash
CARRY_BASE_URL=http://127.0.0.1:8080/v1 CARRY_MODEL=qwen3.8-27b \
  python3 -m carry.generate --stream data/sample.jsonl \
    --out candidates.jsonl --arms-dir arms/
```

Output is one JSON row per held-out item, with `frozen`, `ledger` and `sabotaged` in full.
`--arms-dir` also writes `ledger-vs-frozen.jsonl` and `sabotaged-vs-frozen.jsonl` in the
harness's own format, for both held-out sets, so `carry.evaluate --arms` scores them
without an adapter.

`--print-prompts` shows exactly what would be sent and exits without touching the
network. Temperature is 0 and nothing is retried: a model that fails is reported against
the item it failed on.

The two prompt arms are a floor for the parametric claim, not the claim. A frozen base
model with a hand-made ledger is the outcome [`../docs/03-open-questions.md`](../docs/03-open-questions.md)
says would end the project if it matches an adapted model.

## Two things this deliberately does not do

- **Judge a reply as good or bad.** `LexicalJudge` measures overlap. It is a floor and a
  sabotage detector. A real judge is a separate component with its own blindness
  requirements, documented in `../docs/02-evaluation.md`.
- **Invent a retention number.** With no arms there is nothing to retain, and the
  harness says exactly that rather than reporting a zero.

## Data

`data/fixture-tool-library.jsonl` is the first real fixture: 48 synthetic episodes from
`data/make_fixture.py`, split 30 / 10 / 8, with the model output it produced committed
under `data/first-run/` so the scores in `../docs/04-first-number.md` can be recomputed
without a model. `data/sample.jsonl` is synthetic, written for this repository. The real corpus never
enters the repo: see the data rules in `../AGENTS.md`. Five held-out items is enough to
prove the plumbing and nowhere near enough to prove a method, which is why the sample is
labelled demo rather than fixture.
