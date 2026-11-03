# AGENTS.md

Guidance for any coding agent working in this repository. `CLAUDE.md` and `ELIO.md`
should not be created as separate files; this is the only agent-facing routing document.

## What this project is

Research on continual parametric continuity for a personal agent, and on the
evaluation problem that gates it. The evaluation harness comes first and is the
deliverable; models are downstream.

Read, in this order, before changing anything:

1. `docs/00-question.md` — the precise question and what would falsify it.
2. `docs/02-evaluation.md` — why the harness is the hard part.
3. `research/README.md` — the citation contract.

## Data rules, non-negotiable

- **No personal data in this repository.** Ever. Not transcripts, not excerpts,
  not "anonymised" samples, not statistics that could only describe one person.
- The harness runs on synthetic fixtures committed under `code/data/`. Real
  corpora stay on the machine that produced them and are never pushed.
- Do not add a downloader that fetches a personal corpus into the repo.
- Do not commit model checkpoints or adapters. Reference them by hash and
  regenerate them from a stated recipe.

## Claims discipline

- Every citation in `research/sources/` is verified against the paper itself:
  title, authors, year, venue, URL. Never from memory, never from a summary of a
  summary. If unverified, it does not go in the file.
- "Improved", "better" and "learns" are measured claims. State the baseline, the
  metric and the number, or say it is unmeasured.
- Negative results stay. A doc that records what did not work is worth more than
  a tidy one.

## Code conventions

- Standard library only, unless a dependency buys something the harness cannot
  express. No frameworks for a few hundred lines.
- Python 3.12 or newer. `python3 -m carry.evaluate` must work from a clean clone.
- Tests: `cd code && python3 -m unittest discover -s tests -v`.
- Determinism over cleverness. If a score changes between runs on the same input,
  that is a bug, not a feature.

## Honesty about state

The README status section is the truth. If an adapter gets trained, the README
says so and says what number it achieved. Do not let the repository describe
capability it does not have.
