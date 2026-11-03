# Research

Prior art for the two halves of the project: continual adaptation, and knowing whether
it worked.

## The citation contract

Every entry under `sources/` is verified against the paper: title, authors, year, venue,
URL. **Never from memory.** A remembered citation is the most common way a research
repository quietly becomes untrustworthy, and the failure is invisible because every
citation looks equally confident.

Rules:

- Verify before writing. If it cannot be verified, it does not go in.
- One line on why it matters to *this* project. A reading list with no argument is a
  bibliography, not research.
- Record disagreement with a paper, not just agreement.
- Date entries when checked, because the field moves and the notes should show their age.

## Files

| file | covers |
|---|---|
| `sources/continual-learning.md` | forgetting, stability, online updating |
| `sources/personalisation.md` | adapting a model to one person |
| `sources/collapse-and-recursion.md` | training on your own output |
| `sources/unlearning.md` | deleting from a parametric memory |

Reading order for someone new: `continual-learning`, then `collapse-and-recursion`,
then the evaluation doc. The order matters because the middle one removes the obvious
shortcut people reach for first.
