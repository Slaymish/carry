# Deleting from a parametric memory

Checked 2026-11-03.

## Survey: machine unlearning in large language models

**A Survey of Machine Unlearning in Large Language Models.** arXiv, 2025.
https://arxiv.org/abs/2503.01854

Taxonomy of unlearning approaches, evaluation methods and their weaknesses.

Why it matters here: it is the honest counterargument to "rebuild the adapter from
retained data". Unlearning promises deletion without retaining anything, which would
remove the privacy cost that makes rebuild expensive. The survey's own evaluation
sections are the relevant part: unlearning is measured by probes, and a probe can pass
while the information is still recoverable.

Our position is that rebuild is the honest default for this project and unlearning is a
later optimisation, only once a probe suite exists that could detect a failed deletion.

## Position, not a citation

Any deletion claim on this project has to show a probe that changes. "Removed from the
training set" is a statement about a file, not about a model.
