# Continual learning and forgetting

Checked 2026-11-03.

## Kirkpatrick et al., 2017

**Overcoming catastrophic forgetting in neural networks.** James Kirkpatrick, Razvan
Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran
Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia
Clopath, Dharshan Kumaran, Raia Hadsell. *PNAS* 114(13). https://arxiv.org/abs/1612.00796

Elastic weight consolidation. The canonical statement that sequential training on new
tasks destroys old ones, plus one fix: penalise movement in the parameters that mattered
for earlier tasks.

Why it matters here: this is the exact tension the anchor set is designed to measure. An
adapter updated every day is a sequential learner, and retention is not a bonus
criterion, it is the thing most likely to fail silently.

## Parisi et al., 2019

**Continual lifelong learning with neural networks: A review.** German I. Parisi, Ronald
Kemker, Jose L. Part, Christopher Kanan, Stefan Wermter. *Neural Networks* 113.
https://arxiv.org/abs/1802.07569

Survey of the field: regularisation, dynamic architectures, replay. Useful mainly for
the taxonomy and for how consistently the benchmark settings differ from a real
deployment, where the stream is unlabelled and the task boundaries are invisible.

Why it matters here: our setting is the awkward one. No task labels, no episode
boundaries, no test set at training time. The survey's methods mostly assume at least
one of those.
