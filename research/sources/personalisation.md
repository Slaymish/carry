# Personalising a model to one person

Checked 2026-11-03.

## Salemi et al., 2023

**LaMP: When Large Language Models Meet Personalization.** Alireza Salemi, Sheshera
Mysore, Michael Bendersky, Hamed Zamani. arXiv. https://arxiv.org/abs/2304.11406

A benchmark of personalisation tasks, each built from a user's own history, scored
against held-out items.

Why it matters here: it is the closest existing thing to an evaluation of the property
we care about, and it is built on labelled held-out items from real user history. That
is the shape we need and the data we do not have for a single-user setting. Its
existence is also the strongest argument that "does personalisation work" is already a
measurable question and our contribution has to be narrower than that.

## Hu et al., 2021

**LoRA: Low-Rank Adaptation of Large Language Models.** Edward J. Hu, Yelong Shen,
Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. arXiv.
https://arxiv.org/abs/2106.09685

Low-rank adapters: train a small number of parameters, leave the base frozen.

Why it matters here: it makes the substrate practical. A per-period adapter over a
model small enough to run locally is a realistic shape for this project, and adapters
can be swapped and rebuilt without touching the base weights. The cost is that an
adapter is not a well-understood object, which is why the judge matters more than the
method.
