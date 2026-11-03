"""Judges, and the paired comparison that produces the number.

A judge answers one question about one item: given what was available, which of two
responses is closer to what actually happened? It never sees which arm produced which
response. Any judge that can tell is not a judge, and the guard in `paired_gain` exists
to make that hard to forget.

Two judges ship here. `LexicalJudge` is a floor, not a result: it measures overlap, so
it rewards echoing. It is present so the harness runs with no model and so that
sabotage can be shown to move the number.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

WORD = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(WORD.findall(text.lower()))


class Judge(Protocol):
    """Score a candidate against the reference for one item, in [0, 1]."""

    name: str

    def score(self, context: str, candidate: str, reference: str) -> float: ...


@dataclass(frozen=True, slots=True)
class LexicalJudge:
    """Unigram overlap with the reference, F1-shaped.

    Deliberately dumb. It cannot judge whether a reply was *right*, only whether it
    used the same words. Useful for plumbing, sabotage checks and a floor. It will
    reward a model that repeats the user, which is why the evaluate step reports
    verbosity next to every score.
    """

    name: str = "lexical-f1"

    def score(self, context: str, candidate: str, reference: str) -> float:
        want = set(tokens(reference))
        got = set(tokens(candidate))
        if not want or not got:
            return 0.0
        overlap = len(want & got)
        if not overlap:
            return 0.0
        precision = overlap / len(got)
        recall = overlap / len(want)
        return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class ScriptedJudge:
    """A judge backed by a lookup, for tests and offline replays.

    The mapping is keyed on the candidate text so a test can assert the plumbing
    without a model in the loop.
    """

    table: dict[str, float]
    default: float = 0.0
    name: str = "scripted"

    def score(self, context: str, candidate: str, reference: str) -> float:
        return self.table.get(candidate, self.default)


@dataclass(frozen=True, slots=True)
class Comparison:
    """One paired item: both arms judged on identical context and reference."""

    index: int
    base: float
    adapted: float

    @property
    def delta(self) -> float:
        return self.adapted - self.base


def paired_gain(
    items: Sequence[tuple[str, str, str, str]],
    judge: Judge,
    *,
    blind: Callable[[int], bool] | None = None,
) -> list[Comparison]:
    """Compare two arms on identical items.

    `items` is (context, reference, base_candidate, adapted_candidate). Each judge call
    receives the same context and reference for both arms, so the comparison is paired
    and a bootstrap over items is meaningful.

    `blind` is an assertion hook: pass a function that returns True when the caller
    believes the judge could identify the arm. Raising here is the point. Silent
    leakage is the failure mode this whole file is trying to prevent.
    """
    if blind is not None:
        for i in range(len(items)):
            if blind(i):
                raise ValueError(f"item {i} is not blind; the comparison is invalid")
    results: list[Comparison] = []
    for index, (context, reference, base_text, adapted_text) in enumerate(items):
        results.append(
            Comparison(
                index=index,
                base=judge.score(context, base_text, reference),
                adapted=judge.score(context, adapted_text, reference),
            )
        )
    return results


def bootstrap_interval(
    deltas: Sequence[float],
    *,
    rounds: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over item deltas. Deterministic given the seed."""
    if not deltas:
        raise ValueError("no deltas to resample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(rounds):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    tail = (1 - confidence) / 2
    low = means[max(0, math.floor(tail * rounds) - 1)]
    high = means[min(rounds - 1, math.ceil((1 - tail) * rounds) - 1)]
    return low, high
