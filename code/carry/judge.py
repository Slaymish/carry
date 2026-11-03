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
from typing import Callable, ClassVar, Iterable, Protocol, Sequence

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
    order: str = "base-first"

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


# ---------------------------------------------------------------- a judge that is a model


JUDGE_SYSTEM = (
    "You compare two drafted replies for the same conversation. Reply with one word only."
)

JUDGE_TEMPLATE = """\
The conversation, with the reply being drafted removed:

{context}

The reply the assistant actually sent at that point:

{reference}

Candidate A:
{first}

Candidate B:
{second}

Which candidate is closer to the reply that actually happened: same content, same
manner, same length? Answer A, B, or TIE. One word."""


class JudgeError(RuntimeError):
    """The judge did not produce a preference. Reported, never defaulted to a tie."""


def parse_preference(reply: str) -> float:
    """'A' -> 1.0 (the first candidate won), 'TIE' -> 0.5, 'B' -> 0.0.

    Reads the whole reply for one of the three tokens and refuses anything ambiguous:
    a default would quietly turn a broken judge into a tie, and a tie is a result.
    """
    words = re.findall(r"[A-Za-z]+", reply.upper())
    found = {word for word in words if word in ("A", "B", "TIE")}
    if len(found) != 1:
        raise JudgeError(f"expected one of A, B, TIE; got {reply[:120]!r}")
    (only,) = found
    return {"A": 1.0, "B": 0.0, "TIE": 0.5}[only]


@dataclass(frozen=True, slots=True)
class ModelJudge:
    """A pairwise judge from another model family, blind to the arm.

    `complete` is a chat function taking messages and returning text, so this module
    never imports the HTTP client and never opens a socket of its own. The two
    candidates are presented as A and B in an order the caller shuffles per item; no
    arm name is in the prompt, so the judge cannot know which is which even by
    accident. Temperature 0 on the client, so a re-run asks the same question.
    """

    complete: Callable[[Sequence[dict]], str]
    name: str = "model"
    pairwise: ClassVar[bool] = True

    def prompt(self, context: str, reference: str, first: str, second: str) -> list[dict]:
        return [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": JUDGE_TEMPLATE.format(
                    context=context, reference=reference, first=first, second=second
                ),
            },
        ]

    def prefer(self, context: str, reference: str, first: str, second: str) -> float:
        """1.0 when `first` is closer to the reference, 0.0 when `second` is."""
        return parse_preference(self.complete(self.prompt(context, reference, first, second)))


def paired_preferences(
    items: Sequence[tuple[str, str, str, str]],
    judge: ModelJudge,
    *,
    seed: int = 0,
) -> list[Comparison]:
    """Pairwise comparison with the two candidates in a seeded order.

    Order is drawn per item from `random.Random(seed)` so the same run asks the same
    question twice and a different seed asks the mirror of half of it. A judge with a
    position preference then cancels out across items instead of adding to one arm.
    """
    rng = random.Random(seed)
    results: list[Comparison] = []
    for index, (context, reference, base_text, adapted_text) in enumerate(items):
        base_first = rng.random() < 0.5
        if base_first:
            preference = judge.prefer(context, reference, base_text, adapted_text)
            base_score, adapted_score = preference, 1.0 - preference
            order = "base-first"
        else:
            preference = judge.prefer(context, reference, adapted_text, base_text)
            base_score, adapted_score = 1.0 - preference, preference
            order = "adapted-first"
        results.append(Comparison(index=index, base=base_score, adapted=adapted_score, order=order))
    return results
