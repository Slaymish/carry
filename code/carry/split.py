"""Held-out construction.

The rules here exist because the obvious split leaks. Splitting by turn lets a
response be predicted from its own episode. Shuffling leaks the future into the past.
Both produce a score that goes up and means nothing.

So: split by episode, split by time, and keep an anchor set that is never rotated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

from .stream import Episode, StreamError


@dataclass(frozen=True, slots=True)
class Split:
    """A training window and two held-out sets with different jobs.

    train    everything up to the cutoff, the only material an adapter may see
    recent   the first episodes after the cutoff. Detects learning.
    anchor   a fixed, older stretch, never rotated. Detects forgetting.
    """

    train: tuple[Episode, ...]
    recent: tuple[Episode, ...]
    anchor: tuple[Episode, ...]

    def counts(self) -> dict[str, int]:
        return {
            "train_episodes": len(self.train),
            "recent_episodes": len(self.recent),
            "anchor_episodes": len(self.anchor),
            "train_turns": sum(len(e) for e in self.train),
            "recent_turns": sum(len(e) for e in self.recent),
            "anchor_turns": sum(len(e) for e in self.anchor),
        }


def _assert_disjoint(parts: Sequence[Sequence[Episode]]) -> None:
    seen: set[int] = set()
    for episodes in parts:
        for episode in episodes:
            for turn in episode.turns:
                marker = id(turn)
                if marker in seen:
                    raise StreamError("split is not disjoint: a turn appears twice")
                seen.add(marker)


def holdout(
    episodes: Sequence[Episode],
    *,
    recent_fraction: float = 0.2,
    anchor_episodes: int = 3,
) -> Split:
    """Split chronologically into train / recent / anchor.

    Anchor is taken from the oldest end of the training material, so it is genuinely
    old rather than merely unseen, and it is removed from train rather than copied out
    of it. An anchor that also trains is an anchor that measures nothing.
    """
    if not 0 < recent_fraction < 1:
        raise StreamError("recent_fraction must be between 0 and 1")
    if anchor_episodes < 1:
        raise StreamError("anchor_episodes must be at least 1")
    ordered = sorted(episodes, key=lambda e: e.started)
    if len(ordered) < anchor_episodes + 4:
        raise StreamError(
            f"need at least {anchor_episodes + 4} episodes to split, got {len(ordered)}"
        )

    anchor = tuple(ordered[:anchor_episodes])
    remainder = ordered[anchor_episodes:]
    cut = max(1, min(len(remainder) - 1, round(len(remainder) * (1 - recent_fraction))))
    split = Split(train=tuple(remainder[:cut]), recent=tuple(remainder[cut:]), anchor=anchor)
    _assert_disjoint([split.train, split.recent, split.anchor])
    return split


def check_no_leakage(split: Split, *, gap: timedelta = timedelta(0)) -> None:
    """Fail if any held-out episode starts before training material has ended.

    A cheap guard against a caller sorting differently or slicing by hand. It is a
    check, not a proof: the harness cannot see inside a corpus the adapter trained on
    elsewhere.
    """
    if not split.train:
        raise StreamError("empty training window")
    last_train = max(episode.ended for episode in split.train)
    for name, episodes in (("recent", split.recent),):
        for episode in episodes:
            if episode.started <= last_train + gap:
                raise StreamError(f"{name} episode {episode.started.isoformat()} overlaps training")
