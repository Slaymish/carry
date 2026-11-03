"""An interaction stream, and the episodes it is made of.

A stream is JSONL, one object per line:

    {"ts": "2026-11-03T22:41:00Z", "role": "user"|"assistant", "text": "..."}

Anything else is ignored, so a dump from different harnesses can be reduced to this
schema without touching the harness. Timestamps must be ISO 8601; offsets are honoured,
and a missing offset is read as UTC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

ROLES = ("user", "assistant")


class StreamError(ValueError):
    """The stream is not in the schema above. Never repaired silently."""


def parse_ts(value: str) -> datetime:
    """ISO 8601 to an aware datetime. Naive values are UTC, not local time."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise StreamError(f"unparseable timestamp: {value!r}") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


@dataclass(frozen=True, slots=True)
class Turn:
    ts: datetime
    role: str
    text: str

    def to_json(self) -> dict[str, str]:
        return {"ts": self.ts.isoformat(), "role": self.role, "text": self.text}


@dataclass(frozen=True, slots=True)
class Episode:
    """One sitting. Contiguous turns with no gap longer than the threshold."""

    turns: tuple[Turn, ...]

    @property
    def started(self) -> datetime:
        return self.turns[0].ts

    @property
    def ended(self) -> datetime:
        return self.turns[-1].ts

    def __len__(self) -> int:
        return len(self.turns)

    def user_turns(self) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.role == "user")


def iter_turns(lines: Iterable[str], *, source: str = "<stream>") -> Iterator[Turn]:
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StreamError(f"{source}:{number}: not JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise StreamError(f"{source}:{number}: expected an object, got {type(record).__name__}")
        role = str(record.get("role", "")).strip()
        if role not in ROLES:
            raise StreamError(f"{source}:{number}: role must be one of {ROLES}, got {role!r}")
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise StreamError(f"{source}:{number}: text must be a non-empty string")
        yield Turn(parse_ts(str(record.get("ts", ""))), role, text)


def load_stream(path: str | Path) -> list[Turn]:
    """Read a stream from disk, sorted by time. A tie keeps file order."""
    file = Path(path)
    if not file.is_file():
        raise StreamError(f"no such stream: {file}")
    turns = list(iter_turns(file.read_text(encoding="utf-8").splitlines(), source=str(file)))
    if not turns:
        raise StreamError(f"{file}: stream is empty")
    return sorted(turns, key=lambda turn: turn.ts)


def group_episodes(turns: Sequence[Turn], *, gap: timedelta = timedelta(hours=3)) -> list[Episode]:
    """Group into sittings.

    The gap threshold is the definition of "the same conversation" and it is a
    judgement, not a fact. It is a parameter so that a result can be re-read at a
    different one rather than depending on a constant nobody examined.
    """
    if gap <= timedelta(0):
        raise StreamError("gap must be positive")
    episodes: list[Episode] = []
    current: list[Turn] = []
    for turn in turns:
        if current and turn.ts - current[-1].ts > gap:
            episodes.append(Episode(tuple(current)))
            current = []
        current.append(turn)
    if current:
        episodes.append(Episode(tuple(current)))
    return episodes
