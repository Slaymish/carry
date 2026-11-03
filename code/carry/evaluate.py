"""Run the evaluation and print the number.

    python3 -m carry.evaluate --stream data/sample.jsonl

With no trained arms the harness runs in demo mode: it builds two synthetic arms from
the reference so the plumbing, the interval and the sabotage control can be exercised
without a model. Demo output is labelled and is not a result.

To score real arms, pass candidates as JSONL keyed by item index:

    {"index": 0, "base": "...", "adapted": "..."}

The measured quantities are the ones the design doc commits to: paired gain on recent
held-out episodes, a bootstrap interval, and a sabotage control that must fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .judge import LexicalJudge, bootstrap_interval, paired_gain, tokens
from .split import check_no_leakage, holdout
from .stream import Episode, StreamError, group_episodes, load_stream

Item = tuple[str, str, str, str]  # context, reference, base, adapted


def build_items(episodes: Sequence[Episode]) -> list[tuple[str, str]]:
    """One item per episode: the context before the last assistant reply, and that reply.

    Episodes without an assistant reply are skipped rather than filled. A padded item
    is a fabricated item.
    """
    items: list[tuple[str, str]] = []
    for episode in episodes:
        turns = episode.turns
        last = None
        for i in range(len(turns) - 1, -1, -1):
            if turns[i].role == "assistant":
                last = i
                break
        if last is None or last == 0:
            continue
        context = "\n".join(f"{t.role}: {t.text}" for t in turns[:last])
        items.append((context, turns[last].text))
    return items


def demo_arms(pairs: Sequence[tuple[str, str]]) -> list[Item]:
    """Two synthetic arms, so the harness runs with no model at all.

    The base arm repeats the last line of context, which overlaps the reference only
    by accident. The adapted arm is a truncated reference, which the lexical judge
    rewards. That asymmetry is a property of this stub, not evidence about anything.
    """
    items: list[Item] = []
    for context, reference in pairs:
        last_line = context.splitlines()[-1] if context else ""
        base = last_line.split(": ", 1)[-1]
        words = reference.split()
        adapted = " ".join(words[: max(1, len(words) * 2 // 3)])
        items.append((context, reference, base, adapted))
    return items


def load_arms(path: Path, pairs: Sequence[tuple[str, str]]) -> list[Item]:
    """Attach supplied candidates to the held-out pairs, in index order."""
    supplied: dict[int, dict[str, str]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        index = record.get("index")
        if not isinstance(index, int):
            raise StreamError(f"{path}:{number}: index must be an integer")
        if not isinstance(record.get("base"), str) or not isinstance(record.get("adapted"), str):
            raise StreamError(f"{path}:{number}: base and adapted must be strings")
        supplied[index] = {"base": record["base"], "adapted": record["adapted"]}
    missing = [i for i in range(len(pairs)) if i not in supplied]
    if missing:
        raise StreamError(f"{path}: no candidate for item(s) {missing[:5]}")
    return [
        (context, reference, supplied[i]["base"], supplied[i]["adapted"])
        for i, (context, reference) in enumerate(pairs)
    ]


def controls(items: Sequence[Item], judge) -> dict[str, float]:
    """Agreement-drift and verbosity, reported next to every score.

    A model that agrees more and writes more wins naive comparisons. These two numbers
    are the cheapest defence and they are not optional.
    """
    def agree(text: str) -> bool:
        low = text.lower().strip()
        return low.startswith(("yes", "yep", "sure", "ok", "okay", "you're right", "agreed"))

    def words(text: str) -> int:
        return len(tokens(text))

    n = max(1, len(items))
    return {
        "base_agreement_rate": sum(agree(i[2]) for i in items) / n,
        "adapted_agreement_rate": sum(agree(i[3]) for i in items) / n,
        "base_mean_words": sum(words(i[2]) for i in items) / n,
        "adapted_mean_words": sum(words(i[3]) for i in items) / n,
    }


def sabotage(items: Sequence[Item]) -> list[Item]:
    """The control that decides whether the metric means anything.

    Pair each item's adapted response with the *next* item's reference. If an adaption
    trained on nonsense still wins, the judge is measuring fluency or length rather
    than attunement, and the result is void.
    """
    shifted = items[1:] + items[:1]
    return [
        (context, reference, base, other[3])
        for (context, reference, base, _adapted), other in zip(items, shifted)
    ]


def run(stream: Path, *, gap_hours: float, recent_fraction: float, anchor: int, arms: Path | None) -> dict:
    turns = load_stream(stream)
    episodes = group_episodes(turns, gap=__import__("datetime").timedelta(hours=gap_hours))
    split = holdout(episodes, recent_fraction=recent_fraction, anchor_episodes=anchor)
    check_no_leakage(split)

    pairs = build_items(split.recent)
    if len(pairs) < 3:
        raise StreamError(
            f"only {len(pairs)} usable held-out item(s); a paired comparison needs at least 3"
        )

    items = load_arms(arms, pairs) if arms else demo_arms(pairs)
    judge = LexicalJudge()
    comparisons = paired_gain(items, judge)
    deltas = [c.delta for c in comparisons]
    low, high = bootstrap_interval(deltas)

    sabotaged = paired_gain(sabotage(items), judge)
    sabotage_delta = sum(c.delta for c in sabotaged) / len(sabotaged)

    return {
        "mode": "supplied-arms" if arms else "demo-synthetic-arms",
        "warning": None if arms else "demo arms are built from the reference; not a result",
        "judge": judge.name,
        "stream": str(stream),
        "counts": split.counts(),
        "items": len(items),
        "gain": sum(deltas) / len(deltas),
        "interval": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "sabotage_gain": sabotage_delta,
        "sabotage_beats_real": bool(sabotage_delta >= sum(deltas) / len(deltas)),
        "retention": "not computed: needs trained arms scored on the anchor set",
        "controls": controls(items, judge),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carry.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--stream", required=True, type=Path, help="interaction stream JSONL")
    parser.add_argument("--arms", type=Path, help="candidate responses JSONL keyed by item index")
    parser.add_argument("--gap-hours", type=float, default=3.0, help="episode boundary (default 3)")
    parser.add_argument("--recent-fraction", type=float, default=0.25, help="held-out tail share")
    parser.add_argument("--anchor-episodes", type=int, default=3, help="fixed old episodes")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        report = run(
            args.stream,
            gap_hours=args.gap_hours,
            recent_fraction=args.recent_fraction,
            anchor=args.anchor_episodes,
            arms=args.arms,
        )
    except StreamError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    if report["warning"]:
        print(f"!! {report['warning']}")
    print(f"stream      {report['stream']}")
    print(f"split       {report['counts']['train_episodes']} train / "
          f"{report['counts']['recent_episodes']} recent / "
          f"{report['counts']['anchor_episodes']} anchor episodes")
    print(f"judge       {report['judge']}")
    print(f"items       {report['items']}")
    low, high = report["interval"]
    print(f"gain        {report['gain']:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]  "
          f"{'excludes zero' if report['excludes_zero'] else 'includes zero'}")
    print(f"sabotage    {report['sabotage_gain']:+.3f}  "
          f"{'METRIC VOID: sabotage matches real' if report['sabotage_beats_real'] else 'control passed'}")
    print(f"retention   {report['retention']}")
    for key, value in report["controls"].items():
        print(f"{key:<24}{value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
