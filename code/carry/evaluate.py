"""Run the evaluation and print the number.

    python3 -m carry.evaluate --stream data/sample.jsonl

With no trained arms the harness runs in demo mode: it builds two synthetic arms from
the reference so the plumbing, the interval and the sabotage control can be exercised
without a model. Demo output is labelled and is not a result.

To score real arms, pass candidates as JSONL keyed by held-out set and item index:

    {"split": "recent", "index": 0, "base": "...", "adapted": "..."}
    {"split": "anchor", "index": 0, "base": "...", "adapted": "..."}

`split` defaults to "recent", so files written before anchors were scored still load.
Rows are per item: `base` and `adapted` are two arms answering the same held-out moment.

The measured quantities are the ones the design doc commits to: paired gain on recent
held-out episodes, a bootstrap interval, the same comparison on the anchor set with its
own interval (retention), and a sabotage control that must fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .judge import (
    LexicalJudge,
    ModelJudge,
    bootstrap_interval,
    paired_gain,
    paired_preferences,
    tokens,
)
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


def compare_items(items: Sequence[Item], judge, *, seed: int = 0):
    """Score the paired comparison with whichever kind of judge was handed in.

    A scoring judge (lexical) sees one candidate at a time; a pairwise judge (a model)
    sees both and only says which is closer. Both give one delta per item, so the same
    bootstrap applies to either. The dispatch is one attribute rather than a second
    evaluation path, because two paths would drift.
    """
    if getattr(judge, "pairwise", False):
        return paired_preferences(items, judge, seed=seed)
    return paired_gain(items, judge)


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


SPLITS = ("recent", "anchor")

MIN_ITEMS = 3

ANCHOR_NOTE = "an anchor is only meaningful once something is actually trained"


def load_arm_records(path: Path) -> dict[tuple[str, int], tuple[str, str]]:
    """Read an arms file into (split, index) keys, refusing anything ambiguous.

    A duplicated key and a missing split are both errors rather than last-one-wins: the
    file is the only record of what the model said, and silently dropping a row would
    change the number.
    """
    records: dict[tuple[str, int], tuple[str, str]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StreamError(f"{path}:{number}: not JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise StreamError(f"{path}:{number}: expected an object, got {type(record).__name__}")
        index = record.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise StreamError(f"{path}:{number}: index must be an integer")
        name = record.get("split", "recent")
        if name not in SPLITS:
            raise StreamError(f"{path}:{number}: split must be one of {SPLITS}, got {name!r}")
        base = record.get("base")
        adapted = record.get("adapted")
        if not isinstance(base, str) or not isinstance(adapted, str):
            raise StreamError(f"{path}:{number}: base and adapted must be strings")
        key = (name, index)
        if key in records:
            raise StreamError(f"{path}:{number}: duplicate {name} item {index}")
        records[key] = (base, adapted)
    return records


def attach_arms(
    records: dict[tuple[str, int], tuple[str, str]],
    pairs: Sequence[tuple[str, str]],
    split_name: str,
) -> list[Item] | None:
    """Attach candidates for one held-out set, in index order.

    Returns None when the file carries nothing for that set. Absent is not empty: a file
    with no anchor rows has not measured retention, and the caller says so rather than
    scoring nothing and printing a zero.
    """
    if not pairs:
        return None
    supplied = {index for index in range(len(pairs)) if (split_name, index) in records}
    if not supplied:
        return None
    missing = [index for index in range(len(pairs)) if index not in supplied]
    if missing:
        raise StreamError(f"no {split_name} candidate for item(s) {missing[:5]}")
    return [
        (context, reference, *records[(split_name, index)])
        for index, (context, reference) in enumerate(pairs)
    ]


def load_arms(path: Path, pairs: Sequence[tuple[str, str]], *, split_name: str = "recent") -> list[Item]:
    """Attach supplied candidates to one held-out set, refusing a partial file."""
    attached = attach_arms(load_arm_records(path), pairs, split_name)
    if attached is None:
        raise StreamError(f"{path}: no {split_name} candidates")
    return attached


def retention(
    records: dict[tuple[str, int], tuple[str, str]],
    anchor_pairs: Sequence[tuple[str, str]],
    judge,
    *,
    scored: bool,
    seed: int = 0,
) -> dict:
    """The same paired comparison, on the anchor set, with its own interval.

    Retention asks a different question from gain: not "did it learn" but "did learning
    the recent material cost what it already knew". It is the one number training longer
    cannot buy, which is exactly why it may not be invented. With demo arms there is no
    adaptation to lose, so the harness refuses before scoring rather than printing a
    confident zero.
    """
    if not scored:
        return {
            "computed": False,
            "items": len(anchor_pairs),
            "reason": "demo arms are synthetic; nothing was trained, so nothing can be retained",
            "note": ANCHOR_NOTE,
        }
    items = attach_arms(records, anchor_pairs, "anchor")
    if items is None:
        return {
            "computed": False,
            "items": len(anchor_pairs),
            "reason": "the arms file carries no anchor rows",
            "note": ANCHOR_NOTE,
        }
    if len(items) < MIN_ITEMS:
        return {
            "computed": False,
            "items": len(items),
            "reason": f"{len(items)} anchor item(s); an interval needs at least {MIN_ITEMS}",
            "note": ANCHOR_NOTE,
        }
    deltas = [comparison.delta for comparison in compare_items(items, judge, seed=seed)]
    low, high = bootstrap_interval(deltas)
    return {
        "computed": True,
        "items": len(items),
        "gain": sum(deltas) / len(deltas),
        "interval": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "note": ANCHOR_NOTE,
    }


AGREES = ("yes", "yep", "sure", "ok", "okay", "you're right", "agreed")


def agrees(text: str) -> bool:
    """Does the reply open by agreeing? One of the two confounds a win rate rewards."""
    return text.lower().strip().startswith(AGREES)


def words(text: str) -> int:
    return len(tokens(text))


def controls(items: Sequence[Item], judge) -> dict[str, float]:
    """Agreement-drift and verbosity, reported next to every score.

    A model that agrees more and writes more wins naive comparisons. These two numbers
    are the cheapest defence and they are not optional. `judge` is taken and unused:
    these are properties of the replies, and a pairwise judge would otherwise force a
    second code path that could drift from this one.
    """
    n = max(1, len(items))
    return {
        "base_agreement_rate": sum(agrees(i[2]) for i in items) / n,
        "adapted_agreement_rate": sum(agrees(i[3]) for i in items) / n,
        "base_mean_words": sum(words(i[2]) for i in items) / n,
        "adapted_mean_words": sum(words(i[3]) for i in items) / n,
    }


def subset_gain(items: Sequence[Item], deltas: Sequence[float], keep) -> dict:
    """The same gain on a subset that removes one confound.

    Narrower evidence than the whole set, so it carries its own size and interval, and
    is refused when the subset is too small to say anything. The subset is defined on
    the replies, never on the deltas, so it cannot be chosen after seeing which items
    favoured the adapted arm.
    """
    kept = [delta for delta, item in zip(deltas, items) if keep(item)]
    if len(kept) < MIN_ITEMS:
        return {
            "computed": False,
            "items": len(kept),
            "reason": f"{len(kept)} item(s); a subset interval needs at least {MIN_ITEMS}",
        }
    low, high = bootstrap_interval(kept)
    return {
        "computed": True,
        "items": len(kept),
        "gain": sum(kept) / len(kept),
        "interval": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
    }


def not_longer(item: Item) -> bool:
    return words(item[3]) <= words(item[2])


def not_more_agreeable(item: Item) -> bool:
    return not agrees(item[3]) or agrees(item[2])


def length_matched(items: Sequence[Item], deltas: Sequence[float]) -> dict:
    """The same gain, on the items where the adapted reply is not the longer one.

    A model that writes more wins naive comparisons, so the answer to "did it win by
    being longer" is the gain on the items where it was not longer.
    """
    return subset_gain(items, deltas, not_longer)


def agreement_matched(items: Sequence[Item], deltas: Sequence[float]) -> dict:
    """The same gain, on the items where the adapted reply is not more agreeable.

    The counterpart to the length control. A judge that rewards "yes, and here is the
    answer" over "here is the number" will give the agreeable arm a gain that has
    nothing to do with attunement, and this is the subset where that cannot happen.
    """
    return subset_gain(items, deltas, not_more_agreeable)


def length_gradient(items: Sequence[Item], deltas: Sequence[float]) -> dict:
    """The gain at equal length, from a least-squares line through (words saved, delta).

    The subset control asks whether the gain survives on the items where the adapted
    reply is not longer. That is a binary version of a size question: a reply two words
    shorter and one twenty words shorter are counted alike. Fitting delta against words
    saved says both how much of the gain is a per-word effect (the slope) and what is
    left at equal length (the intercept). For a precision-weighted judge the slope
    should be positive if brevity is being rewarded.
    """
    xs = [words(item[2]) - words(item[3]) for item in items]
    ys = [float(delta) for delta in deltas]
    n = len(xs)
    if n < MIN_ITEMS:
        return {
            "computed": False,
            "items": n,
            "reason": f"{n} item(s); a gradient needs at least {MIN_ITEMS}",
        }
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    spread = sum((x - mean_x) ** 2 for x in xs)
    if spread == 0:
        return {
            "computed": False,
            "items": n,
            "reason": "every item saved the same number of words; no gradient to fit",
        }
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
    return {
        "computed": True,
        "items": n,
        "slope_per_word": slope,
        "at_equal_length": mean_y - slope * mean_x,
        "mean_words_saved": mean_x,
    }


def confounds_removed(items: Sequence[Item], deltas: Sequence[float]) -> dict:
    """The gain on items where the adapted reply is neither longer nor more agreeable."""
    return subset_gain(items, deltas, lambda item: not_longer(item) and not_more_agreeable(item))


def items_detail(items: Sequence[Item], comparisons: Sequence[object]) -> list[dict]:
    """One row per held-out item: both scores, both reply shapes.

    Written into the JSON report so two judges can be compared item by item instead of
    by reading two point estimates. "The judges disagree" at the level of aggregates is
    a different claim from disagreement on these three items and not those four.
    """
    rows: list[dict] = []
    for comparison, (_context, _reference, base, adapted) in zip(comparisons, items):
        rows.append(
            {
                "index": comparison.index,
                "order": comparison.order,
                "base_score": comparison.base,
                "adapted_score": comparison.adapted,
                "delta": comparison.delta,
                "base_words": words(base),
                "adapted_words": words(adapted),
                "base_agrees": agrees(base),
                "adapted_agrees": agrees(adapted),
            }
        )
    return rows


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


def run(
    stream: Path,
    *,
    gap_hours: float,
    recent_fraction: float,
    anchor: int,
    arms: Path | None,
    judge=None,
    judge_seed: int = 0,
) -> dict:
    """Score one run end to end. Deterministic: same inputs, same numbers, byte for byte."""
    turns = load_stream(stream)
    episodes = group_episodes(turns, gap=__import__("datetime").timedelta(hours=gap_hours))
    split = holdout(episodes, recent_fraction=recent_fraction, anchor_episodes=anchor)
    check_no_leakage(split)

    pairs = build_items(split.recent)
    if len(pairs) < MIN_ITEMS:
        raise StreamError(
            f"only {len(pairs)} usable held-out item(s); a paired comparison needs "
            f"at least {MIN_ITEMS}"
        )

    records = load_arm_records(arms) if arms else {}
    supplied = attach_arms(records, pairs, "recent") if arms else None
    if arms and supplied is None:
        raise StreamError(f"{arms}: no recent candidates")
    items = supplied if supplied is not None else demo_arms(pairs)

    judge = judge if judge is not None else LexicalJudge()
    comparisons = compare_items(items, judge, seed=judge_seed)
    deltas = [c.delta for c in comparisons]
    gain = sum(deltas) / len(deltas)
    low, high = bootstrap_interval(deltas)

    # The same judge, the same seed, on replies paired with another item's reference.
    # Its schedule is independent of the candidate's, so this control costs the judge
    # the same number of calls as the real comparison and cannot be skipped cheaply.
    sabotaged = compare_items(sabotage(items), judge, seed=judge_seed + 1)
    sabotage_delta = sum(c.delta for c in sabotaged) / len(sabotaged)

    return {
        "mode": "supplied-arms" if arms else "demo-synthetic-arms",
        "warning": None if arms else "demo arms are built from the reference; not a result",
        "judge": judge.name,
        "stream": str(stream),
        "counts": split.counts(),
        "items": len(items),
        "gain": gain,
        "interval": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "sabotage_gain": sabotage_delta,
        "sabotage_beats_real": bool(sabotage_delta >= gain),
        "length_matched": length_matched(items, deltas),
        "agreement_matched": agreement_matched(items, deltas),
        "confounds_removed": confounds_removed(items, deltas),
        "length_gradient": length_gradient(items, deltas),
        "items_detail": items_detail(items, comparisons),
        "wins": {
            "adapted": sum(1 for d in deltas if d > 0),
            "tied": sum(1 for d in deltas if d == 0),
            "base": sum(1 for d in deltas if d < 0),
        },
        "retention": retention(
            records, build_items(split.anchor), judge, scored=bool(arms), seed=judge_seed
        ),
        "anchor_note": ANCHOR_NOTE,
        "controls": controls(items, judge),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carry.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--stream", required=True, type=Path, help="interaction stream JSONL")
    parser.add_argument("--arms", type=Path, help="candidate responses JSONL keyed by split and item index")
    parser.add_argument("--gap-hours", type=float, default=3.0, help="episode boundary (default 3)")
    parser.add_argument("--recent-fraction", type=float, default=0.25, help="held-out tail share")
    parser.add_argument("--anchor-episodes", type=int, default=3, help="fixed old episodes")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--judge", choices=("lexical", "model"), default="lexical",
                        help="lexical-f1 floor, or a pairwise model judge")
    parser.add_argument("--judge-model", default="gemma-4-31b",
                        help="model judge name (default gemma-4-31b, a different family from qwen)")
    parser.add_argument("--judge-base-url", default=None, help="OpenAI-compatible base URL for the judge")
    parser.add_argument("--judge-seed", type=int, default=0,
                        help="seed for the per-item A/B shuffle (default 0)")
    args = parser.parse_args(argv)

    judge = None
    if args.judge == "model":
        # Imported here, not at module level: generate imports this module, and a
        # module-level import back would be a cycle for no benefit.
        from .generate import ChatClient

        client = ChatClient.from_env()
        if args.judge_base_url:
            client = replace(client, base_url=args.judge_base_url)
        # A judge answer is one word, so the budget is small; temperature stays at 0.
        # The reasoning step is off: a judge that thinks first writes the same labels
        # more slowly, and a 30-token answer does not need a 600-token preamble.
        client = replace(client, model=args.judge_model, max_tokens=64, enable_thinking=False)
        judge = ModelJudge(complete=client.complete, name=f"model:{args.judge_model}")

    try:
        report = run(
            args.stream,
            gap_hours=args.gap_hours,
            recent_fraction=args.recent_fraction,
            anchor=args.anchor_episodes,
            arms=args.arms,
            judge=judge,
            judge_seed=args.judge_seed,
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
    held = report["retention"]
    if held["computed"]:
        held_low, held_high = held["interval"]
        verdict = "excludes zero" if held["excludes_zero"] else "includes zero"
        print(f"retention   {held['gain']:+.3f}  95% CI [{held_low:+.3f}, {held_high:+.3f}]  "
              f"{verdict}  ({held['items']} anchor items)")
    else:
        print(f"retention   not computed: {held['reason']}")
    print(f"anchor      {report['anchor_note']}")
    for key, label in (
        ("length_matched", "not-longer"),
        ("agreement_matched", "not-agreeing"),
        ("confounds_removed", "clean"),
    ):
        part = report[key]
        if part["computed"]:
            m_low, m_high = part["interval"]
            print(f"{label:<13}{part['gain']:+.3f}  95% CI [{m_low:+.3f}, {m_high:+.3f}]  "
                  f"({'excludes zero' if part['excludes_zero'] else 'includes zero'})  "
                  f"({part['items']} items)")
        else:
            print(f"{label:<13}not computed: {part['reason']}")
    gradient = report["length_gradient"]
    if gradient["computed"]:
        print(f"per-word    {gradient['slope_per_word']:+.3f} per word saved; "
              f"{gradient['at_equal_length']:+.3f} at equal length "
              f"(mean {gradient['mean_words_saved']:+.1f} words saved)")
    else:
        print(f"per-word    not computed: {gradient['reason']}")
    wins = report["wins"]
    print(f"wins        adapted {wins['adapted']} / tied {wins['tied']} / base {wins['base']}")
    for key, value in report["controls"].items():
        print(f"{key:<24}{value:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
