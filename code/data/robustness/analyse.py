"""Judge agreement and position bias, item by item.

    python3 data/robustness/analyse.py lexical-ledger-vs-frozen.json model-ledger-vs-frozen-s0-a.json

Reads two `carry.evaluate --json` reports that scored the same arms file, and prints
the per-item deltas side by side, the correlation between them, and which items the two
judges disagree on. Then reports, for every model-judge report given, how often the
candidate in the first position won: a judge whose verdict depends on where a candidate
was placed is not measuring the candidate.

Standard library only, no network: the reports are the record of what was asked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def ranks(values: list[float]) -> list[float]:
    """Average ranks, so tied deltas do not get an arbitrary order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for slot in range(position, end + 1):
            out[order[slot]] = average
        position = end + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    left = sum((x - mx) ** 2 for x in xs) ** 0.5
    right = sum((y - my) ** 2 for y in ys) ** 0.5
    if left == 0 or right == 0:
        return float("nan")
    return top / (left * right)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def permutation_p(xs: list[float], ys: list[float], *, rounds: int = 20000, seed: int = 0) -> float:
    """How often a shuffled pairing correlates at least this well.

    Ten items is too few for a normal approximation to mean anything, and the sample is
    the whole sample: a permutation is the honest test. Fixed seed, so the p-value is
    the same number on every rerun.
    """
    import random

    observed = abs(pearson(xs, ys))
    rng = random.Random(seed)
    shuffled = list(ys)
    hits = 0
    for _ in range(rounds):
        rng.shuffle(shuffled)
        if abs(pearson(xs, shuffled)) >= observed:
            hits += 1
    return hits / rounds


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    left_path, right_path = Path(argv[1]), Path(argv[2])
    left, right = load(left_path), load(right_path)
    a = {row["index"]: row for row in left["items_detail"]}
    b = {row["index"]: row for row in right["items_detail"]}
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared items")
        return 2

    print(f"{'item':>4}  {'lex':>7}  {'model':>6}  {'order':<14} {'words':>11}  agree")
    print(f"{'':>4}  {'':>7}  {'':>6}  {'':<14} {'b/a':>11}")
    for index in shared:
        row_a, row_b = a[index], b[index]
        flag = "" if sign(row_a["delta"]) == sign(row_b["delta"]) else "  <- disagree"
        print(
            f"{index:>4}  {row_a['delta']:+7.3f}  {row_b['delta']:+6.1f}  {row_b['order']:<14} "
            f"{row_a['base_words']:>5}/{row_a['adapted_words']:<5} "
            f"{'b' if row_b['base_agrees'] else '-'}{'a' if row_b['adapted_agrees'] else '-'}{flag}"
        )

    xs = [a[i]["delta"] for i in shared]
    ys = [b[i]["delta"] for i in shared]
    same_sign = sum(1 for x, y in zip(xs, ys) if sign(x) == sign(y))
    print()
    print(f"items            {len(shared)}")
    print(f"pearson          {pearson(xs, ys):+.3f}")
    print(f"spearman         {spearman(xs, ys):+.3f}")
    print(f"same sign        {same_sign}/{len(shared)}")
    print(f"mean lex         {sum(xs) / len(xs):+.3f}")
    print(f"mean model       {sum(ys) / len(ys):+.3f}")
    print(f"permutation p    {permutation_p(xs, ys):.3f}")

    print()
    print("position bias (first-placed candidate won):")
    for path in (left_path, right_path):
        report = load(path)
        rows = report["items_detail"]
        if "order" not in rows[0]:
            continue
        first_won = 0
        second_won = 0
        tied = 0
        for row in rows:
            if row["delta"] == 0:
                tied += 1
                continue
            adapted_first = row["order"] == "adapted-first"
            adapted_won = row["delta"] > 0
            if adapted_first == adapted_won:
                first_won += 1
            else:
                second_won += 1
        decided = first_won + second_won
        print(
            f"  {path.name:<38} first {first_won:>2} / second {second_won:>2} / tie {tied:>2}"
            f"   ({first_won / decided:.0%} to the first slot)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
