"""Generate a fourth arm: the manner lines alone, no memory of any episode.

    python3 data/robustness/manner_arm.py --out data/robustness/regen/manner.jsonl

The ledger block carries two sentences that state the house style the reference replies
were written in ("I prefer two sentences", "the number in the first sentence and the reason
after it") beside six that carry memory. If an arm given only those two scores like the
ledger arm, the gain over frozen is the style being handed to it, not memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carry.evaluate import build_items  # noqa: E402
from carry.generate import ARMS, SYSTEM_PROMPT, ChatClient, build_examples, parse_context  # noqa: E402
from carry.split import holdout  # noqa: E402
from carry.stream import group_episodes, load_stream  # noqa: E402

STYLE_LINES = (
    "- I prefer two sentences; the third one is usually padding.\n"
    "- I'd rather have the number in the first sentence and the reason after it."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stream", type=Path, default=Path("data/fixture-tool-library.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=Path("data/robustness/regen/candidates.jsonl"))
    args = parser.parse_args(argv)

    split = holdout(
        group_episodes(load_stream(args.stream), gap=timedelta(hours=3)),
        recent_fraction=0.25,
        anchor_episodes=8,
    )
    frozen = {}
    for line in args.frozen.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frozen[(row["split"], row["index"])] = row["frozen"]

    client = ChatClient.from_env()
    client = type(client)(
        base_url=client.base_url,
        model=client.model,
        temperature=0.0,
        max_tokens=400,
        enable_thinking=False,
    )
    system = f"{SYSTEM_PROMPT}\n\n{STYLE_LINES}"
    out = []
    for example in build_examples(split, which=("recent",)):
        messages = [{"role": "system", "content": system}, *parse_context(example.context)]
        reply = client.complete(messages)
        out.append(
            {
                "split": example.split,
                "index": example.index,
                "base": frozen[(example.split, example.index)],
                "adapted": reply,
            }
        )
        print(f"{example.index} {reply[:80]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row) + "\n" for row in out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
