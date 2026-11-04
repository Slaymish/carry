"""Generate candidate responses for three arms from an interaction stream.

    CARRY_MODEL=qwen3.8-27b python3 -m carry.generate \
        --stream data/sample.jsonl --out arms.jsonl --arms-dir arms/

The endpoint is any OpenAI-compatible chat completions server on the same machine:
base URL from `CARRY_BASE_URL` (default `http://127.0.0.1:8080/v1`), model from
`CARRY_MODEL` (default `qwen3.8-27b`), temperature 0. Standard library only, so the
generation path does not need a client library installed on the box.

Three arms, all answering the same held-out item from their own system prompt:

- `frozen`    the base model, no memory. The baseline everything else is against.
- `ledger`    the base model with a short block of the user's own earlier sentences,
              quoted verbatim from the training episodes. This is the file layer, and
              it is what an adapted model has to beat.
- `sabotaged` the same block built from the same sentences shuffled. Same length, same
              vocabulary, wrong pairing. If this scores like the ledger, the judge is
              measuring tone rather than attunement.

Shuffling is the weakest useful sabotage: the sentences stay true, so a model given the
shuffled block is still told things about the user, just in the wrong order. That is
enough to break the pairing with the moment being answered and not enough to make the
block false. A first live run on the sample showed exactly that, so treat this arm as a
floor and pair it with a foreign-content control before drawing a conclusion from it.

One generation path only: `ChatClient.complete` returns the reply text and nothing else,
which is exactly what the harness scores. Output is JSONL, one row per held-out item:

    {"split": "recent", "index": 0, "frozen": "...", "ledger": "...", "sabotaged": "..."}

`--arms-dir` also writes files in the harness's own format, one per comparison:

    {"split": "recent", "index": 0, "base": "...", "adapted": "..."}

Nothing here touches the repository's data. Generated output is a candidate file, not a
fixture, and is not committed unless it was produced from a committed synthetic sample.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .evaluate import build_items
from .split import Split, holdout
from .stream import Episode, StreamError, group_episodes, load_stream

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_TOKENS = 2048

ARMS = ("frozen", "ledger", "sabotaged")
WHICH = ("recent", "anchor")

SYSTEM_PROMPT = (
    "You are a personal agent replying to its user. Answer the latest message. "
    "Do not restate what the user just said. Two or three sentences."
)
MEMORY_HEADER = "What you already know from earlier conversations with this user:"


class GenerateError(RuntimeError):
    """Generation did not produce something the harness can score. Never patched over."""


# ---------------------------------------------------------------- the HTTP client


def _reply_text(body: object) -> str:
    """The one field the harness wants, taken from an OpenAI-shaped response."""
    if not isinstance(body, dict):
        raise GenerateError(f"response is not an object: {type(body).__name__}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise GenerateError("response has no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise GenerateError("response has no message content")
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        # Diagnosed rather than reported as a blank: a reasoning model that spends the
        # whole budget thinking returns exactly this, and an empty candidate would be
        # scored as a bad answer rather than as a broken call.
        reason = first.get("finish_reason")
        hint = f" (finish_reason={reason})" if reason else ""
        if reason == "length":
            hint += "; the token budget ran out before any reply was written"
        if isinstance(message, dict) and message.get("reasoning_content"):
            hint += "; the model wrote reasoning_content first, so raise max_tokens"
        raise GenerateError(f"response has no message content{hint}")
    return content.strip()


def _post_json(url: str, payload: dict, *, timeout: float) -> dict:
    """POST one JSON body. The only place in this package that touches a socket."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise GenerateError(f"{url}: HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GenerateError(f"{url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GenerateError(f"{url}: response is not JSON ({exc.msg})") from exc


@dataclass(frozen=True, slots=True)
class ChatClient:
    """A chat completions endpoint, reached with urllib and nothing else.

    `transport` exists so tests can stub the socket out entirely: it takes the URL and
    the exact payload that would have been sent, and returns the parsed body. Every
    test in this suite runs with no network.
    """

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    enable_thinking: bool = True
    timeout: float = DEFAULT_TIMEOUT
    transport: Callable[[str, dict], dict] | None = None

    @classmethod
    def from_env(cls) -> "ChatClient":
        return cls(
            base_url=os.environ.get("CARRY_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("CARRY_MODEL", DEFAULT_MODEL),
            timeout=float(os.environ.get("CARRY_TIMEOUT", DEFAULT_TIMEOUT)),
        )

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def payload(self, messages: Sequence[dict]) -> dict:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if not self.enable_thinking:
            # The Qwen chat template accepts this. Off is not a free win: it trades
            # quality for wall-clock, and it is set for every arm at once so no arm
            # gets the slower, better path. A run that turns it off says so.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def complete(self, messages: Sequence[dict]) -> str:
        """One generation, one string. No retries: a failure is reported, not hidden."""
        payload = self.payload(messages)
        if self.transport is not None:
            body = self.transport(self.url, payload)
        else:
            body = _post_json(self.url, payload, timeout=self.timeout)
        return _reply_text(body)


# ---------------------------------------------------------------- the ledger arm

FIRST_PERSON = ("i ", "i'", "my ", "we ", "our ")
PREFERENCE_MARKERS = (
    "can you",
    "please",
    "stop ",
    "don't",
    "do not",
    "instead",
    "from now on",
    "prefer",
    "rather",
    "always",
    "never",
    "remember",
)


def is_ledger_line(text: str) -> bool:
    """A sentence worth quoting back: the user's own, and about the user.

    Two tests, both cheap and both checkable by eye: at least four words, and either
    first person or an instruction. Nothing is inferred or summarised. A ledger of
    paraphrases would be our reading of the conversation, which is the thing this
    project is trying not to measure.
    """
    words = text.split()
    if len(words) < 4:
        return False
    low = " ".join(words).lower()
    return low.startswith(FIRST_PERSON) or " i " in f" {low} " or any(m in low for m in PREFERENCE_MARKERS)


def ledger_lines(episodes: Iterable[Episode]) -> list[str]:
    """The quotable user sentences, chronological, deduplicated."""
    seen: set[str] = set()
    lines: list[str] = []
    for episode in episodes:
        for turn in episode.user_turns():
            text = " ".join(turn.text.split())
            if not is_ledger_line(text) or text in seen:
                continue
            seen.add(text)
            lines.append(text)
    return lines


def _format(lines: Sequence[str], *, max_lines: int, max_chars: int) -> str:
    if not lines:
        return ""
    body = "\n".join(f"- {line[:max_chars]}" for line in lines[:max_lines])
    return f"{MEMORY_HEADER}\n{body}"


def build_ledger(episodes: Iterable[Episode], *, max_lines: int = 6, max_chars: int = 200) -> str:
    """The memory block: the newest quotable sentences, verbatim, newest first."""
    lines = ledger_lines(episodes)
    return _format(list(reversed(lines)), max_lines=max_lines, max_chars=max_chars)


def build_sabotaged_ledger(
    episodes: Iterable[Episode],
    *,
    seed: int = 0,
    max_lines: int = 6,
    max_chars: int = 200,
) -> str:
    """The same block, from the same sentences, shuffled.

    Scrambling the order is the cheapest way to break the pairing between a fact and
    the moment it belonged to while leaving length, vocabulary and register identical.
    A model given this block sounds just as informed; it is informed about nothing in
    particular. Deterministic given the seed, because a control that moves between runs
    is not a control.

    The selection is the real ledger's selection, newest first and capped at
    `max_lines`, and only the order inside it is scrambled. Shuffling the whole pool
    instead would swap sentences in and out as well, so the two arms would differ in
    content and order at once and no difference between them could be attributed to
    either. That was the bug in the first live run: 4 of 8 lines differed.
    """
    chosen = list(reversed(ledger_lines(episodes)))[:max_lines]
    random.Random(seed).shuffle(chosen)
    return _format(chosen, max_lines=max_lines, max_chars=max_chars)


# ---------------------------------------------------------------- items and arms


@dataclass(frozen=True, slots=True)
class Example:
    """One held-out moment: what was available, and what happened."""

    split: str
    index: int
    context: str
    reference: str


@dataclass(frozen=True, slots=True)
class Candidate:
    """The three arms' answers to one Example."""

    split: str
    index: int
    frozen: str
    ledger: str
    sabotaged: str

    def arm(self, name: str) -> str:
        if name not in ARMS:
            raise GenerateError(f"unknown arm {name!r}; expected one of {ARMS}")
        return getattr(self, name)

    def to_json(self) -> dict[str, object]:
        return {
            "split": self.split,
            "index": self.index,
            "frozen": self.frozen,
            "ledger": self.ledger,
            "sabotaged": self.sabotaged,
        }

    def arms_json(self, adapted: str) -> dict[str, object]:
        return {"split": self.split, "index": self.index, "base": self.frozen, "adapted": self.arm(adapted)}


def parse_context(context: str) -> list[dict[str, str]]:
    """Turn the harness's flat context back into chat messages.

    `evaluate.build_items` joins turns as `role: text`. A line that does not begin with
    a known role raises rather than being guessed at.
    """
    messages: list[dict[str, str]] = []
    for line in context.splitlines():
        role, separator, text = line.partition(": ")
        if not separator or role not in ("user", "assistant"):
            raise GenerateError(f"context line is not 'role: text': {line[:60]!r}")
        messages.append({"role": role, "content": text})
    if not messages:
        raise GenerateError("empty context: nothing to reply to")
    return messages


def build_examples(split: Split, *, which: Sequence[str] = WHICH) -> list[Example]:
    """The held-out items to generate for, recent and anchor, in index order per set."""
    examples: list[Example] = []
    for name in which:
        if name not in WHICH:
            raise GenerateError(f"unknown split {name!r}; expected one of {WHICH}")
        pairs = build_items(getattr(split, name))
        examples.extend(
            Example(split=name, index=index, context=context, reference=reference)
            for index, (context, reference) in enumerate(pairs)
        )
    return examples


def messages_for(arm: str, example: Example, *, ledger: str, sabotaged: str) -> list[dict[str, str]]:
    """One system prompt per arm, then the same conversation for all three."""
    if arm == "frozen":
        system = SYSTEM_PROMPT
    elif arm in ("ledger", "sabotaged"):
        block = ledger if arm == "ledger" else sabotaged
        if not block:
            raise GenerateError(
                "no memory block could be derived from the training episodes; "
                f"the {arm} arm would be identical to frozen"
            )
        system = f"{SYSTEM_PROMPT}\n\n{block}"
    else:
        raise GenerateError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return [{"role": "system", "content": system}, *parse_context(example.context)]


def generate_one(
    client: ChatClient,
    example: Example,
    *,
    ledger: str,
    sabotaged: str,
) -> Candidate:
    """All three arms for one item. Kept separate so a failure names its item."""
    texts: dict[str, str] = {}
    for arm in ARMS:
        try:
            texts[arm] = client.complete(messages_for(arm, example, ledger=ledger, sabotaged=sabotaged))
        except GenerateError as exc:
            raise GenerateError(f"{example.split}[{example.index}] {arm}: {exc}") from exc
    return Candidate(split=example.split, index=example.index, **texts)


def generate(
    client: ChatClient,
    examples: Sequence[Example],
    *,
    ledger: str,
    sabotaged: str,
    progress: Callable[[str], None] | None = None,
) -> list[Candidate]:
    out: list[Candidate] = []
    for number, example in enumerate(examples, start=1):
        out.append(generate_one(client, example, ledger=ledger, sabotaged=sabotaged))
        if progress:
            progress(f"{number}/{len(examples)} {example.split}[{example.index}]")
    return out


# ---------------------------------------------------------------- output


def write_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(candidate.to_json(), ensure_ascii=False) for candidate in candidates]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_pair(
    directory: Path,
    candidates: Sequence[Candidate],
    *,
    base_arm: str,
    adapted_arm: str,
) -> Path:
    """Write one harness-ready comparison file between any two arms.

    Any pair, not just frozen-vs-something: `ledger - sabotaged` is the sabotage gate
    as a paired interval, and it is written by the same code as the headline number so
    the two cannot disagree about item order or indices.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{adapted_arm}-vs-{base_arm}.jsonl"
    lines = [
        json.dumps(
            {
                "split": candidate.split,
                "index": candidate.index,
                "base": candidate.arm(base_arm),
                "adapted": candidate.arm(adapted_arm),
            },
            ensure_ascii=False,
        )
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_arms(directory: Path, candidates: Sequence[Candidate], *, adapted: str) -> Path:
    """Write one harness-ready comparison file: frozen as base, `adapted` as adapted."""
    return write_pair(directory, candidates, base_arm="frozen", adapted_arm=adapted)


# ---------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="carry.generate",
        description="generate frozen / ledger / sabotaged candidates for held-out items",
    )
    parser.add_argument("--stream", required=True, type=Path, help="interaction stream JSONL")
    parser.add_argument("--out", required=True, type=Path, help="candidate JSONL to write")
    parser.add_argument("--arms-dir", type=Path, help="also write harness-format arms files here")
    parser.add_argument("--model", default=None, help=f"model name (default CARRY_MODEL, else {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--timeout", type=float, default=None, help=f"seconds per request (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--max-tokens", type=int, default=None, help=f"reply budget (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="turn off the chat template's reasoning step for every arm")
    parser.add_argument("--gap-hours", type=float, default=3.0, help="episode boundary (default 3)")
    parser.add_argument("--recent-fraction", type=float, default=0.25, help="held-out tail share")
    parser.add_argument("--anchor-episodes", type=int, default=3, help="fixed old episodes")
    parser.add_argument("--max-lines", type=int, default=6, help="lines in the memory block")
    parser.add_argument("--sabotage-seed", type=int, default=0, help="shuffle seed for the bad block")
    parser.add_argument("--which", choices=(*WHICH, "both"), default="both", help="held-out sets to generate")
    parser.add_argument("--limit", type=int, default=0, help="stop after N items (0 = all)")
    parser.add_argument("--print-prompts", action="store_true", help="print one item's prompts and exit, no network")
    args = parser.parse_args(argv)

    client = ChatClient.from_env()
    if args.model:
        client = replace(client, model=args.model)
    if args.base_url:
        client = replace(client, base_url=args.base_url)
    if args.timeout:
        client = replace(client, timeout=args.timeout)
    if args.max_tokens:
        client = replace(client, max_tokens=args.max_tokens)
    if args.disable_thinking:
        client = replace(client, enable_thinking=False)

    try:
        episodes = group_episodes(load_stream(args.stream), gap=timedelta(hours=args.gap_hours))
        split = holdout(episodes, recent_fraction=args.recent_fraction, anchor_episodes=args.anchor_episodes)
        which = WHICH if args.which == "both" else (args.which,)
        examples = build_examples(split, which=which)
        if args.limit:
            examples = examples[: args.limit]
        if not examples:
            raise GenerateError("no held-out items to generate for")

        ledger = build_ledger(split.train, max_lines=args.max_lines)
        sabotaged = build_sabotaged_ledger(split.train, seed=args.sabotage_seed, max_lines=args.max_lines)
        if not ledger or not sabotaged:
            raise GenerateError("no memory block could be derived from the training episodes")

        if args.print_prompts:
            for arm in ARMS:
                messages = messages_for(arm, examples[0], ledger=ledger, sabotaged=sabotaged)
                print(json.dumps({"arm": arm, "model": client.model, "messages": messages}, indent=2))
            return 0

        candidates = generate(
            client,
            examples,
            ledger=ledger,
            sabotaged=sabotaged,
            progress=lambda note: print(f"  {note}", file=sys.stderr),
        )
        write_candidates(args.out, candidates)
        print(f"wrote {len(candidates)} item(s) to {args.out}", file=sys.stderr)
        if args.arms_dir:
            for adapted in ("ledger", "sabotaged"):
                path = write_arms(args.arms_dir, candidates, adapted=adapted)
                print(f"wrote {path}", file=sys.stderr)
            path = write_pair(args.arms_dir, candidates, base_arm="sabotaged", adapted_arm="ledger")
            print(f"wrote {path}", file=sys.stderr)
    except (StreamError, GenerateError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
