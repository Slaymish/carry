"""Tests for the harness. Standard library only.

    cd code && python3 -m unittest discover -s tests -v

The point of most of these is to fail when the harness starts lying: leakage, a
non-deterministic score, or a sabotage control that cannot catch a broken metric.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from unittest import mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from carry import generate as generate_module
from carry.evaluate import (
    ANCHOR_NOTE,
    build_items,
    compare_items,
    demo_arms,
    length_matched,
    load_arm_records,
    run,
    sabotage,
)
from carry.generate import (
    ChatClient,
    GenerateError,
    build_examples,
    build_ledger,
    build_sabotaged_ledger,
    generate,
    generate_one,
    ledger_lines,
    messages_for,
    parse_context,
    write_arms,
    write_candidates,
    write_pair,
)
from carry.judge import (
    JudgeError,
    LexicalJudge,
    ModelJudge,
    ScriptedJudge,
    bootstrap_interval,
    paired_gain,
    paired_preferences,
    parse_preference,
)
from carry.split import check_no_leakage, holdout
from carry.stream import Episode, StreamError, Turn, group_episodes, iter_turns, load_stream, parse_ts

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.jsonl"
START = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)


def turn(minutes: int, role: str = "user", text: str = "x") -> Turn:
    return Turn(START + timedelta(minutes=minutes), role, text)


def episode(offset_hours: int, count: int = 2) -> Episode:
    base = offset_hours * 60
    return Episode(tuple(turn(base + i) for i in range(count)))


class StreamTests(unittest.TestCase):
    def test_schema_rejects_wrong_role_and_empty_text(self):
        bad_role = io.StringIO('{"ts":"2026-01-01T00:00:00Z","role":"system","text":"x"}\n')
        with self.assertRaises(StreamError):
            list(iter_turns(bad_role))
        bad_text = io.StringIO('{"ts":"2026-01-01T00:00:00Z","role":"user","text":"  "}\n')
        with self.assertRaises(StreamError):
            list(iter_turns(bad_text))

    def test_naive_timestamp_is_utc_not_local(self):
        self.assertEqual(parse_ts("2026-01-01T00:00:00"), datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_gap_splits_sittings(self):
        turns = [turn(0), turn(1), turn(240), turn(241)]
        self.assertEqual(len(group_episodes(turns, gap=timedelta(hours=3))), 2)
        self.assertEqual(len(group_episodes(turns, gap=timedelta(hours=5))), 1)

    def test_sample_stream_loads_sorted_and_non_empty(self):
        turns = load_stream(SAMPLE)
        self.assertGreater(len(turns), 20)
        self.assertEqual(turns, sorted(turns, key=lambda t: t.ts))


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(i * 24) for i in range(10)]

    def test_windows_are_disjoint_and_ordered(self):
        split = holdout(self.episodes, recent_fraction=0.25, anchor_episodes=2)
        check_no_leakage(split)
        self.assertEqual(len(split.anchor), 2)
        self.assertGreaterEqual(len(split.recent), 1)
        self.assertLess(max(e.ended for e in split.train), min(e.started for e in split.recent))

    def test_anchor_is_removed_from_training_not_copied(self):
        split = holdout(self.episodes, anchor_episodes=3)
        train_ids = {id(e) for e in split.train}
        self.assertFalse(train_ids & {id(e) for e in split.anchor})

    def test_too_little_data_fails_loudly(self):
        with self.assertRaises(StreamError):
            holdout(self.episodes[:3], anchor_episodes=2)


class JudgeTests(unittest.TestCase):
    def test_lexical_judge_is_bounded_and_rewards_overlap(self):
        judge = LexicalJudge()
        ref = "the build failed on the asset upload"
        self.assertEqual(judge.score("", ref, ref), 1.0)
        self.assertGreater(judge.score("", ref, "the build failed"), judge.score("", ref, "unrelated text"))
        self.assertGreaterEqual(judge.score("", ref, ""), 0.0)

    def test_paired_gain_uses_identical_reference_for_both_arms(self):
        judge = ScriptedJudge({"good": 1.0, "bad": 0.0})
        results = paired_gain([("ctx", "ref", "bad", "good")], judge)
        self.assertEqual(results[0].base, 0.0)
        self.assertEqual(results[0].adapted, 1.0)
        self.assertEqual(results[0].delta, 1.0)

    def test_unblind_comparison_refuses(self):
        judge = ScriptedJudge({"a": 0.5, "b": 0.5})
        with self.assertRaises(ValueError):
            paired_gain([("ctx", "ref", "a", "b")], judge, blind=lambda _i: True)

    def test_bootstrap_is_deterministic_and_ordered(self):
        deltas = [0.1, 0.2, -0.05, 0.3, 0.0, 0.15]
        first = bootstrap_interval(deltas, seed=7)
        second = bootstrap_interval(deltas, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])


class EvaluateTests(unittest.TestCase):
    def test_items_skip_episodes_without_an_assistant_reply(self):
        no_reply = Episode((turn(0, "user", "a"), turn(1, "user", "b")))
        answered = Episode((turn(60, "user", "c"), turn(61, "assistant", "d")))
        self.assertEqual(len(build_items([no_reply])), 0)
        self.assertEqual(len(build_items([no_reply, answered])), 1)

    def test_sabotage_pairs_an_arm_with_the_next_reference(self):
        items = demo_arms([("user: a", "alpha beta gamma"), ("user: b", "delta epsilon zeta")])
        shifted = sabotage(items)
        # The reference stays with its own context; the candidate is swapped. An item
        # scored against somebody else's answer is the control.
        self.assertEqual(shifted[0][1], "alpha beta gamma")
        self.assertEqual(shifted[0][3], items[1][3])
        self.assertEqual(shifted[1][3], items[0][3])

    def test_full_run_is_reproducible(self):
        kwargs = dict(gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=None)
        first = run(SAMPLE, **kwargs)
        second = run(SAMPLE, **kwargs)
        self.assertEqual(first["gain"], second["gain"])
        self.assertEqual(first["interval"], second["interval"])
        self.assertEqual(first["mode"], "demo-synthetic-arms")
        self.assertIsNotNone(first["warning"])
        self.assertGreater(first["items"], 2)

    def test_supplied_arms_are_attached_in_index_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "s.jsonl"
            stream.write_text(SAMPLE.read_text())
            pairs = build_items(
                holdout(
                    group_episodes(load_stream(stream), gap=timedelta(hours=3)),
                    recent_fraction=0.25,
                    anchor_episodes=3,
                ).recent
            )
            arms = Path(tmp) / "arms.jsonl"
            arms.write_text("\n".join(
                json.dumps({"index": i, "base": "plain", "adapted": ref}) for i, (_c, ref) in enumerate(pairs)
            ) + "\n")
            report = run(stream, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
        self.assertEqual(report["mode"], "supplied-arms")
        self.assertIsNone(report["warning"])
        self.assertEqual(report["items"], len(pairs))
        self.assertGreater(report["gain"], 0.0)


# ---------------------------------------------------------------- retention


class RetentionTests(unittest.TestCase):
    """Retention is the comparison on the anchor set. It may not be invented."""

    def setUp(self):
        self.split = holdout(
            group_episodes(load_stream(SAMPLE), gap=timedelta(hours=3)),
            recent_fraction=0.25,
            anchor_episodes=3,
        )

    def arms_file(self, tmp: str, *, anchor: bool = True, anchor_arm: str | None = None) -> Path:
        rows = []
        for name in ("recent", "anchor"):
            if name == "anchor" and not anchor:
                continue
            for index, (_context, reference) in enumerate(build_items(getattr(self.split, name))):
                adapted = anchor_arm if (name == "anchor" and anchor_arm is not None) else reference
                rows.append({"split": name, "index": index, "base": "plain text", "adapted": adapted})
        path = Path(tmp) / "arms.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_anchor_items_are_scored_with_their_own_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = self.arms_file(tmp, anchor_arm="plain text")
            report = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
        held = report["retention"]
        self.assertTrue(held["computed"])
        self.assertEqual(held["items"], 3)
        self.assertAlmostEqual(report["gain"], 1.0)          # recent: reference copied exactly
        self.assertAlmostEqual(held["gain"], 0.0)            # anchor: the arms agree
        self.assertEqual(held["interval"], [0.0, 0.0])
        self.assertFalse(held["excludes_zero"])
        self.assertEqual(held["note"], ANCHOR_NOTE)

    def test_retention_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = self.arms_file(tmp)
            first = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
            second = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
        self.assertEqual(first["retention"], second["retention"])

    def test_arms_without_anchor_rows_report_no_retention_rather_than_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = self.arms_file(tmp, anchor=False)
            report = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
        held = report["retention"]
        self.assertFalse(held["computed"])
        self.assertNotIn("gain", held)
        self.assertIn("no anchor rows", held["reason"])

    def test_demo_arms_refuse_retention_and_say_why(self):
        report = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=None)
        held = report["retention"]
        self.assertFalse(held["computed"])
        self.assertNotIn("gain", held)
        self.assertEqual(report["anchor_note"], ANCHOR_NOTE)
        self.assertIn("nothing was trained", held["reason"])

    def test_partial_anchor_file_is_an_error_not_a_smaller_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            arms = self.arms_file(tmp)
            rows = [json.loads(line) for line in arms.read_text().splitlines()]
            rows = [row for row in rows if not (row["split"] == "anchor" and row["index"] == 2)]
            arms.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(StreamError):
                run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)

    def test_arms_file_refuses_duplicates_and_unknown_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dup.jsonl"
            path.write_text(
                json.dumps({"split": "recent", "index": 0, "base": "a", "adapted": "b"}) + "\n"
                + json.dumps({"split": "recent", "index": 0, "base": "c", "adapted": "d"}) + "\n"
            )
            with self.assertRaises(StreamError):
                load_arm_records(path)
            path = Path(tmp) / "bad.jsonl"
            path.write_text(json.dumps({"split": "train", "index": 0, "base": "a", "adapted": "b"}) + "\n")
            with self.assertRaises(StreamError):
                load_arm_records(path)

    def test_legacy_arms_file_without_split_still_scores_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.jsonl"
            pairs = build_items(self.split.recent)
            path.write_text("\n".join(
                json.dumps({"index": i, "base": "plain", "adapted": ref})
                for i, (_context, ref) in enumerate(pairs)
            ) + "\n")
            report = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=path)
        self.assertEqual(report["mode"], "supplied-arms")
        self.assertAlmostEqual(report["gain"], 1.0)
        self.assertFalse(report["retention"]["computed"])


# ---------------------------------------------------------------- generation


def stub_transport(*, marker: bool = True):
    """A transport that never touches a socket and answers deterministically."""
    calls: list[dict] = []

    def transport(url: str, payload: dict) -> dict:
        calls.append(payload)
        system = payload["messages"][0]["content"]
        remembered = "What you already know" in system
        content = f"memory={remembered} system_words={len(system.split())}"
        return {"choices": [{"message": {"content": content}}]}

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.split = holdout(
            group_episodes(load_stream(SAMPLE), gap=timedelta(hours=3)),
            recent_fraction=0.25,
            anchor_episodes=3,
        )
        self.ledger = build_ledger(self.split.train)
        self.sabotaged = build_sabotaged_ledger(self.split.train, seed=0)
        self.examples = build_examples(self.split)

    def test_defaults_are_the_documented_endpoint_and_model(self):
        client = ChatClient()
        self.assertEqual(client.url, "http://127.0.0.1:8080/v1/chat/completions")
        self.assertEqual(client.model, "qwen3.8-27b")
        self.assertEqual(client.temperature, 0.0)

    def test_environment_overrides_the_endpoint(self):
        with mock.patch.dict(os.environ, {"CARRY_BASE_URL": "http://127.0.0.1:9999/v1/", "CARRY_MODEL": "gemma-4-31b"}):
            client = ChatClient.from_env()
        self.assertEqual(client.url, "http://127.0.0.1:9999/v1/chat/completions")
        self.assertEqual(client.model, "gemma-4-31b")

    def test_payload_is_temperature_zero_and_not_streamed(self):
        transport = stub_transport()
        client = ChatClient(transport=transport)
        client.complete([{"role": "user", "content": "hi"}])
        payload = transport.calls[0]
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "qwen3.8-27b")

    def test_thinking_is_a_payload_flag_and_defaults_on(self):
        transport = stub_transport()
        ChatClient(transport=transport).complete([{"role": "user", "content": "hi"}])
        self.assertNotIn("chat_template_kwargs", transport.calls[0])
        off = stub_transport()
        ChatClient(transport=off, enable_thinking=False).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(off.calls[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(off.calls[0]["temperature"], 0)

    def test_complete_returns_exactly_the_reply_string(self):
        client = ChatClient(transport=lambda url, payload: {"choices": [{"message": {"content": "  fine.  "}}]})
        self.assertEqual(client.complete([{"role": "user", "content": "hi"}]), "fine.")

    def test_a_response_without_content_is_an_error_not_an_empty_string(self):
        for body in ({}, {"choices": []}, {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": " "}}]}):
            with self.subTest(body=body), self.assertRaises(GenerateError):
                ChatClient(transport=lambda url, payload, body=body: body).complete(
                    [{"role": "user", "content": "hi"}]
                )

    def test_a_budget_spent_on_reasoning_is_diagnosed(self):
        body = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "", "reasoning_content": "thinking..."},
                }
            ]
        }
        with self.assertRaises(GenerateError) as caught:
            ChatClient(transport=lambda url, payload: body).complete([{"role": "user", "content": "hi"}])
        message = str(caught.exception)
        self.assertIn("finish_reason=length", message)
        self.assertIn("max_tokens", message)

    def test_every_test_runs_without_network(self):
        transport = stub_transport()
        with mock.patch.object(generate_module, "_post_json", side_effect=AssertionError("socket used")):
            generate(ChatClient(transport=transport), self.examples, ledger=self.ledger, sabotaged=self.sabotaged)
        self.assertEqual(len(transport.calls), len(self.examples) * 3)

    def test_the_three_arms_differ_only_in_the_memory_block(self):
        frozen = messages_for("frozen", self.examples[0], ledger=self.ledger, sabotaged=self.sabotaged)
        ledger = messages_for("ledger", self.examples[0], ledger=self.ledger, sabotaged=self.sabotaged)
        bad = messages_for("sabotaged", self.examples[0], ledger=self.ledger, sabotaged=self.sabotaged)
        self.assertNotIn("What you already know", frozen[0]["content"])
        self.assertIn(self.ledger, ledger[0]["content"])
        self.assertIn(self.sabotaged, bad[0]["content"])
        self.assertNotEqual(ledger[0]["content"], bad[0]["content"])
        self.assertEqual(frozen[1:], ledger[1:])
        self.assertEqual(ledger[1:], bad[1:])

    def test_ledger_lines_are_verbatim_user_turns(self):
        allowed = {
            " ".join(turn.text.split())
            for episode in self.split.train
            for turn in episode.user_turns()
        }
        for line in ledger_lines(self.split.train):
            self.assertIn(line, allowed)
        for line in self.ledger.splitlines()[1:]:
            self.assertIn(line.removeprefix("- "), allowed)

    def test_sabotaged_block_is_deterministic_and_reorders_the_real_one(self):
        again = build_sabotaged_ledger(self.split.train, seed=0)
        self.assertEqual(self.sabotaged, again)
        body = [line for line in self.sabotaged.splitlines()[1:]]
        self.assertTrue(body)
        self.assertEqual(len(body), len(self.ledger.splitlines()[1:]))
        if len(body) > 1:
            self.assertNotEqual(self.sabotaged, self.ledger)

    def test_generation_is_reproducible_byte_for_byte(self):
        transport = stub_transport()
        client = ChatClient(transport=transport)
        first = generate(client, self.examples, ledger=self.ledger, sabotaged=self.sabotaged)
        second = generate(client, self.examples, ledger=self.ledger, sabotaged=self.sabotaged)
        self.assertEqual([c.to_json() for c in first], [c.to_json() for c in second])

    def test_written_files_round_trip_through_the_harness(self):
        transport = stub_transport()
        clients = generate(
            ChatClient(transport=transport), self.examples, ledger=self.ledger, sabotaged=self.sabotaged
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.jsonl"
            write_candidates(out, clients)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), len(self.examples))
            self.assertEqual(set(rows[0]), {"split", "index", "frozen", "ledger", "sabotaged"})

            arms = write_arms(Path(tmp) / "arms", clients, adapted="ledger")
            records = load_arm_records(arms)
            self.assertEqual(len(records), len(self.examples))
            self.assertEqual(records[("recent", 0)][1], clients[0].ledger)
            self.assertEqual(records[("anchor", 0)][0], clients[-3].frozen)

    def test_generated_arms_score_end_to_end(self):
        transport = stub_transport()
        candidates = generate(
            ChatClient(transport=transport), self.examples, ledger=self.ledger, sabotaged=self.sabotaged
        )
        with tempfile.TemporaryDirectory() as tmp:
            arms = write_arms(Path(tmp), candidates, adapted="ledger")
            report = run(SAMPLE, gap_hours=3.0, recent_fraction=0.25, anchor=3, arms=arms)
        self.assertEqual(report["mode"], "supplied-arms")
        self.assertTrue(report["retention"]["computed"])
        self.assertEqual(report["retention"]["items"], 3)

    def test_a_failure_names_the_item_and_the_arm(self):
        def transport(url: str, payload: dict) -> dict:
            return {"choices": []}

        with self.assertRaises(GenerateError) as caught:
            generate_one(
                ChatClient(transport=transport),
                self.examples[0],
                ledger=self.ledger,
                sabotaged=self.sabotaged,
            )
        self.assertIn(f"{self.examples[0].split}[{self.examples[0].index}] frozen", str(caught.exception))

    def test_an_empty_ledger_refuses_rather_than_making_three_identical_arms(self):
        with self.assertRaises(GenerateError):
            messages_for("ledger", self.examples[0], ledger="", sabotaged=self.sabotaged)

    def test_context_round_trips_to_messages(self):
        messages = parse_context("user: hello\nassistant: hi\nuser: again")
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[2]["content"], "again")
        for bad in ("no role prefix", "system: nah", ""):
            with self.subTest(bad=bad), self.assertRaises(GenerateError):
                parse_context(bad)

    def test_print_prompts_runs_the_cli_with_no_network(self):
        with mock.patch.object(generate_module, "_post_json", side_effect=AssertionError("socket used")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "unused.jsonl"
                code = generate_module.main(
                    ["--stream", str(SAMPLE), "--out", str(out), "--print-prompts"]
                )
                self.assertFalse(out.exists())
        self.assertEqual(code, 0)

    def test_cli_reports_a_missing_stream_as_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = generate_module.main(
                ["--stream", str(Path(tmp) / "nope.jsonl"), "--out", str(Path(tmp) / "o.jsonl")]
            )
        self.assertEqual(code, 2)


# ---------------------------------------------------------------- the model judge


class ScriptedPairwise:
    """A pairwise judge with no model in it: prefers whichever text is longer."""

    name = "scripted-pairwise"
    pairwise = True

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def prefer(self, context: str, reference: str, first: str, second: str) -> float:
        self.asked.append((first, second))
        if len(first.split()) == len(second.split()):
            return 0.5
        return 1.0 if len(first.split()) > len(second.split()) else 0.0


class LengthControlTests(unittest.TestCase):
    def test_only_items_where_the_adapted_reply_is_not_longer_are_kept(self):
        items = [
            ("ctx", "ref", "one two three", "one two"),          # shorter: kept
            ("ctx", "ref", "one", "one two three four five"),    # longer: dropped
            ("ctx", "ref", "one two", "one two"),                # equal: kept
            ("ctx", "ref", "one two three", "one"),              # shorter: kept
        ]
        matched = length_matched(items, [0.5, 0.9, 0.1, 0.3])
        self.assertTrue(matched["computed"])
        self.assertEqual(matched["items"], 3)
        self.assertAlmostEqual(matched["gain"], 0.3)

    def test_a_subset_too_small_to_say_anything_is_refused(self):
        matched = length_matched([("ctx", "ref", "a b c", "a")], [0.4])
        self.assertFalse(matched["computed"])
        self.assertNotIn("gain", matched)


class ModelJudgeTests(unittest.TestCase):
    def test_preference_parsing_is_strict(self):
        self.assertEqual(parse_preference("A"), 1.0)
        self.assertEqual(parse_preference(" b."), 0.0)
        self.assertEqual(parse_preference("Tie"), 0.5)
        for bad in ("", "both", "A or B", "candidate one"):
            with self.subTest(bad=bad), self.assertRaises(JudgeError):
                parse_preference(bad)

    def test_the_judge_prompt_never_names_an_arm(self):
        judge = ModelJudge(complete=lambda messages: "A")
        messages = judge.prompt("user: hi", "the reference", "first reply", "second reply")
        text = " ".join(message["content"] for message in messages).lower()
        for arm in ("frozen", "ledger", "sabotaged", "base", "adapted"):
            self.assertNotIn(arm, text)
        self.assertIn("first reply", text)
        self.assertIn("second reply", text)

    def test_a_shuffled_order_is_deterministic_given_a_seed(self):
        items = [("ctx", "ref", f"base {i}", f"adapted {i}") for i in range(12)]
        first = [c.order for c in paired_preferences(items, ScriptedPairwise(), seed=3)]
        second = [c.order for c in paired_preferences(items, ScriptedPairwise(), seed=3)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"base-first", "adapted-first"})

    def test_order_does_not_decide_the_winner(self):
        # A judge that always picks the first candidate must give a zero mean: the
        # shuffle turns a position preference into noise rather than into a gain.
        class AlwaysFirst(ScriptedPairwise):
            def prefer(self, context, reference, first, second):
                return 1.0

        items = [("ctx", "ref", f"base {i}", f"adapted {i}") for i in range(20)]
        results = paired_preferences(items, AlwaysFirst(), seed=1)
        # Every delta is decided by the position the shuffle gave the base arm, and by
        # nothing about the arm itself.
        for comparison in results:
            expected = -1.0 if comparison.order == "base-first" else 1.0
            self.assertEqual(comparison.delta, expected)
        first = sum(1 for c in results if c.order == "base-first")
        self.assertGreater(first, 0)
        self.assertLess(first, len(results))
        self.assertEqual(sum(c.delta for c in results), (len(results) - first) - first)

    def test_both_orderings_map_the_winner_to_the_right_arm(self):
        class PreferAdapted(ScriptedPairwise):
            def prefer(self, context, reference, first, second):
                return 1.0 if first.startswith("adapted") else 0.0

        items = [("ctx", "ref", f"base {i}", f"adapted {i}") for i in range(10)]
        for comparison in paired_preferences(items, PreferAdapted(), seed=5):
            self.assertEqual(comparison.delta, 1.0, comparison)

    def test_compare_items_dispatches_on_the_judge_kind(self):
        items = [("ctx", "ref", "one two", "one two three")]
        pairwise = [c.delta for c in compare_items(items, ScriptedPairwise())]
        scored = [c.delta for c in compare_items(items, ScriptedJudge({"one two": 0.0, "one two three": 1.0}))]
        self.assertEqual(pairwise, [1.0])
        self.assertEqual(scored, [1.0])

    def test_the_harness_scores_a_pairwise_judge_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = holdout(
                group_episodes(load_stream(SAMPLE), gap=timedelta(hours=3)),
                recent_fraction=0.25,
                anchor_episodes=3,
            )
            rows = []
            for name in ("recent", "anchor"):
                for index, (_context, reference) in enumerate(build_items(getattr(split, name))):
                    rows.append(
                        {"split": name, "index": index, "base": "short", "adapted": reference}
                    )
            arms = Path(tmp) / "arms.jsonl"
            arms.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            report = run(
                SAMPLE,
                gap_hours=3.0,
                recent_fraction=0.25,
                anchor=3,
                arms=arms,
                judge=ScriptedPairwise(),
                judge_seed=4,
            )
            again = run(
                SAMPLE,
                gap_hours=3.0,
                recent_fraction=0.25,
                anchor=3,
                arms=arms,
                judge=ScriptedPairwise(),
                judge_seed=4,
            )
        self.assertEqual(report["judge"], "scripted-pairwise")
        self.assertEqual(report["gain"], again["gain"])
        self.assertEqual(report["interval"], again["interval"])
        self.assertEqual(report["wins"], again["wins"])
        self.assertTrue(report["retention"]["computed"])
        self.assertEqual(sum(report["wins"].values()), report["items"])
        self.assertGreater(report["gain"], 0.0)

    def test_write_pair_covers_any_two_arms(self):
        transport = stub_transport()
        split = holdout(
            group_episodes(load_stream(SAMPLE), gap=timedelta(hours=3)),
            recent_fraction=0.25,
            anchor_episodes=3,
        )
        candidates = generate(
            ChatClient(transport=transport),
            build_examples(split),
            ledger=build_ledger(split.train),
            sabotaged=build_sabotaged_ledger(split.train, seed=0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_pair(Path(tmp), candidates, base_arm="sabotaged", adapted_arm="ledger")
            self.assertEqual(path.name, "ledger-vs-sabotaged.jsonl")
            records = load_arm_records(path)
        self.assertEqual(records[("recent", 0)][1], candidates[0].ledger)
        self.assertEqual(records[("recent", 0)][0], candidates[0].sabotaged)


if __name__ == "__main__":
    unittest.main()
