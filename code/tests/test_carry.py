"""Tests for the harness. Standard library only.

    cd code && python3 -m unittest discover -s tests -v

The point of most of these is to fail when the harness starts lying: leakage, a
non-deterministic score, or a sabotage control that cannot catch a broken metric.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from carry.evaluate import build_items, demo_arms, run, sabotage
from carry.judge import LexicalJudge, ScriptedJudge, bootstrap_interval, paired_gain
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


if __name__ == "__main__":
    unittest.main()
