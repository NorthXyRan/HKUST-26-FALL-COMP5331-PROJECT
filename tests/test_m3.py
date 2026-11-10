"""Independent behavioral checks for M3 graph, time and data contracts."""

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.m3_history import History, Window, activity_error, interval_burstiness
from src.m3 import read_verified, digest, replay, summarize


class HistoryTests(unittest.TestCase):
    def test_window_matches_bruteforce(self):
        rng = random.Random(5)
        h = Window(37)
        events = []
        for _ in range(300):
            u, v = rng.sample(range(12), 2)
            h.add(u, v)
            events.append((u, v))
            recent = events[-37:]
            a = Counter()
            n = defaultdict(set)
            pairs = Counter()
            for x, y in recent:
                a[x] += 1
                a[y] += 1
                n[x].add(y)
                n[y].add(x)
                pairs[tuple(sorted((x, y)))] += 1
            score = h.score(u, v)
            self.assertEqual([score["activity_low"], score["activity_high"]], sorted([a[u], a[v]]))
            self.assertEqual(score["degree_mean"], (len(n[u]) + len(n[v])) / 2)
            self.assertEqual(score["common_neighbors"], len(n[u] & n[v]))
            self.assertEqual(score["jaccard"], len(n[u] & n[v]) / len(n[u] | n[v]))
            self.assertEqual(score["recent_pair_count"], pairs[tuple(sorted((u, v)))])

    def test_equal_timestamp_queries_do_not_see_current_events(self):
        h = History((2, 1000))
        h.advance(1)
        h.add_batch(1, [(1, 2)])
        h.advance(3)
        before = h.score(3, 1, 3)
        h.score(3, 4, 5)
        self.assertEqual(before[1000]["activity_low"], 0)
        self.assertEqual(before[1000]["activity_high"], 1)
        self.assertEqual(h.score(3, 1, 3)[1000]["activity_high"], 1)
        h.add_batch(3, [(1, 3), (3, 4)])
        with self.assertRaises(ValueError):
            h.advance(3)
        with self.assertRaises(ValueError):
            h.add_batch(3, [(4, 5)])
        h.advance(4)
        self.assertEqual(h.score(4, 1, 3)[1000]["pair_count"], 1)

    def test_time_window_boundary_and_unseen(self):
        h = History()
        h.advance(1)
        h.add_batch(1, [(1, 2)])
        h.advance(3601)
        self.assertEqual(h.score(3601, 1, 2)[1000]["concentration_1h_24h"], 1)
        self.assertTrue(math.isnan(h.score(3601, 8, 9)[1000]["recency_minutes"]))
        h.add_batch(3601, [(1, 3)])
        h.advance(3602)
        self.assertEqual(h.score(3602, 1, 2)[1000]["concentration_1h_24h"], 0.25)
        h.advance(90000)
        self.assertEqual(h.score(90000, 1, 2)[1000]["concentration_1h_24h"], 0)
        with self.assertRaises(ValueError):
            h.advance(89999)

    def test_batch_order_and_history_eviction(self):
        x, y = History((2, 1000)), History((2, 1000))
        for h, pairs in [(x, [(1, 2), (3, 4), (5, 6)]), (y, [(5, 6), (1, 2), (3, 4)])]:
            h.advance(1)
            h.add_batch(1, pairs)
            h.advance(2)
        self.assertEqual(x.windows[2].activity, y.windows[2].activity)
        self.assertEqual(x.score(2, 1, 2)[2]["recent_pair_count"], 0)
        self.assertEqual(x.score(2, 1, 2)[2]["pair_count"], 1)

    def test_burstiness_regular_irregular_and_missing(self):
        self.assertEqual(interval_burstiness([0, 1, 2, 3, 4, 5]), -1)
        self.assertTrue(math.isnan(interval_burstiness([0] * 6)))
        self.assertTrue(math.isnan(interval_burstiness([0, 1, 2])))
        ts = [0, 0, 0, 0, 0, 10]
        gaps = np.diff(ts)
        expected = (gaps.std() - gaps.mean()) / (gaps.std() + gaps.mean())
        self.assertAlmostEqual(interval_burstiness(ts), expected)
        h = History()
        for t in range(30):
            h.advance(t)
            h.add_batch(t, [(1, 2)])
        self.assertEqual(len(h.times[1]), 21)

    def test_activity_alignment_is_undirected(self):
        self.assertEqual(
            activity_error(
                dict(activity_low=1, activity_high=9), dict(activity_low=1, activity_high=9)
            ),
            0,
        )


class PipelineTests(unittest.TestCase):
    def fixture(self):
        stream = pd.DataFrame(
            [(1, 1, 2, 0), (2, 2, 3, 0), (3, 1, 2, 0), (3, 1, 3, 1)],
            columns=["t", "u", "v", "label"],
        )
        normal = dict(activity_low=1, activity_high=2)
        anomaly = dict(activity_low=1, activity_high=1)
        audit = pd.DataFrame(
            [
                dict(
                    t=3,
                    split="test",
                    reference_u=1,
                    reference_v=2,
                    u=1,
                    v=3,
                    activity_error=activity_error(normal, anomaly),
                    reference_u_activity=1,
                    reference_v_activity=2,
                    u_activity=1,
                    v_activity=1,
                )
            ]
        )
        return stream, audit

    def test_replay_aligns_and_rejects_bad_audits(self):
        stream, audit = self.fixture()
        rows = replay(stream, audit, 1, 42, "random", (1000,), (2, 3))
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[rows.kind == "anomaly"].pair_count.eq(0).all())
        for field, value in [("u_activity", 99), ("split", "train"), ("t", 99), ("reference_u", 7)]:
            changed = audit.copy()
            changed[field] = value
            with self.assertRaises(ValueError):
                replay(stream, changed, 1, 42, "random", (1000,), (2, 3))

    def test_tampered_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("a\n1\n")
            manifest = {"files_sha256": {"x.csv": digest(p)}}
            self.assertEqual(read_verified(p, Path(tmp), manifest).iloc[0, 0], 1)
            p.write_text("a\n2\n")
            with self.assertRaises(ValueError):
                read_verified(p, Path(tmp), manifest)

    def test_summary_retains_missingness_and_seed_pairing(self):
        stream, audit = self.fixture()
        frames = [
            replay(stream, audit, 1, s, condition, (1000,), (2, 3))
            for s in (42, 43)
            for condition in ("random", "activity_matched")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summarize(pd.concat(frames), Path(tmp))
            d = pd.read_csv(Path(tmp) / "distributions.csv")
            self.assertTrue(d[d.metric == "burstiness"].missing_fraction.eq(1).all())
            effects = pd.read_csv(Path(tmp) / "paired_effects.csv")
            self.assertEqual(len(effects), 4)
            self.assertTrue(effects.matched_minus_random.eq(0).all())


if __name__ == "__main__":
    unittest.main()
