import random
import unittest

from src.generators import (
    ActivityWindowIndex,
    HistoryIndex,
    NoEligibleAnomalyError,
    generate_anomaly,
    generate_rank_pool_anomalies,
)
from src.generators._common import edge_key


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            (1, "a", "x"), (2, "a", "y"), (3, "a", "z"),
            (4, "c", "x"), (5, "c", "y"), (6, "c", "z"),
            (7, "b", "x"), (8, "d", "y"),
        ]

    def test_random_is_seeded_novel_and_non_loop(self):
        x = generate_anomaly(9, "a", "b", self.history, rng=random.Random(7))
        y = generate_anomaly(9, "a", "b", self.history, rng=random.Random(7))
        self.assertEqual(x, y)
        _, a, b = x
        self.assertNotEqual(a, b)
        self.assertNotIn(edge_key(a, b), {edge_key(s, d) for _, s, d in self.history})

    def test_same_timestamp_and_future_edges_are_invisible(self):
        history = [(1, "a", "b"), (2, "a", "c"), (5, "b", "c"), (8, "d", "e")]
        # At t=5 the b-c edge is not yet visible, so it is a valid anomaly pair.
        _, a, b = generate_anomaly(5, "a", "b", history, rng=random.Random(3))
        self.assertEqual(edge_key(a, b), edge_key("b", "c"))

    def test_activity_matched_strategy_is_available(self):
        event = generate_anomaly(
            9, "a", "b", self.history, "activity_matched",
            rng=random.Random(11), activity_window=100, quantile_bins=2,
        )
        self.assertEqual(event[0], 9)
        self.assertNotEqual(event[1], event[2])
        self.assertNotIn(edge_key(event[1], event[2]), {edge_key(s, d) for _, s, d in self.history})

    def test_incremental_history_index_is_strict_and_seeded(self):
        index = HistoryIndex()
        for event in self.history:
            index.add_batch([event])
        first = generate_anomaly(9, "a", "b", index, rng=random.Random(7))
        second = generate_anomaly(9, "a", "b", index, rng=random.Random(7))
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            generate_anomaly(8, "a", "b", index)

    def test_no_coverage_is_reported_not_random_fallback(self):
        with self.assertRaises(NoEligibleAnomalyError):
            generate_anomaly(2, "a", "b", [(1, "a", "b")], "random", rng=random.Random(0))

    def test_activity_window_index_slides_and_enforces_strict_time(self):
        index = ActivityWindowIndex(window=2)
        index.add_batch([(1, "a", "b"), (1, "a", "c")])
        self.assertEqual(index.activity["a"], 2)
        index.add_batch([(2, "d", "e")])
        self.assertEqual(index.activity["a"], 1)
        self.assertEqual(index.activity["b"], 0)
        self.assertEqual(index.activity["d"], 1)

    def test_rank_pool_generator_returns_requested_unique_novel_edges(self):
        events = [
            (1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5),
            (5, 5, 6), (6, 6, 7), (7, 7, 8), (8, 8, 9),
            (9, 9, 10), (10, 10, 11), (11, 11, 12),
            (12, 1, 3), (13, 2, 4), (14, 5, 7), (15, 8, 10),
        ]
        generated = generate_rank_pool_anomalies(
            events, 2, reference_count=2, window=10, candidates_per_endpoint=10,
            timestamp_seed=5, reference_seed=7,
        )
        self.assertEqual(len(generated), 2)
        self.assertEqual(len({edge_key(u, v) for _, u, v in generated}), 2)
        prior = set()
        by_time = {}
        for t, u, v in events:
            by_time.setdefault(t, []).append(edge_key(u, v))
        generated_by_time = {}
        for t, u, v in generated:
            generated_by_time.setdefault(t, []).append(edge_key(u, v))
        for t in sorted(by_time):
            for pair in generated_by_time.get(t, []):
                self.assertNotIn(pair, prior)
                self.assertNotIn(pair, by_time[t])
            prior.update(by_time[t])


if __name__ == "__main__":
    unittest.main()
