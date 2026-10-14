import random
import unittest

from src.generators import HistoryIndex, NoEligibleAnomalyError, generate_anomaly
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


if __name__ == "__main__":
    unittest.main()
