import random
import unittest

import numpy as np
import pandas as pd

from src.final_pipeline import (LABEL, activity_error, fit_score, generate_condition,
                                make_schedule, split_boundaries, split_name,
                                validation_threshold)
from src.generators._common import HistoryIndex, edge_key
from src.strict_features import WindowGraph, strict_features
from src.features import FEATURE_MAIN, official_features


class StrictFeaturesTests(unittest.TestCase):
    def test_current_and_simultaneous_events_cannot_change_features(self):
        prefix = [(1, 1, 2, 0), (2, 2, 3, 0)]
        one = strict_features(prefix + [(3, 1, 3, 1)], (2,)).iloc[-1]
        together = strict_features(prefix + [(3, 4, 5, 1), (3, 1, 3, 1)], (2,)).iloc[-1]
        feature_cols = [c for c in one.index if c.startswith('H2_')]
        pd.testing.assert_series_equal(one[feature_cols], together[feature_cols], check_names=False)
        self.assertEqual(one.H2_u_v_weight, 0)
        self.assertEqual(one.H2_total_weight, 2)
        # Labels are not graph inputs, and future events cannot alter the prefix.
        other = strict_features(prefix + [(3, 1, 3, 0), (4, 1, 3, 1)], (2,)).iloc[2]
        pd.testing.assert_series_equal(one[feature_cols], other[feature_cols], check_names=False)

    def test_unseen_endpoints_and_empty_history(self):
        frame = strict_features([(1, 1, 2, 0), (2, 3, 4, 1)], (5,))
        self.assertEqual(frame.iloc[0].H5_number_of_nodes, 0)
        self.assertEqual(frame.iloc[1].H5_u_degree, 0)
        self.assertEqual(frame.iloc[1].H5_u_greater_degree, 2)
        self.assertFalse(frame.isna().any().any())

    @unittest.skipUnless(FEATURE_MAIN.is_file(), "optional pinned TGF checkout unavailable")
    def test_feature_formulas_match_upstream_on_identical_history(self):
        # Upstream includes the current edge. Compare formulas on that same graph;
        # the separate timing tests verify that our actual scoring excludes it.
        rng = random.Random(23)
        rows = [(t, *rng.sample(range(8), 2), t % 2) for t in range(30)]
        upstream, _ = official_features(rows, (5,))
        graph = WindowGraph(5)
        for i, row in enumerate(rows):
            graph.add(row)
            if i in upstream.index:
                _, u, v, _ = row
                actual = graph.score(u, v, graph.snapshot())
                for name, value in actual.items():
                    self.assertEqual(value, upstream.loc[i, 'H5_' + name], (i, name))

    def test_even_distributions_use_upstream_lower_median(self):
        graph = WindowGraph(5)
        for row in [(0,1,2), (1,1,3), (2,2,4)]:
            graph.add(row)
        snapshot, _ = graph.snapshot()
        # Endpoint activity/degree values are [1,1,2,2], so the lower median is 1.
        self.assertEqual(snapshot['weighted_degrees_median'], 1)
        self.assertEqual(snapshot['degrees_median'], 1)

    def test_sliding_graph_matches_bruteforce(self):
        rng = random.Random(19)
        graph = WindowGraph(13)
        stream = []
        for t in range(100):
            u, v = rng.sample(range(8), 2)
            stream.append((t, u, v, 0))
            graph.add(stream[-1])
            recent = stream[-13:]
            neighbors, activity, weights = {}, {}, {}
            for _, a, b, _ in recent:
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)
                activity[a] = activity.get(a, 0) + 1
                activity[b] = activity.get(b, 0) + 1
                pair = tuple(sorted((a, b)))
                weights[pair] = weights.get(pair, 0) + 1
            self.assertEqual(dict(graph.degree), {n: len(xs) for n, xs in neighbors.items()})
            self.assertEqual(dict(graph.activity), activity)
            self.assertEqual(dict(graph.weights), weights)
            scored = graph.score(u, v, graph.snapshot())
            self.assertEqual(scored['total_weight'], len(recent))
            self.assertEqual(scored['u_v_weight'], weights[tuple(sorted((u, v)))])


class ProtocolTests(unittest.TestCase):
    def test_split_keeps_timestamp_ties_together(self):
        events = [(t//3, t, t+1) for t in range(100)]
        bounds = split_boundaries(events)
        for t in set(e[0] for e in events):
            self.assertEqual(len({split_name(e[0], bounds) for e in events if e[0] == t}), 1)

    def test_schedule_paired_exact_and_no_early_injection(self):
        events = [(t, t, t+1) for t in range(100)]
        bounds = split_boundaries(events)
        schedule = make_schedule(events, bounds, 10, 42, warmup=10)
        self.assertEqual(len(schedule), 10)
        self.assertEqual(schedule, make_schedule(events, bounds, 10, 42, warmup=10))
        self.assertGreaterEqual(min(schedule), 10)
        self.assertEqual({split_name(t, bounds) for t in schedule}, {'train', 'validation', 'test'})
        # Selection can run on a prefix containing no test events.
        prefix = [e for e in events if e[0] < bounds[1]]
        self.assertLess(max(make_schedule(prefix, bounds, 10, 42, warmup=10)), bounds[1])

    def test_both_conditions_share_constraints_and_are_reproducible(self):
        events = [(t, t, t+1) for t in range(1, 31)]
        timestamps = [15, 15, 20, 25]
        for strategy in ('random', 'activity_matched'):
            a, audit, _ = generate_condition(events, timestamps, 42, strategy, 10, 2, (18, 24))
            b, other, _ = generate_condition(events, timestamps, 42, strategy, 10, 2, (18, 24))
            self.assertEqual(a, b)
            pd.testing.assert_frame_equal(audit, other)
            self.assertEqual([r[0] for r in a], timestamps)
            seen, pairs = set(), set()
            for t, u, v in events:
                current = [x for x in a if x[0] == t]
                self.assertEqual(len({edge_key(x,y) for _,x,y in current}), len(current))
                for _, x, y in current:
                    self.assertIn(x, seen)
                    self.assertIn(y, seen)
                    self.assertNotIn(edge_key(x,y), pairs)
                    self.assertNotEqual(edge_key(x,y), edge_key(u,v))
                pairs.add(edge_key(u,v))
                pairs.update(edge_key(x,y) for _,x,y in current)
                seen.update((u,v))

    def test_threshold_handles_ties_within_validation_budget(self):
        for n in (5, 100, 1000):
            y = np.zeros(n)
            for scores in (np.ones(n), np.arange(n)/n):
                threshold = validation_threshold(y, scores)
                self.assertLessEqual(np.mean(scores >= threshold), .01)

    def test_test_labels_do_not_select_threshold_or_scores(self):
        train = pd.DataFrame({'x': range(100), LABEL: [0]*50 + [1]*50})
        validation = pd.DataFrame({'x': range(100), LABEL: [0]*50 + [1]*50})
        test = pd.DataFrame({'x': range(100), LABEL: [0,1]*50,
                             'event_id': range(100), 't': range(100)})
        first, prediction = fit_score(train, validation, test, ['x'], 42)
        test[LABEL] = 1-test[LABEL]
        second, changed = fit_score(train, validation, test, ['x'], 42)
        self.assertEqual(first['threshold'], second['threshold'])
        np.testing.assert_array_equal(prediction.score, changed.score)
        self.assertLessEqual(first['actual_validation_fpr'], .01)


if __name__ == '__main__':
    unittest.main()
