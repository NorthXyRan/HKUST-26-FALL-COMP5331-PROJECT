"""Rank-window candidate-pool anomaly generator with a rolling activity index."""
from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections import Counter, deque
from typing import Iterable, Sequence

from ._common import Event, NoEligibleAnomalyError, edge_key, require_reference_edge


class ActivityWindowIndex:
    """History with O(1)-amortized rolling activity updates and activity buckets."""

    def __init__(self, window: int = 1000):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self.events: deque[Event] = deque()
        self.activity: Counter = Counter()
        self.nodes: set = set()
        self.nodes_by_activity: dict[int, set] = {}
        self.pairs: set[frozenset] = set()
        self.last_timestamp = None

    def _change_activity(self, node, delta: int) -> None:
        old = self.activity[node]
        old_bucket = self.nodes_by_activity.get(old)
        if old_bucket is not None:
            old_bucket.discard(node)
            if not old_bucket:
                del self.nodes_by_activity[old]
        new = old + delta
        if new < 0:
            raise RuntimeError("rolling activity count became negative")
        self.activity[node] = new
        self.nodes_by_activity.setdefault(new, set()).add(node)

    def add_batch(self, events: Iterable[Sequence]) -> None:
        batch = list(events)
        if not batch:
            return
        if any(len(row) != 3 for row in batch):
            raise ValueError("history rows must be (timestamp, source, target)")
        times = {row[0] for row in batch}
        if len(times) != 1:
            raise ValueError("add_batch accepts one timestamp at a time")
        timestamp = batch[0][0]
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("history batches must be added in strictly increasing timestamp order")
        for _, u, v in batch:
            self.nodes.update((u, v))
            self.pairs.add(edge_key(u, v))
            self._change_activity(u, 1)
            self._change_activity(v, 1)
            self.events.append((timestamp, u, v))
            if len(self.events) > self.window:
                _, old_u, old_v = self.events.popleft()
                self._change_activity(old_u, -1)
                self._change_activity(old_v, -1)
        self.last_timestamp = timestamp


def generate_rank_pool_anomalies(
    events: Sequence[Sequence],
    target_count: int,
    *,
    reference_count: int | None = None,
    window: int = 1000,
    candidates_per_endpoint: int = 10,
    timestamp_seed: int = 42,
    reference_seed: int = 43,
    return_activity_errors: bool = False,
) -> list[Event] | tuple[list[Event], list[float]]:
    """Build candidate pairs around sampled reference edges, then select by activity error.

    Candidate matching uses benign-history snapshots. This allows global ranking of
    the candidate pool; injected anomalies do not feed back into later candidate
    activity counts in this batch-generation strategy.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if reference_count is None:
        reference_count = max(1, target_count // 10)
    if reference_count <= 0 or candidates_per_endpoint <= 0:
        raise ValueError("reference_count and candidates_per_endpoint must be positive")

    base_by_time: dict[int, list[tuple[int, int]]] = {}
    for t, u, v in events:
        base_by_time.setdefault(t, []).append((u, v))
    eligible_times = list(base_by_time)[2:]
    if not eligible_times:
        raise ValueError("need at least three distinct timestamps for injection")

    timestamp_rng = random.Random(timestamp_seed)
    reference_rng = random.Random(reference_seed)
    sampled_times = sorted(timestamp_rng.choices(eligible_times, k=reference_count))
    references_by_time: dict[int, list[tuple[int, int]]] = {}
    for t in sampled_times:
        references_by_time.setdefault(t, []).append(reference_rng.choice(base_by_time[t]))

    history = ActivityWindowIndex(window)
    candidates: dict[tuple[int, frozenset], tuple[float, int, int, int]] = {}
    for t in sorted(base_by_time):
        normal_edges = base_by_time[t]
        forbidden = {edge_key(u, v) for u, v in normal_edges}
        for u, v in references_by_time.get(t, ()):
            require_reference_edge(u, v)
            ordered_nodes = sorted(history.nodes, key=lambda node: (history.activity[node], node))
            ordered_keys = [(history.activity[node], node) for node in ordered_nodes]
            pools = []
            for endpoint in (u, v):
                target_activity = history.activity[endpoint]
                insertion = bisect_left(ordered_keys, (target_activity, endpoint))
                start = max(0, insertion - candidates_per_endpoint // 2)
                end = min(len(ordered_nodes), start + candidates_per_endpoint)
                start = max(0, end - candidates_per_endpoint)
                pools.append(ordered_nodes[start:end])

            for a in pools[0]:
                for b in pools[1]:
                    pair = edge_key(a, b)
                    if a == b or pair in history.pairs or pair in forbidden:
                        continue
                    direct = (abs(math.log1p(history.activity[u]) - math.log1p(history.activity[a]))
                              + abs(math.log1p(history.activity[v]) - math.log1p(history.activity[b]))) / 2
                    swapped = (abs(math.log1p(history.activity[u]) - math.log1p(history.activity[b]))
                               + abs(math.log1p(history.activity[v]) - math.log1p(history.activity[a]))) / 2
                    score = min(direct, swapped)
                    key = (t, pair)
                    old = candidates.get(key)
                    normalized = tuple(sorted((a, b)))
                    record = (score, t, normalized[0], normalized[1])
                    if old is None or record < old:
                        candidates[key] = record
        history.add_batch([(t, u, v) for u, v in normal_edges])

    ranked = sorted(candidates.values())
    selected = []
    selected_errors = []
    used_pairs = set()
    for score, t, u, v in ranked:
        pair = edge_key(u, v)
        if pair in used_pairs:
            continue
        selected.append((t, u, v))
        selected_errors.append(score)
        used_pairs.add(pair)
        if len(selected) == target_count:
            selected.sort()
            return (selected, selected_errors) if return_activity_errors else selected
    raise NoEligibleAnomalyError(
        f"candidate pool contains only {len(selected)} unique legal pairs for target {target_count}"
    )
