"""Recent-activity-matched link generator; new implementation for this project."""
from __future__ import annotations

import itertools
import random as _random
from bisect import bisect_left
from typing import Iterable, Sequence

from ._common import Event, NoEligibleAnomalyError, endpoint_activity, history_before, is_novel_pair, require_reference_edge


def _quantile_bucket(value: int, sorted_values: list[int], bins: int) -> int:
    # Rank-based buckets tolerate count ties and avoid fixed, scale-dependent cutoffs.
    # The sorted list makes each rank lookup O(log n), keeping stream generation
    # practical when this function is called thousands of times.
    rank = bisect_left(sorted_values, value)
    return min(bins - 1, (rank * bins) // max(1, len(sorted_values)))


def generate_activity_matched_anomaly(
    t: float,
    u,
    v,
    history: Iterable[Sequence],
    *,
    rng: _random.Random | None = None,
    activity_window: int = 100,
    quantile_bins: int = 10,
) -> Event:
    """Return (t, a, b), matching each endpoint's recent activity quantile.

    Activity is interaction count over the last `activity_window` events with time < t.
    Each synthetic endpoint is sampled from the same coarse rank-quantile as its
    corresponding reference endpoint. Candidate coverage failures raise
    `NoEligibleAnomalyError`; there is deliberately no fallback to random sampling.
    Tune window/bin settings on training/validation data, then freeze them for test.
    """
    require_reference_edge(u, v)
    if activity_window <= 0:
        raise ValueError("activity_window must be positive")
    if quantile_bins < 2:
        raise ValueError("quantile_bins must be at least 2")

    state = history_before(t, history, activity_window)
    if len(state.nodes) < 2:
        raise NoEligibleAnomalyError("need at least two past-observed nodes")

    values = [endpoint_activity(state, node, activity_window) for node in state.nodes]
    sorted_values = sorted(values)
    ref_u = endpoint_activity(state, u, activity_window)
    ref_v = endpoint_activity(state, v, activity_window)
    bucket_u = _quantile_bucket(ref_u, sorted_values, quantile_bins)
    bucket_v = _quantile_bucket(ref_v, sorted_values, quantile_bins)
    buckets = [_quantile_bucket(value, sorted_values, quantile_bins) for value in values]
    pool_u = [node for node, bucket in zip(state.nodes, buckets) if bucket == bucket_u]
    pool_v = [node for node, bucket in zip(state.nodes, buckets) if bucket == bucket_v]

    generator = rng if rng is not None else _random.Random()
    if not pool_u or not pool_v:
        raise NoEligibleAnomalyError("reference activity bin has no eligible nodes")
    for _ in range(4096):
        a, b = generator.choice(pool_u), generator.choice(pool_v)
        if is_novel_pair(a, b, state, u, v):
            return (t, a, b)

    # Low-coverage fallback is exact and uses O(1) auxiliary memory.
    selected = None
    valid_count = 0
    for a, b in itertools.product(pool_u, pool_v):
        if is_novel_pair(a, b, state, u, v):
            valid_count += 1
            if generator.randrange(valid_count) == 0:
                selected = (a, b)
    if selected is None:
        raise NoEligibleAnomalyError(
            "no novel pair has endpoints in the reference activity bins; record this as a coverage failure"
        )
    return (t, selected[0], selected[1])


def generate_anomaly(t, u, v, history, strategy="activity_matched", *, rng=None, **kwargs):
    if strategy != "activity_matched":
        raise ValueError("src.generators.activity_matched supports strategy='activity_matched' only")
    return generate_activity_matched_anomaly(t, u, v, history, rng=rng, **kwargs)
