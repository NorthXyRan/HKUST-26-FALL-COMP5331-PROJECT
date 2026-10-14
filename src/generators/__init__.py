"""Common interface for the two controlled anomaly generators."""
from __future__ import annotations

from typing import Iterable, Sequence

from ._common import Event, HistoryIndex, NoEligibleAnomalyError
from .activity_matched import generate_activity_matched_anomaly
from .random import generate_random_anomaly


def generate_anomaly(
    t,
    u,
    v,
    history,
    strategy="random",
    *,
    rng=None,
    activity_window=100,
    quantile_bins=10,
):
    """Generate (t, a, b) using `random` or `activity_matched` strategy.

    `history` is an iterable of (timestamp, source, target) events. Events at time t
    and later are ignored, so simultaneous events never see one another.
    """
    options = {"rng": rng}
    if strategy == "random":
        return generate_random_anomaly(t, u, v, history, **options)
    if strategy == "activity_matched":
        return generate_activity_matched_anomaly(
            t, u, v, history,
            activity_window=activity_window,
            quantile_bins=quantile_bins,
            **options,
        )
    raise ValueError("strategy must be 'random' or 'activity_matched'")


__all__ = [
    "Event",
    "HistoryIndex",
    "NoEligibleAnomalyError",
    "generate_anomaly",
    "generate_random_anomaly",
    "generate_activity_matched_anomaly",
]
