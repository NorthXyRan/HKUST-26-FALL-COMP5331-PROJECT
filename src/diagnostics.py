"""Leakage-safe diagnostics for generated versus reference links."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .generators._common import edge_key, endpoint_activity, history_before


def _jaccard(a, b) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def compare_generators(reference_events, generated_events, history_snapshots, *, activity_window=100):
    """Compare aligned normal events and generated links using their pre-event histories.

    Inputs are aligned sequences of (t,u,v), (t,a,b), and history iterables. This
    reports intended endpoint-activity matching plus pair-history/neighborhood checks.
    It never reads events at or after each event timestamp.
    """
    n = len(reference_events)
    if len(generated_events) != n or len(history_snapshots) != n:
        raise ValueError("reference, generated, and history sequences must have equal length")
    if n == 0:
        raise ValueError("at least one aligned event is required")

    activity_errors = []
    reference_repeat = []
    generated_repeat = []
    reference_overlap = []
    generated_overlap = []
    for reference, generated, history in zip(reference_events, generated_events, history_snapshots):
        t, u, v = reference
        gt, a, b = generated
        if gt != t:
            raise ValueError("reference and generated events must share timestamps")
        state = history_before(t, history, activity_window)
        recent_a = endpoint_activity(state, u, activity_window)
        recent_b = endpoint_activity(state, v, activity_window)
        recent_x = endpoint_activity(state, a, activity_window)
        recent_y = endpoint_activity(state, b, activity_window)
        activity_errors.append((abs(math.log1p(recent_a) - math.log1p(recent_x)) + abs(math.log1p(recent_b) - math.log1p(recent_y))) / 2)
        reference_repeat.append(float(edge_key(u, v) in state.pairs))
        generated_repeat.append(float(edge_key(a, b) in state.pairs))
        reference_overlap.append(_jaccard(state.neighbors.get(u, frozenset()), state.neighbors.get(v, frozenset())))
        generated_overlap.append(_jaccard(state.neighbors.get(a, frozenset()), state.neighbors.get(b, frozenset())))

    mean = lambda xs: sum(xs) / len(xs)
    return {
        "n_events": n,
        "mean_log_activity_abs_error": mean(activity_errors),
        "reference_prior_pair_rate": mean(reference_repeat),
        "generated_prior_pair_rate": mean(generated_repeat),
        "reference_mean_neighborhood_jaccard": mean(reference_overlap),
        "generated_mean_neighborhood_jaccard": mean(generated_overlap),
        "generated_novel_pair_rate": 1.0 - mean(generated_repeat),
    }
