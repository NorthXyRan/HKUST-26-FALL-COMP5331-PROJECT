"""Uniform random non-edge generator, independent of the official TGF code."""
from __future__ import annotations

import itertools
import random as _random
from typing import Iterable, Sequence

from ._common import Event, NoEligibleAnomalyError, history_before, is_novel_pair, require_reference_edge


def generate_random_anomaly(
    t: float,
    u,
    v,
    history: Iterable[Sequence],
    *,
    rng: _random.Random | None = None,
    max_attempts: int = 4096,
) -> Event:
    """Return (t, a, b), uniformly choosing a novel undirected pair from past nodes.

    Rejection sampling avoids materializing O(nodes^2) candidate pairs. If coverage
    is extremely low, a one-pass reservoir fallback samples exactly from all valid
    pairs. Pass a seeded `random.Random` instance for repeatability.
    """
    require_reference_edge(u, v)
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    state = history_before(t, history)
    if len(state.nodes) < 2:
        raise NoEligibleAnomalyError("need at least two nodes observed before t")
    generator = rng if rng is not None else _random.Random()

    for _ in range(max_attempts):
        a, b = generator.sample(state.nodes, 2)
        if is_novel_pair(a, b, state, u, v):
            return (t, a, b)

    # Rare low-coverage fallback: reservoir sampling keeps memory O(1) and is uniform.
    selected = None
    valid_count = 0
    for a, b in itertools.combinations(state.nodes, 2):
        if is_novel_pair(a, b, state, u, v):
            valid_count += 1
            if generator.randrange(valid_count) == 0:
                selected = (a, b)
    if selected is None:
        raise NoEligibleAnomalyError("no novel non-self pair among nodes observed before t")
    return (t, selected[0], selected[1])


def generate_anomaly(t, u, v, history, strategy="random", *, rng=None, **kwargs):
    """Module-level interface retained for callers importing this file directly."""
    if strategy != "random":
        raise ValueError("src.generators.random supports strategy='random' only")
    return generate_random_anomaly(t, u, v, history, rng=rng, **kwargs)
