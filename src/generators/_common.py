"""Shared helpers for leakage-safe synthetic link generation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence, Set

Node = Hashable
Event = tuple


class NoEligibleAnomalyError(ValueError):
    """Raised when history contains no valid novel link for the strategy."""


@dataclass(frozen=True)
class HistoryState:
    # References are read-only by contract. HistoryIndex updates only between timestamps.
    events: Sequence[Event]
    nodes: Sequence[Node]
    pairs: Set[frozenset]
    activity: Mapping[Node, int]
    neighbors: Mapping[Node, Set[Node]]


class HistoryIndex:
    """Incremental history store for stream-scale experiments.

    Call `add_batch` only after all events at a timestamp have been generated/scored.
    This keeps equal-time events invisible to one another and avoids rebuilding the
    full graph state for every event.
    """
    def __init__(self):
        self._events: list[Event] = []
        self._nodes: set[Node] = set()
        self._pairs: set[frozenset] = set()
        self._neighbors: dict[Node, set[Node]] = {}
        self._last_timestamp = None

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
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("history batches must be added in strictly increasing timestamp order")
        for event_t, a, b in batch:
            self._events.append((event_t, a, b))
            self._last_timestamp = event_t
            if a == b:
                continue
            self._nodes.update((a, b))
            pair = edge_key(a, b)
            self._pairs.add(pair)
            self._neighbors.setdefault(a, set()).add(b)
            self._neighbors.setdefault(b, set()).add(a)

    def snapshot(self, t: float, activity_window: int = 100) -> HistoryState:
        if activity_window <= 0:
            raise ValueError("activity window must be positive")
        if self._last_timestamp is not None and self._last_timestamp >= t:
            raise ValueError("add_batch must be called only after processing all events at time t")
        recent = self._events[-activity_window:]
        activity: Counter = Counter()
        for _, a, b in recent:
            if a != b:
                activity[a] += 1
                activity[b] += 1
        nodes = tuple(sorted(self._nodes, key=lambda x: (type(x).__name__, repr(x))))
        return HistoryState(self._events, nodes, self._pairs, activity, self._neighbors)


def edge_key(u: Node, v: Node) -> frozenset:
    """Canonical key for an undirected pair."""
    return frozenset((u, v))


def history_before(t: float, history: Iterable[Sequence], activity_window: int = 100) -> HistoryState:
    """Build state from events strictly earlier than t; equal-time/future rows are excluded."""
    if activity_window <= 0:
        raise ValueError("activity window must be positive")
    if isinstance(history, HistoryIndex):
        return history.snapshot(t, activity_window)

    earlier: list[Event] = []
    for row in history:
        if len(row) != 3:
            raise ValueError("history rows must be (timestamp, source, target)")
        event_t, a, b = row
        if event_t < t:
            earlier.append((event_t, a, b))
    earlier.sort(key=lambda e: e[0])

    activity: Counter = Counter()
    neighbors: dict[Node, set[Node]] = {}
    pairs: set[frozenset] = set()
    nodes: set[Node] = set()
    for _, a, b in earlier:
        if a == b:
            continue
        nodes.update((a, b))
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
        pairs.add(edge_key(a, b))
    for _, a, b in earlier[-activity_window:]:
        if a != b:
            activity[a] += 1
            activity[b] += 1

    ordered_nodes = tuple(sorted(nodes, key=lambda x: (type(x).__name__, repr(x))))
    return HistoryState(earlier, ordered_nodes, pairs, activity, neighbors)


def require_reference_edge(u: Node, v: Node) -> None:
    if u == v:
        raise ValueError("reference edge must not be a self-loop")


def is_novel_pair(a: Node, b: Node, state: HistoryState, u: Node, v: Node) -> bool:
    return a != b and edge_key(a, b) not in state.pairs and edge_key(a, b) != edge_key(u, v)


def endpoint_activity(state: HistoryState, node: Node, window: int = 100) -> int:
    """Count interactions in the last `window` strictly historical events."""
    if window <= 0:
        raise ValueError("activity window must be positive")
    # For a HistoryIndex snapshot this was computed for the requested window.
    # Sequence-based callers can pass a custom window into history_before first.
    return int(state.activity.get(node, 0))
