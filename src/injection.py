"""Input validation and seeded anomaly-stream construction."""
from __future__ import annotations

import gzip
import random
from collections import defaultdict
from pathlib import Path

from .generators import HistoryIndex, NoEligibleAnomalyError, generate_activity_matched_anomaly


def read_events(path: Path) -> list[tuple[int, int, int]]:
    events = []
    previous = None
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_no}: expected three columns t u v")
            t, u, v = map(int, fields)
            if previous is not None and t < previous:
                raise ValueError(f"{path}:{line_no}: input is not timestamp-sorted")
            if u == v:
                raise ValueError(f"{path}:{line_no}: self-loop in preprocessed input")
            events.append((t, u, v))
            previous = t
    if not events:
        raise ValueError(f"no events in {path}")
    return events


def _eligible_timestamps(events: list[tuple[int, int, int]]) -> list[int]:
    seen: set[int] = set()
    eligible = []
    for t, _, _ in events:
        if t not in seen:
            seen.add(t)
            if len(seen) > 2:
                eligible.append(t)
    if not eligible:
        raise ValueError("need at least three distinct timestamps for injection")
    return eligible


def make_shared_timestamps(events, n_injected: int, seed: int) -> list[int]:
    """Sample timestamps with replacement and share them between generators."""
    return sorted(random.Random(seed).choices(_eligible_timestamps(events), k=n_injected))


def make_activity_matched(events, timestamps, seed, *, activity_window=100, bins=10):
    """Generate edges at sampled times using strict prior-timestamp history."""
    base_by_time: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for t, u, v in events:
        base_by_time[t].append((u, v))
    counts: dict[int, int] = defaultdict(int)
    for t in timestamps:
        counts[t] += 1

    rng = random.Random(seed)
    history = HistoryIndex()
    generated = []
    for t in sorted(base_by_time):
        current = []
        blocked = {frozenset((u, v)) for u, v in base_by_time[t]}
        for _ in range(counts.get(t, 0)):
            reference_u, reference_v = rng.choice(base_by_time[t])
            accepted = None
            for _attempt in range(256):
                try:
                    candidate = generate_activity_matched_anomaly(
                        t, reference_u, reference_v, history, rng=rng,
                        activity_window=activity_window, quantile_bins=bins,
                    )
                except NoEligibleAnomalyError:
                    break
                key = frozenset(candidate[1:])
                if key not in blocked:
                    accepted = candidate
                    blocked.add(key)
                    break
            if accepted is None:
                raise NoEligibleAnomalyError(
                    f"activity matching failed at timestamp {t}; lower bins or report coverage failure"
                )
            current.append(accepted)
            generated.append(accepted)
        # Same-timestamp events become visible only after all candidates at t.
        history.add_batch([(t, u, v) for u, v in base_by_time[t]] + current)
    return generated


def make_random_tgf_style(events, timestamps, seed):
    """Seeded equivalent of TGF's timestamp/global-node candidate procedure."""
    rng = random.Random(seed)
    nodes = sorted({node for _, u, v in events for node in (u, v)})
    original_at_time: dict[int, set[frozenset[int]]] = defaultdict(set)
    for t, u, v in events:
        original_at_time[t].add(frozenset((u, v)))
    used: dict[int, set[frozenset[int]]] = defaultdict(set)
    generated = []
    for t in timestamps:
        accepted = None
        for _attempt in range(10000):
            u, v = rng.sample(nodes, 2)
            key = frozenset((u, v))
            if key not in original_at_time[t] and key not in used[t]:
                accepted = (t, min(u, v), max(u, v))
                used[t].add(key)
                break
        if accepted is None:
            raise RuntimeError(f"could not find a random candidate for timestamp {t}")
        generated.append(accepted)
    return generated


def stream_rows(events, anomalies):
    rows = [(t, u, v, 0) for t, u, v in events]
    rows.extend((t, u, v, 1) for t, u, v in anomalies)
    return sorted(rows, key=lambda row: (row[0], str(row[1]), str(row[2]), row[3]))


def write_stream(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write("%d %d %d %d\n" % row)
