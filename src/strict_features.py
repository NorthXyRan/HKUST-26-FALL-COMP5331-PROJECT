"""Independent undirected TGF-style features from strict pre-timestamp history.

Feature names/formulas follow the pinned TGF main.py (see features.OFFICIAL_COMMIT).
Unlike upstream, the current event and simultaneous events are never inserted
before scoring. Absent endpoints/pairs have zero counters. Distribution statistics
refer to active historical nodes/pairs; zero absent counters have frequency zero.
The lower median matches upstream DSC.median (descending order, index n//2). No upstream code is imported.
"""
from collections import Counter, deque
from itertools import groupby

import pandas as pd

from .features import DEFAULT_HORIZONS


class WindowGraph:
    def __init__(self, horizon):
        self.horizon = horizon
        self.events = deque()
        self.weights = Counter()
        self.degree = Counter()
        self.activity = Counter()
        self.distributions = {name: Counter() for name in
                              ("degrees", "weighted_degrees", "link_weights")}

    def _change(self, counter, distribution, key, delta):
        old = counter[key]
        if old:
            distribution[old] -= 1
            if not distribution[old]:
                del distribution[old]
        new = old + delta
        if new < 0:
            raise AssertionError("negative graph counter")
        if new:
            counter[key] = new
            distribution[new] += 1
        else:
            del counter[key]

    def add(self, row):
        t, u, v = row[:3]
        pair = tuple(sorted((u, v)))
        if not self.weights[pair]:
            for node in (u, v):
                self._change(self.degree, self.distributions["degrees"], node, 1)
        self._change(self.weights, self.distributions["link_weights"], pair, 1)
        for node in (u, v):
            self._change(self.activity, self.distributions["weighted_degrees"], node, 1)
        self.events.append((t, u, v))
        if len(self.events) > self.horizon:
            _, a, b = self.events.popleft()
            old = tuple(sorted((a, b)))
            self._change(self.weights, self.distributions["link_weights"], old, -1)
            if not self.weights[old]:
                for node in (a, b):
                    self._change(self.degree, self.distributions["degrees"], node, -1)
            for node in (a, b):
                self._change(self.activity, self.distributions["weighted_degrees"], node, -1)

    def snapshot(self):
        result = {"history_size": self.horizon, "number_of_nodes": len(self.degree),
                  "number_of_links": len(self.weights), "total_weight": len(self.events)}
        distributions = {}
        for name, histogram in self.distributions.items():
            frequencies = histogram.copy()
            distributions[name] = frequencies
            values = sorted(frequencies)
            midpoint = (sum(frequencies.values()) - 1) // 2
            cumulative, median = 0, 0
            for value in values:
                cumulative += frequencies[value]
                if cumulative > midpoint:
                    median = value
                    break
            result.update({f"{name}_nb_one": frequencies[1], f"{name}_nb_two": frequencies[2],
                           f"{name}_min": values[0] if values else 0,
                           f"{name}_max": values[-1] if values else 0,
                           f"{name}_median": median})
        return result, distributions

    def score(self, u, v, snapshot):
        result, distributions = snapshot
        result = result.copy()
        # Canonical node ID breaks equal-degree ties deterministically.
        u, v = sorted((u, v), key=lambda n: (self.degree[n], n))
        for prefix, node in (("u", u), ("v", v)):
            for suffix, counts, distribution in (("degree", self.degree, "degrees"),
                                                  ("weighted_degree", self.activity, "weighted_degrees")):
                value = counts[node]
                freq = distributions[distribution]
                result[f"{prefix}_{suffix}"] = value
                result[f"{prefix}_same_{suffix}"] = freq[value]
                result[f"{prefix}_greater_{suffix}"] = sum(n for k, n in freq.items() if k > value)
        value = self.weights[tuple(sorted((u, v)))]
        freq = distributions["link_weights"]
        result["u_v_weight"] = value
        result["u_v_same_weight"] = freq[value]
        result["u_v_greater_weight"] = sum(n for k, n in freq.items() if k > value)
        return result


def strict_features(rows, horizons=DEFAULT_HORIZONS):
    """Preserve every row and its timestamp/label, including warm-up rows.

    Caller decides eligibility (the final pipeline uses a common benign warm-up).
    Histories carry forward across chronological splits without model refitting.
    """
    graphs = {h: WindowGraph(h) for h in horizons}
    records = []
    previous = None
    for t, batch_iter in groupby(rows, key=lambda r: r[0]):
        if previous is not None and t <= previous:
            raise ValueError("rows must be timestamp-sorted")
        batch = list(batch_iter)
        snapshots = {h: graph.snapshot() for h, graph in graphs.items()}
        for _, u, v, label in batch:
            if u == v or label not in (0, 1):
                raise ValueError("invalid stream row")
            record = {"t": t, "u": u, "v": v, "H5_is_fraud": label}
            for h, graph in graphs.items():
                record.update({f"H{h}_{key}": value for key, value in graph.score(u, v, snapshots[h]).items()})
            records.append(record)
        # Canonical order makes the event-count window deterministic when a batch
        # straddles its boundary. No row in this batch sees another row in it.
        for row in sorted(batch, key=lambda r: (r[1], r[2], r[3])):
            for graph in graphs.values():
                graph.add(row)
        previous = t
    return pd.DataFrame.from_records(records)
