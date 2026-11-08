"""Independent M3 counters: no classifier inputs or anomaly labels are queried.

Member A owns endpoint activity/window checks. Member B owns structure/timing.
All scoring is read-only. Call score_batch before add_batch for each timestamp.
"""

from collections import Counter, defaultdict, deque
import math


class Window:
    def __init__(self, size):
        if size <= 0:
            raise ValueError("window must be positive")
        self.size = size
        self.events = deque()
        self.activity = Counter()
        self.weights = Counter()
        self.neighbors = defaultdict(set)

    def add(self, u, v):
        pair = tuple(sorted((u, v)))
        self.events.append((u, v))
        self.activity[u] += 1
        self.activity[v] += 1
        self.weights[pair] += 1
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)
        if len(self.events) > self.size:
            a, b = self.events.popleft()
            self.activity[a] -= 1
            self.activity[b] -= 1
            key = tuple(sorted((a, b)))
            self.weights[key] -= 1
            if not self.weights[key]:
                del self.weights[key]
                self.neighbors[a].discard(b)
                self.neighbors[b].discard(a)

    def score(self, u, v):
        low, high = sorted((self.activity[u], self.activity[v]))
        a, b = self.neighbors.get(u, set()), self.neighbors.get(v, set())
        return dict(
            activity_low=low,
            activity_high=high,
            degree_mean=(len(a) + len(b)) / 2,
            common_neighbors=len(a & b),
            jaccard=len(a & b) / len(a | b) if a | b else 0,
            recent_pair_count=self.weights[tuple(sorted((u, v)))],
        )


def interval_burstiness(times, min_intervals=5):
    """B=(population SD-mean)/(SD+mean), last <=20 inter-event intervals.

    Same-time endpoint events create zero intervals. Fewer than five intervals
    or an all-zero denominator is undefined (NaN), not an imputed zero.
    """
    gaps = [b - a for a, b in zip(times, list(times)[1:])]
    if len(gaps) < min_intervals:
        return math.nan
    mean = sum(gaps) / len(gaps)
    sd = math.sqrt(sum((x - mean) ** 2 for x in gaps) / len(gaps))
    return (sd - mean) / (sd + mean) if sd + mean else math.nan


class History:
    def __init__(self, windows=(100, 1000, 10000)):
        self.windows = {w: Window(w) for w in windows}
        self.pairs = Counter()
        self.times = defaultdict(lambda: deque(maxlen=21))
        self.queues = {3600: deque(), 86400: deque()}
        self.counts = {h: Counter() for h in self.queues}
        self.last_batch = None
        self.query_time = None

    def advance(self, t):
        if self.last_batch is not None and t <= self.last_batch:
            raise ValueError("queries must be strictly after inserted history")
        if self.query_time is not None and t < self.query_time:
            raise ValueError("query time cannot move backwards")
        for horizon, q in self.queues.items():
            while q and q[0][0] < t - horizon:
                _, u, v = q.popleft()
                self.counts[horizon][u] -= 1
                self.counts[horizon][v] -= 1
        self.query_time = t

    def score(self, t, u, v):
        if t != self.query_time:
            raise ValueError("advance to query timestamp first")
        a, b = self.times.get(u, ()), self.times.get(v, ())
        bursts = [interval_burstiness(x) for x in (a, b)]
        timing = dict(
            pair_count=self.pairs[tuple(sorted((u, v)))],
            recency_minutes=((t - a[-1]) + (t - b[-1])) / 120 if a and b else math.nan,
            unseen_endpoint=int(not a or not b),
            burstiness=sum(bursts) / 2,
            concentration_1h_24h=sum(
                self.counts[3600][n] / self.counts[86400][n] if self.counts[86400][n] else 0
                for n in (u, v)
            )
            / 2,
        )
        return {w: dict(**window.score(u, v), **timing) for w, window in self.windows.items()}

    def add_batch(self, t, pairs):
        if self.query_time != t or (self.last_batch is not None and t <= self.last_batch):
            raise ValueError("advance/score before inserting a new timestamp batch")
        pairs = list(pairs)
        if any(u == v for u, v in pairs):
            raise ValueError("self-loop")
        # Matches frozen stream's numeric endpoint ordering at event-window boundaries.
        for u, v in sorted(pairs):
            self.pairs[tuple(sorted((u, v)))] += 1
            self.times[u].append(t)
            self.times[v].append(t)
            for w in self.windows.values():
                w.add(u, v)
            for horizon, q in self.queues.items():
                q.append((t, u, v))
                self.counts[horizon][u] += 1
                self.counts[horizon][v] += 1
        self.last_batch = t


def activity_error(reference, anomaly):
    return (
        sum(
            abs(math.log1p(reference[k]) - math.log1p(anomaly[k]))
            for k in ("activity_low", "activity_high")
        )
        / 2
    )
