# Anomaly generators

`generate_anomaly(t, u, v, history, strategy="random")` returns `(t, a, b)`.
`history` is an iterable of `(timestamp, source, target)` events. Only events whose
timestamp is strictly less than `t` are visible; same-timestamp events are excluded.
The returned pair is undirected, has no self-loop, is absent from observed history,
and is different from the reference pair.

Strategies:

- `random`: uniformly selects among novel pairs on nodes already seen in history.
- `activity_matched`: samples each endpoint from the same rank-quantile of recent
  interaction-count activity as its reference endpoint. Defaults are 100 past events
  and 10 bins. Select these using training/validation data, then freeze for testing.
  Inadequate candidate coverage raises `NoEligibleAnomalyError`; it never silently
  falls back to random sampling.

Example:

```python
import random
from src.generators import generate_anomaly

history = [(1, "a", "b"), (2, "a", "c"), (3, "b", "d"), (4, "a", "d")]
edge = generate_anomaly(5, "a", "c", history, "activity_matched", rng=random.Random(7))
```

The caller owns stream ordering and history updates. Use one immutable history
snapshot for all events with the same timestamp; decide whether prior injected links
enter subsequent history and apply the same policy to both strategies and feature
extraction. Generator code does not mutate history.

For the full stream, `HistoryIndex` avoids rebuilding the observed node/pair sets for
every event. Generate the whole equal-timestamp batch against one snapshot, then call
`history.add_batch(batch)` before advancing to the next timestamp.
