# Results

Tracked `metrics_*.csv` and `seed_*/` files are earlier preliminary runs. The
current staged pipeline writes new outputs under a run-specific directory, for
example `results/full_pipeline/seed_42/`. Generated streams, extracted feature
tables, and optional baseline comparisons are ignored by Git because they can
be regenerated from the recorded run configuration.

Commands and output definitions are in [PIPELINE.md](../PIPELINE.md).
