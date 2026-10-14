# Preliminary experiment outputs

The `metrics_*.csv` files contain the three-seed comparison. Each `seed_<n>/`
directory stores the metrics and a `summary.json` manifest with input hashes,
TGF commit and extractor hash, versions, split counts/boundary, settings, and
sub-seeds. Labeled injected streams and extracted feature tables are intentionally
not tracked; reproduce them with `python -m src.experiment --seeds 42,43,44`.
