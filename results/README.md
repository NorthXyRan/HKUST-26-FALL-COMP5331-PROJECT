# Results

The `full_pipeline/` directory contains complete seed-42 runs at 1%, 5%, and
10% injection. Each run includes both injected streams, H1000 and combined-H
feature tables, per-setting and combined metrics, a run configuration,
`summary.json`, and a readable `REPORT.md`.

All three manifests record the same input dataset SHA-256, pinned TGF commit,
injection settings, and random seeds. The project code revision and source hashes
are recorded with the run metadata. The metric CSVs contain PR-AUC, ROC-AUC,
recall at 1% FPR, actual FPR, sample counts, and feature counts.

See [PIPELINE.md](../PIPELINE.md) for reproduction and optional paper-comparison
commands. These results use one seed and are preliminary pending repeated-seed
analysis and generator diagnostics.
