# Frozen M2/M4 package: strict_m2_m4_v1

This directory is the canonical input for M3 diagnostics and M5 ablations.
The legacy `results/full_pipeline/` directories are preliminary reproduction
artifacts and must not be mixed with these results.

## Protocol

See `protocol.json` for exact parameters. Normal events are split at fixed benign
60/20/20 timestamp boundaries before anomaly construction. Equal-time events stay
together. Both conditions share timestamps, reference edges, target counts,
previously observed candidate nodes, novel undirected-pair restrictions, and
strict past-only history. Prior injected edges enter later history. Same-time
normal pairs and duplicate anomalies are blocked. There is no random fallback or
bin widening; an infeasible run fails closed.

Six activity-window/bin configurations were evaluated on a train/validation-only
prefix at 5% injection and seed 42. The fixed comparison metric is endpoint
log-activity error in the previous 1,000 observed events, with the smaller error
over the two undirected endpoint alignments. Select feasibility, lowest validation
mean, lowest p90, smaller window, then fewer bins. Test data are not used to choose
settings. `parameter_selection.csv` retains all candidates and their runtimes.
The selected configuration is window 1,000 / 10 bins.

Injection percentages mean injected count/original benign count. Per-split quotas
use largest-remainder rounding to preserve the exact total floor(p E). No anomalies
are injected in the initial 15,000 benign-event warm-up; training quotas are placed
in the rest of the training period. Consequently the retained training anomaly
share is higher than p/(1+p). Counts are reported per run. This is common to both
conditions. History carries into validation/test and advances using observed events
without using their labels. Canonical numeric endpoint order breaks ties at window
boundaries; all events at t are scored before any event at t enters history.

The feature implementation independently computes TGF-style graph counters and
statistics from prior timestamps only. This deliberately changes the upstream
current-edge-inclusive semantics. Absent endpoints/pairs have zero counts; graph
distributions include active historical nodes/pairs, use lower medians, and assign
zero frequency to absent counters. The original injected label is retained.
Feature formulas/names are based on the upstream TGF revision recorded in
`protocol.json`; no upstream implementation is imported. These are comparative
project results, not exact upstream paper reproduction scores.

Random Forest settings, balancing seeds, and feature horizons are fixed in the
protocol. Thresholds come from validation negatives of the training condition,
with ties handled so validation FPR does not exceed 1%. The random-trained model's
threshold stays the same for random and matched testing. Actual test FPR is reported
and may differ from 1%; the recall column is therefore explicitly named
`recall_at_validation_1pct_fpr`. PR-AUC means average precision.

## Files

- `metrics_all_seeds.csv`: 54 rows = 3 rates × 3 seeds × 2 feature sets × 3 settings.
- `metrics_summary.csv`: mean, sample standard deviation, minimum and maximum.
- `paired_deltas.csv`: within-seed cross-generator change and retraining recovery.
- `paired_deltas_summary.csv`: means/sample standard deviations of paired deltas.
- `manifest.json`: dataset provenance, exact source hashes, and artifact hashes.
- Each `*p_seed_*` directory contains paired streams, anomaly audit/reference rows,
  features with timestamps/splits/event IDs, predictor-column lists, metrics,
  generation diagnostics, shared timestamps, and per-event test predictions.

## M5 use

For each rate/seed use the two saved `*_features.csv.gz` tables. Use the saved
`split` column, original `H5_is_fraud` label, and the exact predictor list from
`H1000_columns.json` or `combined_H_columns.json`. Never train on `t`, `u`, `v`,
`event_id`, `split`, or the label. Retrain each ablated model on train, recalibrate
its threshold on the corresponding validation set, and evaluate on the same test
rows. `src.final_pipeline.fit_score` accepts a reduced column list for this purpose.
Do not select feature groups or thresholds using test outcomes. The underlying
TGF-style features contain degree/activity and pair-weight statistics; claims about
neighborhood overlap require separate M3 diagnostics, not invented feature groups.

## Reproduce and verify

From the repository root with requirements installed:

```sh
python -m src.final_pipeline --output results/reproduced_m2_m4
python -m unittest discover -s tests -v
```

Use a fresh output directory: the runner refuses to overwrite an existing protocol.
Gzip feature/prediction files are deterministic. Runtime fields and provenance
revision may differ; numeric results should reproduce with the recorded runtime
versions. Frozen-artifact tests validate the canonical package's hashes, stream
constraints, split membership, and metrics independently from generation.

## Limits

Three injection/model seeds give preliminary variability estimates, not a large
sample confidence claim. Parameter selection uses one validation seed/rate and a
small declared grid. Only UCI Messages and one fixed classifier are tested. The
rank-pool generator remains experimental and is not used for these comparisons.
The controlled random condition has stricter eligibility than the original TGF
reproduction baseline, so changes relative to legacy scores cannot be attributed
solely to activity matching. Use the paired conditions within this package.
