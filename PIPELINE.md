# Reproduction pipeline (preliminary snapshot)

This repository contains the independent experiment driver, configuration, tests,
and metrics. The official TGF source and UCI Messages input are not copied into
this repository. They are available from the upstream TGF repository.

## Setup and run

From the repository root, clone the official implementation at the commit recorded
in the result manifests, then create a Python virtual environment:

```bash
git clone https://github.com/StephanyRajeh/TGF.git TGF
git -C TGF checkout af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment --seeds 42,43,44
```

The data path defaults to `TGF/1-PreprocessingOfDatasets/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt`.
If the cleaned input is elsewhere, pass it with `--data`. The official feature
extractor is called in place from `TGF/3-FeatureGeneration/main.py`; generated
feature tables and labeled streams are ignored by Git because they can be
regenerated from the recorded seeds.

## Preliminary result

The current run adds 2,989 anomalies (5% of 59,795 original events) per condition.
The resulting combined stream has a 4.76% anomaly share. The Random Forest uses
the official notebook's chronological 70/30 feature-row split, balanced training
by undersampling normal rows with seed 12, training shuffle seed 42, and default
`RandomForestClassifier` settings. The run uses TGF H histories 5, 50, 100, 1000,
10000, and 15000, drops the 15,000 warm-up rows, and evaluates PR-AUC, ROC-AUC,
and recall at 1% FPR.

| Train → test | PR-AUC mean (SD) | ROC-AUC mean (SD) | Recall at 1% FPR mean (SD) |
|---|---:|---:|---:|
| Random → random | 0.7541 (0.0061) | 0.9872 (0.0004) | 0.6228 (0.0057) |
| Random → activity-matched | 0.2303 (0.0108) | 0.9014 (0.0059) | 0.0983 (0.0122) |
| Activity-matched → activity-matched | 0.5282 (0.0176) | 0.9611 (0.0066) | 0.3525 (0.0386) |

See [`results/metrics_all_seeds.csv`](results/metrics_all_seeds.csv),
[`results/metrics_mean_sd.csv`](results/metrics_mean_sd.csv), and each seed's
`summary.json` for exact values and code/data hashes.

## Scope and limitations

These are preliminary results, not the final controlled comparison. The random
injection is a seeded reimplementation of TGF's timestamp/global-node candidate
procedure; upstream shell scripts use unseeded randomness. The activity-matched
generator additionally requires a pair to be absent from strict prior history,
while TGF's random candidate filter only excludes an exact timestamp/pair already
in the input. The measured difference therefore includes both activity matching
and candidate-eligibility differences.

The experiment currently injects before making the chronological feature-row
split, and the official extractor processes equal-timestamp rows sequentially.
This reproduces the saved notebook's overall feature/training setup, but it does
not yet satisfy the proposal's stricter split-before-injection and timestamp-strict
history requirements for a final test. The feature extractor source remains
unchanged; changes required for the final study should be implemented in the
independent pipeline and clearly distinguished from the reproduction baseline.
