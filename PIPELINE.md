# Experiment pipeline

Run commands from the project root with `.venv` activated and the pinned TGF
checkout installed as described in [README.md](README.md). The pipeline has
three explicit stages so the injected data, extracted predictors, and model
results can be inspected independently.

## 1. Inject

Generate random and activity-matched injection streams using the same sampled
timestamps. `--anomaly-percent` is the number injected divided by the number of
original events. The final stream's anomaly share is therefore lower; for 5%
injection relative to original edges, it is about 4.76%.

```bash
python -m src.pipeline inject \
  --data data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt \
  --seed 42 \
  --anomaly-percent 5 \
  --output results/full_pipeline/seed_42
```

This writes `random_injected.tsv.gz`, `activity_matched_injected.tsv.gz`, and
`run_config.json`. Timestamp sampling uses replacement and the two generators
share the sampled schedule. Activity matching uses strict prior-time history
(`t' < t`). The random generator is a seeded implementation of the timestamp
and endpoint candidate procedure in TGF; the original shell scripts are
unseeded.

## 2. Extract features

The untouched official extractor is invoked for each required history size.
H5 is always run to obtain the official fraud label. For `H1000`, H5 is not a
predictor; only the H1000 feature columns are used. `combined_H` uses H5, H50,
H100, H1000, H10000, and H15000 predictors.

```bash
python -m src.pipeline extract \
  --run-dir results/full_pipeline/seed_42 \
  --feature-set all
```

To extract one configuration only, pass `--feature-set H1000` or
`--feature-set combined_H`. Feature CSVs are saved alongside the input streams.

## 3. Evaluate

```bash
python -m src.pipeline evaluate \
  --run-dir results/full_pipeline/seed_42 \
  --train-fraction 0.70
```

Each feature configuration evaluates the following chronological train/test
settings:

1. `random_to_random`
2. `random_to_activity_matched`
3. `activity_matched_to_activity_matched`

The training set undersamples normal events to the anomaly count with
`random_state=12`, then shuffles with `random_state=42`. The model is the
scikit-learn default `RandomForestClassifier(random_state=seed)`. Metrics are
PR-AUC, ROC-AUC, and recall at 1% FPR.

Outputs include one metrics CSV per feature configuration,
`metrics_all_configurations.csv`, `summary.json`, and `REPORT.md`.

## Paper baseline comparison

The main experiment does not compare against or depend on paper scores. To
generate a separate comparison after evaluation, run:

```bash
python -m src.baseline_comparison \
  --run-dir results/full_pipeline/seed_42
```

This writes `baseline_comparison.json` and `BASELINE_COMPARISON.md` in the run
directory. The paper comparison uses only random→random ROC-AUC; the two
activity-matched settings are project experiments and have no corresponding
value in the paper's table.

| Injection | Configuration | Local ROC-AUC | Paper ROC-AUC | Difference |
|---:|---|---:|---:|---:|
| 1% | H1000 | 0.95909 | 0.9580 | +0.00109 |
| 1% | Combined H | 0.97996 | 0.9857 | **−0.00574** |
| 5% | H1000 | 0.96683 | 0.9635 | +0.00333 |
| 5% | Combined H | 0.98691 | 0.9876 | −0.00069 |
| 10% | H1000 | 0.96315 | 0.9646 | −0.00145 |
| 10% | Combined H | 0.98852 | 0.9872 | +0.00132 |

Difference is local ROC-AUC minus the paper value. These are the current seed-42
results. H1000 is close to the paper at all three injection rates. Combined H is
also close at 5% and 10%; the largest gap is at 1% (−0.00574).

## Three-setting results

The following tables compare ROC-AUC across the three train/test generator
settings. Random→random is the paper-comparable setting; activity-matched rows
are the project's extension.

### H1000

| Injection | Random → Random | Random → Activity-matched | Activity-matched → Activity-matched |
|---:|---:|---:|---:|
| 1% | 0.95909 | 0.81854 | 0.86980 |
| 5% | 0.96683 | 0.82833 | 0.87499 |
| 10% | 0.96315 | 0.84254 | 0.88323 |

### Combined H

| Injection | Random → Random | Random → Activity-matched | Activity-matched → Activity-matched |
|---:|---:|---:|---:|
| 1% | 0.97996 | 0.88372 | 0.94365 |
| 5% | 0.98691 | 0.89705 | 0.95354 |
| 10% | 0.98852 | 0.90554 | 0.96882 |

The cross-generator setting has lower ROC-AUC than random→random at each
injection rate. Training and testing on activity-matched anomalies recovers
some of that performance, while remaining below random→random.

The paper does not publish injection seeds, and its dataset table reports
59,835 UCI Messages links while the local preprocessed input contains 59,795.
Treat the comparison as a protocol/metric check rather than an exact injected
sample match.

## Run tests

```bash
python -m unittest discover -s tests -v
```
