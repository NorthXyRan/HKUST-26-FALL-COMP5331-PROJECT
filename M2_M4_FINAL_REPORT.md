# M2 and M4 finalization report

Date: 6 November 2026 (Hong Kong time).

## What was updated

The actual Git checkout is `publish-fixed/` inside the supplied project directory. It was clean and was fast-forwarded from `5324acf` to `178ef4f` before implementation. The outer project directory is not a Git repository.

The final controlled comparison is frozen under `results/final_m2_m4/`. It replaces the preliminary dataset for M3/M5 comparative work. The original reproduction pipeline and its scores remain available and are explicitly labeled legacy; they should not be mixed with final results.

## Problems resolved

1. **Generator ambiguity:** finalized the original activity-bin method. The rank-pool method remains experimental because global candidate selection changes its timestamps/history semantics.
2. **Unfair candidate rules:** both final generators use historically observed nodes, undirected novel pairs, no self-loops, no same-time benign pair collisions, and no duplicate anomalies. They share injection timestamps and normal reference edges.
3. **History inconsistency:** both use only events strictly earlier than the current timestamp; previous anomalies enter later history. The entire timestamp batch is scored before its history update.
4. **Current-event feature contamination:** independently implemented TGF-style historical features. The original upstream extractor inserts the current edge first; the final extractor does not. Missing endpoints/pairs have zero historical counters.
5. **Missing validation:** benign timestamps define 60/20/20 train/validation/test splits before construction. Generator selection sees only the train/validation prefix. Splits never cut a timestamp batch.
6. **Test-derived thresholds:** the final model calibrates its threshold exclusively on validation negatives, handles score ties conservatively, and reports actual test FPR. The same random-trained threshold is reused for random and matched testing.
7. **Single-seed evidence:** completed 1%, 5%, and 10% injection at seeds 42, 43, and 44; evaluated H1000 and combined-H in all three settings (54 result rows). Saved sample standard deviations and paired performance differences.

## Final configuration

- Generator: activity-bin matching; window **1,000 events**, **10 bins**.
- Parameter grid: windows 100/1,000 × 2/5/10 bins; selection uses seed 42, 5% injection, a fixed 1,000-event diagnostic activity window, and validation matching error. All six candidates were feasible.
- Threshold objective: validation FPR ≤ 1%; actual test FPR is measured separately.
- Predictors: 34 for H1000; 204 for combined H (H5/H50/H100/H1000/H10000/H15000). Metadata and labels are excluded.
- Model: 100-tree Random Forest, fixed declared settings; normal training undersampling seed 12, shuffle seed 42, model seed equal to run seed.
- Histories carry across splits and use past observed events without their labels.
- First 15,000 benign events form a common anomaly-free warm-up. Per-split anomaly quotas are based on original benign counts; training anomalies are concentrated after warm-up, so the retained training prevalence is higher than the nominal combined-stream share.
- Injection percentages mean anomalies/original events, not anomalies/combined stream. Targets are 597, 2,989, and 5,979 anomalies per condition.
- No silent random fallback or bin widening. Inadequate coverage aborts the run; all final runs completed.

## Validation selection evidence

| Window | Bins | Validation mean error | Validation p90 |
|---:|---:|---:|---:|
| 100 | 2 | 1.0685 | 1.9520 |
| 100 | 5 | 1.0685 | 1.9520 |
| 100 | 10 | 1.0398 | 1.9923 |
| 1000 | 2 | 1.0274 | 2.1114 |
| 1000 | 5 | 0.7876 | 1.6384 |
| 1000 | 10 | 0.5296 | 1.2449 |

The same diagnostic window is used for every candidate so changing the generator window does not change the error definition. Selection is not based on final test performance.

## Final model results

Values are mean ± sample standard deviation across three seeds. PR-AUC is average precision. Recall uses the validation-calibrated threshold; its test FPR is not forced to 1%.

| Injection | Features | Train → test | PR-AUC | ROC-AUC | Recall | Actual test FPR |
|---:|---|---|---:|---:|---:|---:|
| 1% | H1000 | Matched → Matched | 0.0695 ± 0.0129 | 0.9006 ± 0.0122 | 0.0868 ± 0.0175 | 0.0064 ± 0.0010 |
| 1% | H1000 | Random → Matched | 0.0251 ± 0.0007 | 0.8213 ± 0.0065 | 0.0028 ± 0.0049 | 0.0056 ± 0.0018 |
| 1% | H1000 | Random → Random | 0.1083 ± 0.0168 | 0.9465 ± 0.0094 | 0.0924 ± 0.0445 | 0.0042 ± 0.0008 |
| 1% | combined_H | Matched → Matched | 0.1900 ± 0.0298 | 0.9654 ± 0.0047 | 0.0224 ± 0.0318 | 0.0007 ± 0.0009 |
| 1% | combined_H | Random → Matched | 0.0624 ± 0.0057 | 0.9240 ± 0.0078 | 0.0000 ± 0.0000 | 0.0013 ± 0.0006 |
| 1% | combined_H | Random → Random | 0.2642 ± 0.0179 | 0.9798 ± 0.0047 | 0.0840 ± 0.0468 | 0.0013 ± 0.0005 |
| 5% | H1000 | Matched → Matched | 0.2607 ± 0.0090 | 0.9085 ± 0.0060 | 0.0647 ± 0.0265 | 0.0049 ± 0.0017 |
| 5% | H1000 | Random → Matched | 0.1159 ± 0.0020 | 0.8276 ± 0.0029 | 0.0028 ± 0.0035 | 0.0032 ± 0.0008 |
| 5% | H1000 | Random → Random | 0.5185 ± 0.0403 | 0.9687 ± 0.0039 | 0.0808 ± 0.0445 | 0.0028 ± 0.0008 |
| 5% | combined_H | Matched → Matched | 0.5572 ± 0.0138 | 0.9699 ± 0.0010 | 0.0948 ± 0.0520 | 0.0025 ± 0.0016 |
| 5% | combined_H | Random → Matched | 0.2808 ± 0.0055 | 0.9354 ± 0.0036 | 0.0139 ± 0.0070 | 0.0038 ± 0.0006 |
| 5% | combined_H | Random → Random | 0.7025 ± 0.0275 | 0.9834 ± 0.0023 | 0.1193 ± 0.0756 | 0.0011 ± 0.0009 |
| 10% | H1000 | Matched → Matched | 0.3542 ± 0.0230 | 0.8972 ± 0.0105 | 0.0343 ± 0.0196 | 0.0036 ± 0.0022 |
| 10% | H1000 | Random → Matched | 0.2122 ± 0.0046 | 0.8353 ± 0.0053 | 0.0045 ± 0.0034 | 0.0060 ± 0.0015 |
| 10% | H1000 | Random → Random | 0.6536 ± 0.0361 | 0.9654 ± 0.0009 | 0.0847 ± 0.0439 | 0.0029 ± 0.0011 |
| 10% | combined_H | Matched → Matched | 0.6803 ± 0.0153 | 0.9681 ± 0.0028 | 0.0674 ± 0.0212 | 0.0024 ± 0.0008 |
| 10% | combined_H | Random → Matched | 0.4710 ± 0.0216 | 0.9407 ± 0.0043 | 0.0460 ± 0.0497 | 0.0055 ± 0.0059 |
| 10% | combined_H | Random → Random | 0.7850 ± 0.0165 | 0.9789 ± 0.0016 | 0.1469 ± 0.0735 | 0.0012 ± 0.0009 |

## Generator checks

| Injection | Seed | Target/generated per condition | Random mean activity error | Matched mean activity error | Violations |
|---:|---:|---:|---:|---:|---:|
| 1% | 42 | 597 / 597 | 2.0251 | 0.6190 | 0 |
| 1% | 43 | 597 / 597 | 2.0527 | 0.6617 | 0 |
| 1% | 44 | 597 / 597 | 2.1014 | 0.6722 | 0 |
| 5% | 42 | 2989 / 2989 | 1.9481 | 0.6377 | 0 |
| 5% | 43 | 2989 / 2989 | 1.9326 | 0.6408 | 0 |
| 5% | 44 | 2989 / 2989 | 1.9626 | 0.6424 | 0 |
| 10% | 42 | 5979 / 5979 | 1.9004 | 0.6303 | 0 |
| 10% | 43 | 5979 / 5979 | 1.9168 | 0.6357 | 0 |
| 10% | 44 | 5979 / 5979 | 1.9170 | 0.6305 | 0 |

Activity errors compare each anomaly with its saved normal reference using the same 1,000-event diagnostic window in that condition’s strict historical stream. Matching reduces this discrepancy; residual neighborhood, repeated-interaction, and burstiness differences remain M3 interpretation work.

## Tests and verification

**All 21 tests passed with none skipped** (6.398 seconds). Executed the full repository suite with `python -m unittest discover -s tests -v`. See [saved test output](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/M2_M4_TEST_RESULTS.txt).

Checks include original generator tests; brute-force sliding graph validation; identical-state formula comparison against the pinned upstream extractor; lower-median regression; current/same-time/future exclusion; unseen-node counters; tied timestamps; exact reproducible schedules; validation-threshold ties; test-label independence; and independent frozen-package checks of every source/artifact hash, all 18 streams, shared references, split labels, predictor metadata exclusion, and all 54 saved prediction/metric sets.

Also ran Python compilation and `git diff --check`. Source hashes, dataset SHA-256, runtime versions, all candidate selection results, and generation runtimes are saved.

## Exact M5 handoff

Use only [the frozen package](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/README.md) for the final ablation comparison.

1. Choose the same rate/seed folder for every full-model and ablated run.
2. Load the two saved feature tables; preserve the `split` column and row IDs.
3. Load the saved H1000 or combined-H predictor list. Exclude metadata and the label from predictors.
4. For every omitted feature group, retrain using train only, recalibrate on that training condition’s validation rows, and evaluate on the same test rows. `src.final_pipeline.fit_score` accepts the reduced column list.
5. Report changes relative to the saved full-model results with the same seeds and metrics. Use separate M3 diagnostics to explain neighborhood effects: the TGF-style predictor table itself does not contain explicit common-neighbor/Jaccard features.

## Files and commands

- [Final runner](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/src/final_pipeline.py).
- [Independent strict-history feature extractor](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/src/strict_features.py).
- [Protocol](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/protocol.json), [manifest](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/manifest.json), [runtime environment](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/environment.json).
- [All seed results](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/metrics_all_seeds.csv), [aggregate results](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/metrics_summary.csv), [paired deltas](https://github.com/NorthXyRan/HKUST-26-FALL-COMP5331-PROJECT/blob/main/results/final_m2_m4/paired_deltas.csv).

```sh
python -m src.final_pipeline --output results/reproduced_m2_m4
python -m unittest discover -s tests -v
```

Use a fresh output directory; frozen packages are not overwritten.

## Limits and remaining work

These are controlled comparative results, not exact upstream paper reproduction scores: candidate eligibility, chronological splitting, current-edge handling, and feature-history semantics have been corrected together. Attribution and the upstream reference revision are recorded. Three seeds and one validation selection seed/rate provide limited variability evidence. Results cover one dataset and one classifier. This work finalizes M2 generation and M4 comparisons; M3’s full distributional interpretation and M5’s feature ablations remain separate work. No messages were sent to other team members.
