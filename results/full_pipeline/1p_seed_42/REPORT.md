# TGF injection experiment

- Data: `data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt` (59,795 original events)
- Injection: 1% of original events; 597 anomalies per generator.
- Seed: 42; chronological train fraction: 70%.
- Official feature extractor commit: `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`.
- Training: normal undersampling state 12, shuffle state 42, default Random Forest.
- Evaluation: PR-AUC, ROC-AUC, recall at 1% FPR.

## H1000

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.158868 | 0.959089 | 0.285714 |
| random_to_activity_matched | 0.034903 | 0.818537 | 0.071429 |
| activity_matched_to_activity_matched | 0.043578 | 0.869803 | 0.076923 |

## combined_H

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.329575 | 0.979965 | 0.474820 |
| random_to_activity_matched | 0.063537 | 0.883721 | 0.093525 |
| activity_matched_to_activity_matched | 0.151827 | 0.943647 | 0.215827 |

## Reproduction commands

```bash
python -m src.pipeline inject --data data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt --seed 42 --anomaly-percent 1 --output results/full_pipeline/1p_seed_42
python -m src.pipeline extract --run-dir results/full_pipeline/1p_seed_42 --feature-set all
python -m src.pipeline evaluate --run-dir results/full_pipeline/1p_seed_42 --train-fraction 0.7
```
