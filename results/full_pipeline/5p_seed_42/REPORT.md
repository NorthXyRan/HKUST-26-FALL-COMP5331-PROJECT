# TGF injection experiment

- Data: `data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt` (59,795 original events)
- Injection: 5% of original events; 2,989 anomalies per generator.
- Seed: 42; chronological train fraction: 70%.
- Official feature extractor commit: `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`.
- Training: normal undersampling state 12, shuffle state 42, default Random Forest.
- Evaluation: PR-AUC, ROC-AUC, recall at 1% FPR.

## H1000

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.522872 | 0.966832 | 0.323991 |
| random_to_activity_matched | 0.140232 | 0.828334 | 0.036996 |
| activity_matched_to_activity_matched | 0.174004 | 0.874993 | 0.047085 |

## combined_H

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.760949 | 0.986913 | 0.617176 |
| random_to_activity_matched | 0.241567 | 0.897047 | 0.104803 |
| activity_matched_to_activity_matched | 0.525322 | 0.953539 | 0.358079 |

## Reproduction commands

```bash
python -m src.pipeline inject --data data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt --seed 42 --anomaly-percent 5 --output results/full_pipeline/5p_seed_42
python -m src.pipeline extract --run-dir results/full_pipeline/5p_seed_42 --feature-set all
python -m src.pipeline evaluate --run-dir results/full_pipeline/5p_seed_42 --train-fraction 0.7
```
