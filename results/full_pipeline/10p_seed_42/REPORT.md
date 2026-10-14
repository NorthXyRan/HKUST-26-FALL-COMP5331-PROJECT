# TGF injection experiment

- Data: `data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt` (59,795 original events)
- Injection: 10% of original events; 5,979 anomalies per generator.
- Seed: 42; chronological train fraction: 70%.
- Official feature extractor commit: `af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`.
- Training: normal undersampling state 12, shuffle state 42, default Random Forest.
- Evaluation: PR-AUC, ROC-AUC, recall at 1% FPR.

## H1000

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.650707 | 0.963149 | 0.285060 |
| random_to_activity_matched | 0.256815 | 0.842536 | 0.045220 |
| activity_matched_to_activity_matched | 0.316328 | 0.883229 | 0.057813 |

## combined_H

| Train → test | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---:|---:|---:|
| random_to_random | 0.875920 | 0.988523 | 0.685673 |
| random_to_activity_matched | 0.396210 | 0.905541 | 0.116228 |
| activity_matched_to_activity_matched | 0.718210 | 0.968825 | 0.412281 |

## Reproduction commands

```bash
python -m src.pipeline inject --data data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt --seed 42 --anomaly-percent 10 --output results/full_pipeline/10p_seed_42
python -m src.pipeline extract --run-dir results/full_pipeline/10p_seed_42 --feature-set all
python -m src.pipeline evaluate --run-dir results/full_pipeline/10p_seed_42 --train-fraction 0.7
```
