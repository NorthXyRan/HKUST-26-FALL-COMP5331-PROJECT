# HKUST · COMP5331 · Fall 2026
## Stress-Testing Simple Graph Features under Controlled Link-Anomaly Injection
## Important Dates

All dates are in **2026**. All times are in **Hong Kong time (HKT, UTC+8)**.

| Item | Deadline / Schedule | Notes |
| --- | --- | --- |
| ~~Group Forming~~ | ~~18 September, 13:30~~ | ~~Group forming deadline.~~ |
| ~~Proposal~~ | ~~25 September, 13:30~~ | ~~Proposal submission deadline.~~ |
| ~~Project Presentation Schedule Availability Indication~~ | ~~2 October, 13:30~~ | ~~Deadline to indicate presentation availability.~~ |
| Final Report | 13 November, 13:30 | Final report submission deadline. |
| Student Presentation Order Form | **Before our assigned presentation time slot** | Submit before the session begins. |
| PPT / Source Code | **Before our assigned presentation time slot** | Submit the presentation slides and source code before the session begins. |
| Presentation Dates | **29 November, 09:00–10:30** | Venue: **TBA**; face-to-face oral presentation; **6 members, 18–20 minutes total**. |
| Declaring Contribution of Each Member | **Last day of all presentations, 23:59** | **Every member must submit the declaration independently.** |

## Milestones

The dates below are internal targets in **2026 (HKT)**. The official deadlines remain those listed above. The core study uses **UCI Messages, two anomaly generators, TGF features, and three train/test settings**.

The seeded random baseline, activity-matched generator, and one-seed main comparison are now implemented. Generator diagnostics, repeated-seed analysis, ablations, and the final report remain in progress.

| Milestone | Target | Completion criteria |
| --- | --- | --- |
| ~~**M1: Reproduce the TGF baseline**~~ | ~~**16 October**~~ | ~~Run the official TGF pipeline with random injection on UCI Messages. Compare with the published results for the corresponding settings and investigate discrepancies. Record the data/code versions, preprocessing, history configuration, classifier settings, anomaly proportion, seeds, and reproduction command.~~ |
| **M2: Implement controlled injection** | **23 October** | ~~Provide random and activity-matched generators through a common interface.~~ The activity-matched generator uses strict prior-timestamp history (`t' < t`); strict equal-timestamp handling in official feature extraction, validation-based selection of the activity window/binning, and eligibility/fallback documentation remain pending. |
| **M3: Validate generator matching** | **27 October** | Compare normal edges, random anomalies, and matched anomalies using distribution plots and diagnostic statistics. Verify that endpoint-activity discrepancies shrink; report residual differences in degree, neighborhood structure, pair history, and temporal burstiness. If matching is ineffective, revise M2 before proceeding. |
| **M4: Complete the main comparison** | **2 November** | ~~Run **random -> random**, **random -> matched**, and **matched -> matched** at 1%, 5%, and 10% injection with seed 42; keep the split, feature definitions, classifier, and evaluation rules consistent; pair timestamps; report PR-AUC, ROC-AUC, recall at 1% FPR, and performance deltas.~~ **Pending:** repeat across seeds and report variability. |
| **M5: Explain the feature-level effects** | **6 November** | Define and ablate the endpoint-activity, pair-history, and neighborhood feature groups, retraining each model. Combine ablation and feature-importance evidence with M3 diagnostics to explain performance changes. Use M4 to assess how much matched-condition retraining recovers performance; limit conclusions to the tested features and classifier. |
| **M6: Finalize the report and reproducibility package** | **12 November** | Answer RQ1-RQ3 with the main results, diagnostics, ablations, uncertainty, and limitations. Include code attribution, configurations, seeds, and runnable instructions. Ensure every reported result is traceable to a saved run. Finish the report for submission by **13 November, 13:30**. |
| **M7: Prepare the presentation** | **26 November** | Finish the slides and source-code package, assign speaking sections to all six members, and rehearse within the presentation limits below. Submit the presentation order form, slides, and source code before the assigned session on **29 November, 09:00-10:30**. |

Implementation and reporting rules:

- Split the benign stream chronologically before anomaly construction. Lock generator/model configurations and thresholds using training/validation data before final testing.
- Document how equal-timestamp events are handled and whether injected anomalies enter subsequent history; use consistent rules across generators.
- Use the official TGF implementation unchanged only for the declared reproduction baseline. Independently implement the new matched generator, diagnostic checks, ablation harness, and comparative evaluation pipeline. Explicitly acknowledge any reused fragments.
- Support evaluation-bias claims with generator diagnostics, concrete performance deltas, and feature-level evidence. A performance drop is not required for a valid result; explain small or absent gaps as well.

Optional extensions: community/proximity matching, MovieLens, fixed-budget feature selection, an additional classifier, or the remaining cross-generator setting. Attempt these only after **M1-M5** are complete and the report schedule remains on track.

## Presentation Guidelines

1. **Speaking time:** Present during the assigned time slot. Every member must speak for **at least 3 minutes and at most 3.33 minutes**.
2. **Presenter names:** On **every slide** of the PPT file, include the **name of that slide's presenter in the bottom-left corner**.
3. **Q&A:** Be ready to answer questions from students and the instructor.
4. **Questions for another group:** On the presentation day, our group will be assigned another group to ask questions. **Each member must ask at least one question** to that group during the assigned time slot.

## Setup and reproduction

The UCI Messages input is included at
`data/UnipartiteNetworks/opsahl-ucsocial/ucimessages.txt`. The official TGF
feature extractor is kept as an upstream checkout, pinned to commit
`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
git clone --filter=blob:none --no-checkout https://github.com/StephanyRajeh/TGF.git TGF
git -C TGF sparse-checkout init --cone
git -C TGF sparse-checkout set 3-FeatureGeneration
git -C TGF checkout af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b
```

Run injection, feature extraction, and evaluation separately. Full commands and
the optional paper comparison command are in [`PIPELINE.md`](PIPELINE.md).

## Current seed-42 results

The full run artifacts for 1%, 5%, and 10% injection—including injection
streams, extracted features, metrics, reports, and run manifests—are checked in
under [`results/full_pipeline/`](results/full_pipeline/). Every run uses the
same 59,795-event UCI input (SHA-256 recorded in each manifest), official TGF
feature-extractor commit above, and seed 42. The project pipeline and current
core-code hashes are recorded in each run manifest.

| Injection | Configuration | Random → Random ROC-AUC | Random → Activity-matched ROC-AUC | Activity-matched → Activity-matched ROC-AUC |
|---:|---|---:|---:|---:|
| 1% | H1000 | 0.95909 | 0.81854 | 0.86980 |
| 1% | Combined H | 0.97996 | 0.88372 | 0.94365 |
| 5% | H1000 | 0.96683 | 0.82833 | 0.87499 |
| 5% | Combined H | 0.98691 | 0.89705 | 0.95354 |
| 10% | H1000 | 0.96315 | 0.84254 | 0.88323 |
| 10% | Combined H | 0.98852 | 0.90554 | 0.96882 |

The per-run `metrics_all_configurations.csv` files contain PR-AUC, ROC-AUC,
recall at 1% FPR, actual FPR, row counts, and feature counts. The results are
single-seed preliminary results. The strict timestamp-history and
train/validation selection requirements above are not fully satisfied yet.

## Presentation Guidelines

1. **Speaking time:** Every member must speak for at least 3 minutes and at most 3.33 minutes.
2. **Presenter names:** Put each slide presenter's name in the bottom-left corner of every slide.
3. **Q&A:** Prepare to answer questions from students and the instructor.
4. **Questions for another group:** Each member must ask at least one question during the assigned slot.
