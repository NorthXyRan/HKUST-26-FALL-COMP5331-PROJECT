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

Start the common history interface and activity-matched generator alongside baseline reproduction. Validate the generator before running the full comparison and feature-group ablation.

| Milestone | Target | Completion criteria |
| --- | --- | --- |
| **M1: Reproduce the TGF baseline** | **16 October** | Run the official TGF pipeline with random injection on UCI Messages. Compare with the published results for the corresponding settings and investigate discrepancies. Record the data/code versions, preprocessing, history configuration, classifier settings, anomaly proportion, seeds, and reproduction command. |
| **M2: Implement controlled injection** | **23 October** | Provide random and activity-matched generators through a common interface. Enforce history with timestamps **< t** for both generation and feature extraction. Select the activity statistic, history window, and binning using training/validation data; document eligibility constraints, candidate coverage, and any fallback behavior. |
| **M3: Validate generator matching** | **27 October** | Compare normal edges, random anomalies, and matched anomalies using distribution plots and diagnostic statistics. Verify that endpoint-activity discrepancies shrink; report residual differences in degree, neighborhood structure, pair history, and temporal burstiness. If matching is ineffective, revise M2 before proceeding. |
| **M4: Complete the main comparison** | **2 November** | Run **random -> random**, **random -> matched**, and **matched -> matched** training/testing. Keep splits, feature definitions, classifier family, and evaluation rules consistent; pair timestamps, anomaly proportions, and seed schedules where feasible. Report PR-AUC, ROC-AUC, recall at a pre-specified low false-positive rate, performance deltas, and variability across repeated seeds. |
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
