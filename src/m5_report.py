"""Render saved M5 tables without retraining models."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"random_to_random": "R → R", "random_to_activity_matched": "R → M",
          "activity_matched_to_activity_matched": "M → M"}


def write_report(out, tables):
    ablations, permutation = tables["ablations"], tables["permutation"]
    summary = permutation.groupby(["feature_set", "setting", "group"]).pr_auc_drop.agg(["mean", "std"]).reset_index()
    summary.to_csv(out / "permutation_summary.csv", index=False)
    lines = ["# M5: fixed-data feature analysis", "",
             "Existing seed-42 injection data; feature groups are a local proposal. No generator changes.", "",
             "PR-AUC below means average precision, following the existing M4 implementation.", "",
             "## Retrained ablations", "",
             "Positive drop means removing a group reduced performance; negative means performance increased.", "",
             "| Features | Setting | Removed group | PR-AUC | PR drop | ROC-AUC | ROC drop |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in ablations.itertuples():
        lines.append(f"| {r.feature_set} | {LABELS[r.setting]} | {r.omitted_group} | {r.pr_auc:.6f} | {r.pr_auc_drop:+.6f} | {r.roc_auc:.6f} | {r.roc_auc_drop:+.6f} |")
    lines.extend(["", "## Group permutation importance", "",
                  "The full model is held fixed. All columns in a group share one row permutation, across H windows as well.",
                  "Error bars are standard deviations over shuffles, not across independently generated datasets.", "",
                  "| Features | Setting | Group | Mean PR drop | Shuffle SD |",
                  "|---|---|---|---:|---:|"])
    for r in summary.itertuples():
        lines.append(f"| {r.feature_set} | {LABELS[r.setting]} | {r.group} | {r.mean:+.6f} | {r.std:.6f} |")
    matched = tables["baseline_check"].matches.all()
    lines.extend(["", "## Baseline check", "", f"All full-model metrics match saved M4 within 1e-10: **{bool(matched)}**.", "",
                  "## Interpretation boundaries", "",
                  "- Ablation measures the effect of retraining without a family; permutation measures dependence of the fitted full model. They need not agree.",
                  "- Correlated families can substitute for one another. Neither analysis proves a causal mechanism.",
                  "- The global_graph group is not a neighborhood-overlap group; no such predictor exists in these tables.",
                  "- One injection seed is available. No population uncertainty or independent-seed robustness is claimed.",
                  "- Recall follows M4's test-negative quantile threshold, including ties. This is descriptive, not a validation-calibrated operating threshold; actual FPR is in ablations.csv.",
                  "- Existing row-based 70/30 splits and history conventions are retained for comparability, pending M2/M4 alignment.",
                  "- M3 diagnostics are needed to link model behavior to the generator's residual distribution differences.", "",
                  "![Ablation and grouped permutation](effects.png)", ""])
    (out / "REPORT.md").write_text("\n".join(lines))
    feature_sets = list(ablations.feature_set.unique())
    fig, axes = plt.subplots(len(feature_sets), 2, figsize=(13, 4.4 * len(feature_sets)), squeeze=False)
    colors = ["#386cb0", "#e68a2e", "#3c9c75"]
    for i, fs in enumerate(feature_sets):
        for j, (frame, value, title) in enumerate(((ablations[(ablations.feature_set == fs) & (ablations.omitted_group != "none")], "pr_auc_drop", "Retrained ablation"),
                                                  (summary[summary.feature_set == fs], "mean", "Grouped permutation"))):
            ax = axes[i, j]
            key = "omitted_group" if j == 0 else "group"
            groups = list(ablations.loc[ablations.omitted_group != "none", "omitted_group"].unique())
            x = np.arange(len(groups))
            for k, setting in enumerate(LABELS):
                rows = frame[frame.setting == setting].set_index(key).loc[groups]
                ax.bar(x + (k - 1) * .25, rows[value], width=.24, color=colors[k], label=LABELS[setting],
                       yerr=rows["std"] if j else None, capsize=2)
            ax.axhline(0, color="0.3", lw=.7)
            ax.set_xticks(x, [g.replace("_", "\n") for g in groups])
            ax.set_ylabel("Full-model AP minus perturbed AP")
            ax.set_title(f"{fs}: {title}")
            ax.legend(frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(out.name.replace("_", " ") + " · fixed injection data", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(out / "effects.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1])
    write_report(out, {name: pd.read_csv(out / f"{name}.csv") for name in
                       ("ablations", "permutation", "baseline_check")})
