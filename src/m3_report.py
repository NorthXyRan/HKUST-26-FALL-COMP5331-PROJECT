"""M3 plots and data-derived report; all inference is descriptive across seeds."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"random": "#d47b27", "activity_matched": "#167c80"}


def report(output, rows=None):
    output = Path(output)
    if rows is None:
        rows = pd.concat(
            [pd.read_csv(p) for p in sorted((output / "rows").glob("*.csv.gz"))], ignore_index=True
        )
    effects = pd.read_csv(output / "paired_effects.csv")
    distances = pd.read_csv(output / "distances.csv")
    distribution = pd.read_csv(output / "distributions.csv")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150}
    )

    def save(fig, name):
        fig.savefig(figures / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    primary = effects.query("split=='test' and window==1000 and history_policy=='observed'")
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy, offset in [("random", -0.07), ("activity_matched", 0.07)]:
        g = primary.groupby("rate")[strategy]
        ax.errorbar(
            np.arange(3) + offset,
            g.mean(),
            yerr=g.std(),
            fmt="o-",
            capsize=4,
            color=COLORS[strategy],
            label=strategy,
        )
    ax.set(
        xticks=range(3),
        xticklabels=["1%", "5%", "10%"],
        xlabel="Injection rate",
        ylabel="Endpoint log-activity error",
        title="A1 | Test matching error: mean +/- SD of 3 injection seeds",
    )
    ax.legend()
    save(fig, "A1_activity_error")
    sample = rows.query(
        "split=='test' and window==1000 and history_policy=='observed' and rate==5 and seed==42"
    )

    def ecdf(ax, metric):
        for strategy in COLORS:
            for kind in ("normal_reference", "anomaly"):
                a = (
                    sample[(sample.strategy == strategy) & (sample.kind == kind)][metric]
                    .dropna()
                    .sort_values()
                    .to_numpy()
                )
                ax.step(
                    a,
                    np.arange(1, len(a) + 1) / len(a),
                    where="post",
                    color=COLORS[strategy],
                    ls="--" if kind == "normal_reference" else "-",
                    label=strategy + " " + kind,
                )
        ax.set(xlabel=metric, ylabel="ECDF")
        if metric not in ("jaccard", "concentration_1h_24h", "burstiness"):
            ax.set_xscale("symlog", linthresh=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m in zip(axes, ["activity_low", "activity_high"]):
        ecdf(ax, m)
    axes[0].legend(fontsize=6)
    fig.suptitle("A2 | Activity ECDFs: test, 5%, seed 42 (illustrative)")
    fig.tight_layout()
    save(fig, "A2_activity_ecdf")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, m in zip(
        axes.flat,
        [
            "degree_mean",
            "common_neighbors",
            "jaccard",
            "pair_count",
            "recency_minutes",
            "burstiness",
        ],
    ):
        ecdf(ax, m)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("B1 | Residual ECDFs: test, 5%, seed 42 (illustrative)")
    fig.tight_layout()
    save(fig, "B1_residual_ecdf")
    metrics = [
        "activity_low",
        "activity_high",
        "degree_mean",
        "common_neighbors",
        "jaccard",
        "pair_count",
        "concentration_1h_24h",
        "recency_minutes",
        "burstiness",
    ]
    dist = distances.query("split=='test' and window==1000 and history_policy=='observed'")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, rate in zip(axes, [1, 5, 10]):
        a = (
            dist[dist.rate == rate]
            .groupby(["metric", "strategy"])
            .ks.mean()
            .unstack()
            .reindex(metrics)[["random", "activity_matched"]]
        )
        ax.imshow(a, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set(
            xticks=[0, 1],
            xticklabels=["Random", "Matched"],
            yticks=range(len(metrics)),
            yticklabels=metrics,
            title=f"{rate}%",
        )
        for i in range(len(a)):
            for j in range(2):
                ax.text(j, i, f"{a.iloc[i,j]:.2f}", ha="center", va="center")
    fig.suptitle("B2 | KS distance to own-history reference: mean of 3 seeds, test")
    fig.tight_layout()
    save(fig, "B2_ks_distances")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    test = effects.query("split=='test'")
    for ax, rate in zip(axes, [1, 5, 10]):
        for policy, style in [("observed", "-"), ("benign_only", "--")]:
            g = (
                test[(test.rate == rate) & (test.history_policy == policy)]
                .groupby("window")
                .reduction_pct
            )
            ax.errorbar(
                g.mean().index, g.mean(), yerr=g.std(), fmt="o" + style, capsize=3, label=policy
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set(xscale="log", xlabel="Diagnostic event window", title=f"{rate}%")
    axes[0].set_ylabel("Activity error reduction (%)")
    axes[0].legend()
    fig.suptitle("A3 | Fixed-generator sensitivity: mean +/- SD, test")
    fig.tight_layout()
    save(fig, "A3_window_history_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["burstiness", "concentration_1h_24h"]):
        sub = dist[dist.metric == metric]
        for strategy in COLORS:
            g = sub[sub.strategy == strategy].groupby("rate").ks
            ax.errorbar(
                g.mean().index,
                g.mean(),
                yerr=g.std(),
                fmt="o-",
                label=strategy,
                color=COLORS[strategy],
                capsize=4,
            )
        ax.set(xlabel="Injection rate (%)", ylabel="KS distance", title=metric, xticks=[1, 5, 10])
        ax.legend()
    fig.suptitle("B3 | Timing diagnostics: mean +/- SD of injection seeds, test")
    fig.tight_layout()
    save(fig, "B3_timing")
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy in COLORS:
        z = effects.query("window==1000 and history_policy=='observed'").groupby("split")[strategy]
        order = ["train", "validation", "test"]
        ax.errorbar(
            range(3),
            z.mean().reindex(order),
            yerr=z.std().reindex(order),
            fmt="o-",
            capsize=4,
            label=strategy,
            color=COLORS[strategy],
        )
    ax.set(
        xticks=range(3),
        xticklabels=order,
        ylabel="Endpoint log-activity error",
        title="A4 | Split context: mean +/- SD across 9 rate/seed cells",
    )
    ax.legend()
    save(fig, "A4_split_context")
    lines = [
        "# M3 results: matching and residual differences",
        "",
        "Generated from frozen strict M2/M4 streams. Primary: test split, observed history, 1000-event window.",
        "All 18 conditions replayed; no model training, reinjection or test-based parameter selection.",
        "",
        "## Member A: activity matching",
        "",
        "| Rate | Random mean | Matched mean | Reduction mean ± seed SD | Reduction range |",
        "|---|---:|---:|---:|---:|",
    ]
    for rate, g in primary.groupby("rate"):
        lines.append(
            f"| {rate}% | {g.random.mean():.4f} | {g.activity_matched.mean():.4f} | {g.reduction_pct.mean():.2f} ± {g.reduction_pct.std():.2f}% | {g.reduction_pct.min():.2f}–{g.reduction_pct.max():.2f}% |"
        )
    lines += [
        "",
        "![A1](figures/A1_activity_error.png)",
        "![A2](figures/A2_activity_ecdf.png)",
        "",
        "## Member A: sensitivity and common-history control",
        "",
        "Sensitivity changes only the diagnostic window, not the generator. Benign-only is a counterfactual rescore of the same edges, not a new generator experiment.",
        "",
        "| Window | Observed reduction (9-cell mean) | Common benign-history reduction |",
        "|---|---:|---:|",
    ]
    for window in sorted(test.window.unique()):
        g = test[test.window == window].groupby("history_policy").reduction_pct.mean()
        lines.append(f"| {window} | {g.observed:.2f}% | {g.benign_only:.2f}% |")
    lines += [
        "",
        "![A3](figures/A3_window_history_sensitivity.png)",
        "![A4](figures/A4_split_context.png)",
        "",
        "The split figure mixes rate and seed variation (9 cells); it is descriptive drift context, not a confidence interval.",
        "",
        "## Member B: residual structure and timing",
        "",
        "| Metric | Random KS | Matched KS |",
        "|---|---:|---:|",
    ]
    for m in metrics:
        g = dist[dist.metric == m].groupby("strategy").ks.mean()
        lines.append(f"| {m} | {g.random:.4f} | {g.activity_matched:.4f} |")
    lines += [
        "",
        "![B1](figures/B1_residual_ecdf.png)",
        "![B2](figures/B2_ks_distances.png)",
        "![B3](figures/B3_timing.png)",
        "",
        "Distances above average the nine rate/seed cells. ECDF panels show only 5%, seed 42. Each anomaly group has its own-history normal reference; the dashed curves are intentionally distinct.",
        "",
        "### Missingness and novelty",
        "",
    ]
    primary_distribution = distribution.query(
        "split=='test' and window==1000 and history_policy=='observed'"
    )
    for strategy in COLORS:
        g = primary_distribution[
            (primary_distribution.strategy == strategy)
            & (primary_distribution.metric == "pair_count")
            & (primary_distribution.kind == "normal_reference")
        ]
        lines.append(
            f"- {strategy} normal reference repeat-pair fraction: {(1-g.zero_fraction).mean():.2%}; injected pairs: 0% by construction."
        )
        for kind in ("normal_reference", "anomaly"):
            b = primary_distribution[
                (primary_distribution.strategy == strategy)
                & (primary_distribution.metric == "burstiness")
                & (primary_distribution.kind == kind)
            ]
            lines.append(
                f"- {strategy} / {kind} burstiness missing fraction: {b.missing_fraction.mean():.2%} (mean of nine cells). Distances exclude missing values; do not read them as population-wide equivalence."
            )
    lines += [
        "",
        "## Joint interpretation and M5 handoff",
        "",
        "Activity matching improves the intended endpoint statistic, but residual differences remain. Novel-pair rules leave a large pair-history discrepancy; timing and degree are not fully matched. These are diagnostic associations, not evidence that a classifier uses a given shortcut.",
        "The current remote M5 package uses legacy `results/full_pipeline` and its 70/30/current-event-inclusive protocol. This M3 uses `results/final_m2_m4` and strict pre-timestamp features with 60/20/20 splits. Do not join their scores or claim a controlled feature-level explanation until M5 is rerun on the strict package.",
        "Common neighbors/Jaccard and timing diagnostics are not predictor columns in the current TGF-style classifier. Recommend a strict-protocol pair-history ablation and endpoint-activity ablation; do not rename global_graph as neighborhood.",
        "",
        "## Limits",
        "",
        "Three injection seeds on one dataset; temporally dependent samples; sampled reference edges are not a uniform sample of all normal edges. SD is descriptive, not a confidence interval. No p-values are reported. Endpoint burstiness uses the last at most 20 intervals with >=5 required, population SD; zero-interval/all-zero cases are explicitly handled. Concentration is a separate 1h/24h proxy. Window/history controls are post-hoc descriptive checks and cannot tune M2 on test data.",
        "",
        "Reproduction and ownership: [docs/M3.md](../../docs/M3.md). Tables retain every split, window, policy and seed. Input/source/environment/output hashes: [manifest.json](manifest.json).",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
