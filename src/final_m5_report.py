"""Render the Final M5 tables as a Chinese report and comparison figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTING_LABELS = {
    "random_to_random": "R→R",
    "random_to_activity_matched": "R→M",
    "activity_matched_to_activity_matched": "M→M",
}
GROUP_LABELS = {
    "endpoint_activity": "endpoint activity",
    "pair_history": "pair history",
    "global_graph": "global graph",
}


def number(value, digits=4):
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def mean_sd(row, metric):
    return f"{number(getattr(row, metric + '_mean'))} ± {number(getattr(row, metric + '_std'))}"


def make_effect_plot(output: Path, ablation_summary, permutation_summary):
    rates = sorted(ablation_summary.rate.unique())
    feature_sets = list(ablation_summary.feature_set.unique())
    fig, axes = plt.subplots(len(rates), len(feature_sets) * 2,
                             figsize=(16, 4.2 * len(rates)), squeeze=False)
    groups = list(GROUP_LABELS)
    colors = ["#386cb0", "#e68a2e", "#3c9c75"]
    for row_index, rate in enumerate(rates):
        for fs_index, feature_set in enumerate(feature_sets):
            panels = (
                (
                    ablation_summary[
                        (ablation_summary.rate == rate)
                        & (ablation_summary.feature_set == feature_set)
                        & (ablation_summary.omitted_group != "none")
                    ],
                    "omitted_group",
                    "pr_auc_drop_mean",
                    "pr_auc_drop_std",
                    "Retrained ablation",
                ),
                (
                    permutation_summary[
                        (permutation_summary.rate == rate)
                        & (permutation_summary.feature_set == feature_set)
                    ],
                    "group",
                    "pr_auc_drop_seed_mean",
                    "pr_auc_drop_seed_sd",
                    "Grouped permutation",
                ),
            )
            for panel_index, (frame, key, value, error, title) in enumerate(panels):
                ax = axes[row_index, fs_index * 2 + panel_index]
                x = np.arange(len(groups))
                for setting_index, setting in enumerate(SETTING_LABELS):
                    selected = frame[frame.setting == setting].set_index(key).loc[groups]
                    ax.bar(
                        x + (setting_index - 1) * .25,
                        selected[value],
                        width=.24,
                        yerr=selected[error],
                        capsize=2,
                        color=colors[setting_index],
                        label=SETTING_LABELS[setting],
                    )
                ax.axhline(0, color="0.25", lw=.7)
                ax.set_xticks(x, [GROUP_LABELS[group].replace(" ", "\n") for group in groups])
                ax.set_ylabel("Full AP - perturbed AP")
                ax.set_title(f"{rate}% · {feature_set} · {title}")
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Final M5 feature-family effects (mean ± SD across seeds)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output / "effects.png", dpi=180)
    plt.close(fig)


def write_report(output: Path, tables):
    ablations = tables["ablations"]
    summary = tables["ablation_summary"]
    checks = tables["baseline_check"]
    prediction_checks = tables["baseline_prediction_check"]
    paired_summary = tables["paired_deltas_summary"]
    permutation_summary = tables["permutation_seed_summary"]
    make_effect_plot(output, summary, permutation_summary)

    rates = sorted(ablations.rate.unique())
    seeds = sorted(ablations.seed.unique())
    lines = [
        "# Final M5：特征组影响分析",
        "",
        "## 1. 实验结论范围",
        "",
        f"本实验使用冻结的 `strict_m2_m4_v1` 输入，覆盖注入率 {rates}、seeds {seeds}、H1000/combined-H，以及 R→R、R→M、M→M。",
        "每个上下文训练完整模型及三个删除特征组后的模型；所有模型均只使用 train 训练，并用训练条件对应的 validation 集重新校准 1% FPR 阈值。",
        "",
        f"完整模型通过冻结 M4 的跨平台漂移审计：**{bool(checks.matches.all() and prediction_checks.matches.all())}**。54 个上下文的 test event/时间/标签逐行完全一致；预测分数相关系数最低为 {number(prediction_checks.score_correlation.min(), 6)}，平均绝对分数差最大为 {number(prediction_checks.mean_absolute_score_difference.max(), 6)}，分类一致率最低为 {number(prediction_checks.prediction_agreement.min(), 6)}。",
        "冻结结果来自 macOS ARM/Python 3.9，本次 Final M5 来自 Linux x86/Python 3.11；随机森林跨平台重训并非 bitwise 相同。冻结 M4 只用于输入、协议和漂移审计，M5 的 full/ablation 配对比较全部来自同一当前环境。",
        "指标漂移使用预先固定的灾难性漂移护栏，而不是把宽容差解释为数值等价；低正例率下 recall 会随少数样本跨过阈值而离散跳变。各指标最大绝对差如下。",
        "",
        "| 指标 | 最大绝对差 | 护栏 |",
        "|---|---:|---:|",
        "",
    ]
    for metric, frame in checks.groupby("metric", sort=False):
        lines.append(
            f"| {metric} | {number(frame.difference.abs().max(), 6)} | "
            f"{number(frame.tolerance.iloc[0], 6)} |"
        )
    lines.extend([
        "",
        "正的 AP drop 表示删除/打乱该组后性能下降；负值表示性能反而上升。后者可能来自相关特征替代、噪声或有限样本，不应直接解释为该组‘有害’。",
        "",
        "## 2. 完整模型结果",
        "",
        "| 注入率 | 特征 | 设置 | AP | ROC-AUC | Recall@val-1%FPR | Test FPR |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    full = summary[summary.omitted_group == "none"]
    for row in full.itertuples():
        lines.append(
            f"| {row.rate}% | {row.feature_set} | {SETTING_LABELS[row.setting]} | "
            f"{mean_sd(row, 'pr_auc')} | {mean_sd(row, 'roc_auc')} | "
            f"{mean_sd(row, 'recall_at_validation_1pct_fpr')} | "
            f"{mean_sd(row, 'actual_test_fpr')} |"
        )

    lines.extend([
        "",
        "## 3. 重新训练消融：AP 变化",
        "",
        "下表是 `完整模型 AP − 删除该组后 AP` 的跨 seed 均值和样本标准差。",
        "",
        "| 注入率 | 特征 | 设置 | 删除组 | AP drop |",
        "|---:|---|---|---|---:|",
    ])
    for row in summary[summary.omitted_group != "none"].itertuples():
        lines.append(
            f"| {row.rate}% | {row.feature_set} | {SETTING_LABELS[row.setting]} | "
            f"{GROUP_LABELS[row.omitted_group]} | {mean_sd(row, 'pr_auc_drop')} |"
        )

    lines.extend([
        "",
        "## 4. Grouped permutation importance",
        "",
        "完整模型保持不变，同一组的全部列（包括 combined-H 的所有窗口）使用相同的 test-row 排列。先在每个 seed 内对 5 次 shuffle 求均值，再跨 seed 汇总。",
        "",
        "| 注入率 | 特征 | 设置 | 打乱组 | AP drop |",
        "|---:|---|---|---|---:|",
    ])
    for row in permutation_summary.itertuples():
        lines.append(
            f"| {row.rate}% | {row.feature_set} | {SETTING_LABELS[row.setting]} | "
            f"{GROUP_LABELS[row.group]} | {number(row.pr_auc_drop_seed_mean)} ± "
            f"{number(row.pr_auc_drop_seed_sd)} |"
        )

    lines.extend([
        "",
        "## 5. 跨条件变化与 matched retraining recovery",
        "",
        "`R→M − R→R` 衡量跨生成器性能变化；`M→M − R→M` 衡量在 matched 条件重新训练后的恢复。",
        "",
        "| 注入率 | 特征 | 变体 | AP 跨条件变化 | AP retraining recovery |",
        "|---:|---|---|---:|---:|",
    ])
    for row in paired_summary.itertuples():
        lines.append(
            f"| {row.rate}% | {row.feature_set} | {row.omitted_group} | "
            f"{number(row.pr_auc_cross_minus_random_mean)} ± {number(row.pr_auc_cross_minus_random_std)} | "
            f"{number(row.pr_auc_retraining_recovery_mean)} ± {number(row.pr_auc_retraining_recovery_std)} |"
        )

    lines.extend([
        "",
        "## 6. 如何解读这些证据",
        "",
        "- Retrained ablation 询问：没有某组特征时，其余特征重新学习后还能达到什么性能。",
        "- Grouped permutation 询问：已经训练好的完整模型当前依赖某组到什么程度。",
        "- MDI 用于查看组内具体列，但会偏向高基数特征，并在相关特征之间分摊重要性，因此不作为主证据。",
        "- 三个特征组并不统计独立；endpoint rank 与 pair rank 本身也依赖全图分布。",
        "- `global_graph` 不是 common-neighbor/Jaccard neighborhood 特征，本报告不会把它改名为 neighborhood。",
        "",
        "## 7. 尚未完成的机制解释",
        "",
        "M3 的完整 diagnostics 尚未冻结，因此当前结果只能说明特征组与性能变化的关联。关于 neighborhood overlap、prior-pair residual、degree residual 和 temporal burstiness 的解释仍为 pending，不能从消融结果单独推出因果结论。",
        "",
        "## 8. 局限",
        "",
        "结果仅覆盖 UCI Messages、固定 TGF-style 特征、固定 Random Forest 和三个 injection/model seeds。Permutation repeats 表示 shuffle 波动，不是新的独立数据集。所有结论应限制在当前实验范围内。",
        "",
        "![Final M5 effects](effects.png)",
        "",
    ])
    (output / "FINAL_M5_REPORT_CN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import sys

    destination = Path(sys.argv[1])
    names = (
        "ablations",
        "ablation_summary",
        "baseline_check",
        "baseline_prediction_check",
        "paired_deltas_summary",
        "permutation_seed_summary",
    )
    write_report(destination, {name: pd.read_csv(destination / f"{name}.csv") for name in names})
