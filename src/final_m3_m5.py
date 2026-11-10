"""Build and verify the strict-protocol M3--M5 integration package."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from .final_m5_verify import verify as verify_m5
from .m3_verify import verify as verify_m3


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = {"split": "test", "history_policy": "observed", "window": 1000}
RATES = (1, 5, 10)
SEEDS = (42, 43, 44)
FEATURE_SETS = ("H1000", "combined_H")
SETTINGS = (
    "random_to_random",
    "random_to_activity_matched",
    "activity_matched_to_activity_matched",
)
GROUPS = ("endpoint_activity", "pair_history", "global_graph")
KS_METRICS = (
    "degree_mean",
    "common_neighbors",
    "jaccard",
    "pair_count",
    "recency_minutes",
    "burstiness",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path):
    return json.loads(path.read_text())


def component_contract(m3_root: Path, m5_root: Path) -> dict:
    """Validate that M3 and M5 are genuinely comparable."""
    m3 = read_json(m3_root / "manifest.json")
    m5 = read_json(m5_root / "manifest.json")
    protocol_path = ROOT / "results/final_m2_m4/protocol.json"
    base_manifest_path = ROOT / "results/final_m2_m4/manifest.json"
    protocol = read_json(protocol_path)

    shared = {
        "results/final_m2_m4/protocol.json": digest(protocol_path),
        "results/final_m2_m4/manifest.json": digest(base_manifest_path),
    }
    for name, expected in shared.items():
        require(m3["input_sha256"].get(name) == expected, f"M3 base hash mismatch: {name}")
        require(m5["input_sha256"].get(name) == expected, f"M5 base hash mismatch: {name}")

    require(protocol["protocol_version"] == "strict_m2_m4_v1", "unexpected base protocol")
    require(tuple(protocol["rates_percent_of_original"]) == RATES, "base rates differ")
    require(tuple(protocol["seeds"]) == SEEDS, "base seeds differ")
    require(tuple(m5["rates"]) == RATES, "M5 rates differ")
    require(tuple(m5["seeds"]) == SEEDS, "M5 seeds differ")
    require(tuple(m5["feature_sets"]) == FEATURE_SETS, "M5 feature sets differ")
    require(set(m5["groups"]) == set(GROUPS), "M5 groups differ")
    require(m3["configuration"]["primary_split"] == PRIMARY["split"], "M3 split differs")
    require(
        m3["configuration"]["primary_window"] == PRIMARY["window"],
        "M3 primary window differs",
    )
    require(
        PRIMARY["history_policy"] in m3["configuration"]["history_policies"],
        "M3 observed history is missing",
    )
    require(
        m3["configuration"]["seed_variation"] == "injection only; no model training",
        "M3 seed semantics changed",
    )
    require(m5["permutation_repeats"] == 5, "M5 permutation repeat count differs")
    require(m5["permutation_seed"] == 31415, "M5 permutation seed differs")

    graph_names = m5["groups"]["global_graph"]
    require(
        not any(token in name for name in graph_names for token in ("neighbor", "jaccard", "common")),
        "global_graph must not be relabeled as a neighborhood group",
    )
    return {
        "protocol": protocol["protocol_version"],
        "protocol_sha256": shared["results/final_m2_m4/protocol.json"],
        "base_manifest_sha256": shared["results/final_m2_m4/manifest.json"],
        "rates": list(RATES),
        "injection_seeds": list(SEEDS),
        "classifier_seed_policy": "same numeric seed as each frozen injection run; not independently crossed",
        "permutation_repeats": m5["permutation_repeats"],
        "permutation_seed": m5["permutation_seed"],
    }


def _primary(frame: pd.DataFrame) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for name, value in PRIMARY.items():
        mask &= frame[name].eq(value)
    return frame.loc[mask].copy()


def build_joint_evidence(m3_root: Path, m5_root: Path) -> pd.DataFrame:
    """Join M3 diagnostics and M5 effects on rate/injection seed/feature set."""
    paired = _primary(pd.read_csv(m3_root / "paired_effects.csv"))
    require(len(paired) == len(RATES) * len(SEEDS), "incomplete M3 primary pairing")
    paired = paired.rename(
        columns={
            "random": "m3_activity_error_random",
            "activity_matched": "m3_activity_error_matched",
            "matched_minus_random": "m3_activity_error_matched_minus_random",
            "reduction_pct": "m3_activity_error_reduction_pct",
        }
    )[
        [
            "rate",
            "seed",
            "m3_activity_error_random",
            "m3_activity_error_matched",
            "m3_activity_error_matched_minus_random",
            "m3_activity_error_reduction_pct",
        ]
    ]

    distances = _primary(pd.read_csv(m3_root / "distances.csv"))
    distances = distances[distances.metric.isin(KS_METRICS)]
    distance_wide = distances.pivot(
        index=["rate", "seed"], columns=["strategy", "metric"], values="ks"
    )
    expected_distance_columns = {(strategy, metric) for strategy in ("random", "activity_matched") for metric in KS_METRICS}
    require(set(distance_wide.columns) == expected_distance_columns, "incomplete M3 KS matrix")
    distance_wide.columns = [f"m3_ks_{strategy}_{metric}" for strategy, metric in distance_wide.columns]
    distance_wide = distance_wide.reset_index()

    ablations = pd.read_csv(m5_root / "ablations.csv")
    ablations = ablations[ablations.omitted_group.isin(GROUPS)]
    ablation_wide = ablations.pivot(
        index=["rate", "seed", "feature_set"],
        columns=["setting", "omitted_group"],
        values="pr_auc_drop",
    )
    expected_ablation_columns = {(setting, group) for setting in SETTINGS for group in GROUPS}
    require(set(ablation_wide.columns) == expected_ablation_columns, "incomplete M5 ablation matrix")
    ablation_wide.columns = [f"m5_ap_drop_{setting}_{group}" for setting, group in ablation_wide.columns]
    evidence = ablation_wide.reset_index().merge(paired, on=["rate", "seed"], validate="many_to_one")
    evidence = evidence.merge(distance_wide, on=["rate", "seed"], validate="many_to_one")
    evidence = evidence.sort_values(["rate", "seed", "feature_set"]).reset_index(drop=True)

    require(len(evidence) == len(RATES) * len(SEEDS) * len(FEATURE_SETS), "incomplete joint matrix")
    require(not evidence.isna().any().any(), "joint matrix contains missing values")
    require(evidence.m3_activity_error_reduction_pct.gt(0).all(), "activity matching failed")
    for metric in ("degree_mean", "common_neighbors", "jaccard"):
        require(
            evidence[f"m3_ks_activity_matched_{metric}"].lt(
                evidence[f"m3_ks_random_{metric}"]
            ).all(),
            f"matched {metric} KS did not improve in every rate/seed",
        )
    return evidence


def summarize(evidence: pd.DataFrame, m3_root: Path, m5_root: Path) -> dict:
    primary = evidence.drop_duplicates(["rate", "seed"])
    m3_metrics = {}
    for metric in KS_METRICS:
        for strategy in ("random", "activity_matched"):
            name = f"m3_ks_{strategy}_{metric}"
            m3_metrics[name] = float(primary[name].mean())

    m5_effects = {}
    for setting in SETTINGS:
        m5_effects[setting] = {}
        for group in GROUPS:
            name = f"m5_ap_drop_{setting}_{group}"
            m5_effects[setting][group] = {
                "mean": float(evidence[name].mean()),
                "sample_sd": float(evidence[name].std(ddof=1)),
                "min": float(evidence[name].min()),
                "max": float(evidence[name].max()),
            }

    distributions = _primary(pd.read_csv(m3_root / "distributions.csv"))
    anomaly_pair = distributions[
        (distributions.kind == "anomaly") & (distributions.metric == "pair_count")
    ]
    reference_pair = distributions[
        (distributions.kind == "normal_reference") & (distributions.metric == "pair_count")
    ]
    m5_manifest = read_json(m5_root / "manifest.json")
    return {
        "protocol": "final_m3_m5_integration_v1",
        "primary_diagnostic": PRIMARY,
        "counts": {
            "joint_rows": len(evidence),
            "m3_rate_seed_pairs": len(primary),
            "m5_ablations": m5_manifest["counts"]["ablations"],
            "m5_permutation_measurements": m5_manifest["counts"]["permutation"],
            "m5_mdi_rows": m5_manifest["counts"]["mdi"],
        },
        "m3": {
            "activity_error_reduction_pct_mean": float(primary.m3_activity_error_reduction_pct.mean()),
            "activity_error_reduction_pct_sample_sd": float(primary.m3_activity_error_reduction_pct.std(ddof=1)),
            "activity_error_reduction_pct_min": float(primary.m3_activity_error_reduction_pct.min()),
            "activity_error_reduction_pct_max": float(primary.m3_activity_error_reduction_pct.max()),
            **m3_metrics,
            "anomaly_pair_count_zero_fraction_mean": float(anomaly_pair.zero_fraction.mean()),
            "normal_reference_pair_count_zero_fraction_mean": float(reference_pair.zero_fraction.mean()),
        },
        "m5_pr_auc_drop": m5_effects,
        "randomness_accounting": {
            "injection_seed_variation": list(SEEDS),
            "m3": "injection-seed variation only; no classifier",
            "m5_classifier": "random_state equals the frozen run seed; classifier seed is not independently crossed with injection seed",
            "permutation": "five shuffles per context from permutation seed 31415; shuffles are not independent datasets",
        },
    }


def _number(value: float) -> str:
    return f"{value:.4f}"


def make_report(summary: dict) -> str:
    m3 = summary["m3"]
    effects = summary["m5_pr_auc_drop"]
    lines = [
        "# Final M3–M5 联合测试与研究结论",
        "",
        "## 1. 联合测试结论",
        "",
        "M3 与 Final M5 使用同一个 `strict_m2_m4_v1` 冻结输入包，协议和基础 manifest 的 SHA-256 完全一致；注入率、注入 seed、历史定义和数据切分均可对齐。因此，当前 M3 诊断可以与 Final M5 消融作联合解释。旧版 `results/full_pipeline` 和 `results/m5` 仍只作为历史产物。",
        "",
        f"本联合包对齐了 {summary['counts']['m3_rate_seed_pairs']} 个 rate/seed 诊断配置与两个特征集合，共 {summary['counts']['joint_rows']} 行证据；同时复核 {summary['counts']['m5_ablations']} 条消融、{summary['counts']['m5_permutation_measurements']} 条置换测量和 {summary['counts']['m5_mdi_rows']} 条 MDI 记录。",
        "",
        "## 2. M3 告诉我们：匹配是否真的有效",
        "",
        f"在 test、observed history、1,000-event window 的预先声明主分析中，activity matching 在九个 rate/seed 配置中都降低了端点活动误差，平均降低 {_number(m3['activity_error_reduction_pct_mean'])}%（范围 {_number(m3['activity_error_reduction_pct_min'])}%–{_number(m3['activity_error_reduction_pct_max'])}%）。",
        "",
        "平均 KS 距离也显示匹配后的结构更接近对应正常边：",
        "",
        "| 诊断 | Random | Activity-matched |",
        "| --- | ---: | ---: |",
        f"| degree mean | {_number(m3['m3_ks_random_degree_mean'])} | {_number(m3['m3_ks_activity_matched_degree_mean'])} |",
        f"| common neighbors | {_number(m3['m3_ks_random_common_neighbors'])} | {_number(m3['m3_ks_activity_matched_common_neighbors'])} |",
        f"| Jaccard | {_number(m3['m3_ks_random_jaccard'])} | {_number(m3['m3_ks_activity_matched_jaccard'])} |",
        f"| recency | {_number(m3['m3_ks_random_recency_minutes'])} | {_number(m3['m3_ks_activity_matched_recency_minutes'])} |",
        f"| burstiness（仅定义样本） | {_number(m3['m3_ks_random_burstiness'])} | {_number(m3['m3_ks_activity_matched_burstiness'])} |",
        f"| pair count | {_number(m3['m3_ks_random_pair_count'])} | {_number(m3['m3_ks_activity_matched_pair_count'])} |",
        "",
        "`pair_count` 几乎没有改善：两种生成器都强制异常边是从未出现过的新边，因此异常的历史 pair count 全为 0，而对应正常 reference edge 通常有重复历史。这是活动匹配没有消除的明显残余差异。",
        "",
        "## 3. M5 告诉我们：模型实际用了什么",
        "",
        "下表给出完整模型 AP 减去删除特征组后 AP 的均值；正数越大，说明重新训练后缺少该组造成的损失越大。均值覆盖 3 个注入率、3 个 seed 和 2 个特征集合。",
        "",
        "| 设置 | Endpoint activity | Pair history | Global graph |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "random_to_random": "R→R",
        "random_to_activity_matched": "R→M",
        "activity_matched_to_activity_matched": "M→M",
    }
    for setting in SETTINGS:
        lines.append(
            f"| {labels[setting]} | {_number(effects[setting]['endpoint_activity']['mean'])} | "
            f"{_number(effects[setting]['pair_history']['mean'])} | "
            f"{_number(effects[setting]['global_graph']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "联合起来可以得到较稳妥的解释：随机异常留下很强的端点活动捷径，所以 R→R 高度依赖 endpoint activity；活动匹配削弱了该捷径，R→M 中 endpoint 组的平均作用明显变小。由于所有异常仍是历史新边，pair-history 残差没有被匹配掉，R→M 和 M→M 对 pair-history 的依赖更强。",
            "",
            "这不是完整的因果证明。消融组 `global_graph` 是窗口级全图统计，不能改名为 neighborhood 组；M3 的 common-neighbor、Jaccard 和 burstiness 也不是当前分类器的预测列。因此只能说这些残差存在，不能声称模型直接使用了它们。",
            "",
            "## 4. 随机性与统计范围",
            "",
            "- M3 的 seeds 42、43、44 只代表注入 seed 变化，不训练分类器。",
            "- M5 在每个冻结注入 run 上令 Random Forest `random_state` 使用同一个数值 seed；没有把注入 seed 与独立分类器 seed 做完整交叉实验。",
            "- grouped permutation 每个上下文做 5 次 shuffle，固定 permutation seed 31415；这些 shuffle 只刻画置换波动，不是新的数据集或独立注入重复。",
            "",
            "所以当前的跨-seed 波动混合了注入样本变化和与之绑定的模型随机性。若最终报告需要把两者分别估计，仍需额外运行 injection-seed × classifier-seed 的交叉实验；这属于统计增强，不影响当前预先声明的 M5 完成标准。",
            "",
            "## 5. 完成范围",
            "",
            "M3–M5 的核心实验、联合协议检查、结果哈希、完整矩阵和机制边界均已覆盖。项目整体仍需 M6 最终报告整合与 M7 展示材料；单数据集、单分类器、无真实异常外部验证仍是研究限制，而不是已经通过的测试。",
            "",
            "机器可读的逐 rate/seed/feature-set 证据见 `joint_evidence.csv`，汇总与随机性说明见 `joint_summary.json`，输入、源码、依赖和输出哈希见 `manifest.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def input_hashes(m3_root: Path, m5_root: Path) -> dict:
    paths = (
        m3_root / "manifest.json",
        m3_root / "paired_effects.csv",
        m3_root / "distances.csv",
        m3_root / "distributions.csv",
        m5_root / "manifest.json",
        m5_root / "ablations.csv",
        m5_root / "permutation_shuffle_summary.csv",
        ROOT / "results/final_m2_m4/protocol.json",
        ROOT / "results/final_m2_m4/manifest.json",
    )
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def generate(output: Path, m3_root: Path, m5_root: Path) -> dict:
    require(not output.exists(), f"refusing to overwrite existing output: {output}")
    contract = component_contract(m3_root, m5_root)
    component_results = {
        "m3": verify_m3(m3_root),
        "final_m5": verify_m5(m5_root),
    }
    evidence = build_joint_evidence(m3_root, m5_root)
    summary = summarize(evidence, m3_root, m5_root)

    output.mkdir(parents=True)
    evidence.to_csv(output / "joint_evidence.csv", index=False)
    (output / "joint_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    (output / "FINAL_M3_M5_REPORT_CN.md").write_text(make_report(summary))
    outputs = {
        name: digest(output / name)
        for name in ("joint_evidence.csv", "joint_summary.json", "FINAL_M3_M5_REPORT_CN.md")
    }
    manifest = {
        "protocol_version": "final_m3_m5_integration_v1",
        "component_contract": contract,
        "component_verifier_results": component_results,
        "input_sha256": input_hashes(m3_root, m5_root),
        "source_sha256": {"src/final_m3_m5.py": digest(ROOT / "src/final_m3_m5.py")},
        "output_sha256": outputs,
        "python": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "counts": summary["counts"],
        "randomness_accounting": summary["randomness_accounting"],
        "scope": "Post-hoc integration of immutable M3 and Final M5 strict-protocol results; no model refit.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def verify(output: Path, check_components: bool = True) -> dict:
    manifest = read_json(output / "manifest.json")
    require(manifest["protocol_version"] == "final_m3_m5_integration_v1", "bad integration protocol")
    for name, expected in manifest["input_sha256"].items():
        require(digest(ROOT / name) == expected, f"input hash mismatch: {name}")
    for name, expected in manifest["source_sha256"].items():
        require(digest(ROOT / name) == expected, f"source hash mismatch: {name}")
    actual_outputs = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    require(actual_outputs == set(manifest["output_sha256"]), "output file set differs")
    for name, expected in manifest["output_sha256"].items():
        require(digest(output / name) == expected, f"output hash mismatch: {name}")

    m3_root = ROOT / "results/m3"
    m5_root = ROOT / "results/final_m5"
    contract = component_contract(m3_root, m5_root)
    require(contract == manifest["component_contract"], "component contract changed")
    expected = build_joint_evidence(m3_root, m5_root)
    saved = pd.read_csv(output / "joint_evidence.csv")
    pd.testing.assert_frame_equal(saved, expected, check_exact=False, rtol=1e-12, atol=1e-14)
    require(
        read_json(output / "joint_summary.json") == summarize(expected, m3_root, m5_root),
        "joint summary changed",
    )
    if check_components:
        require(verify_m3(m3_root)["status"] == "passed", "M3 verifier failed")
        require(verify_m5(m5_root)["status"] == "passed", "Final M5 verifier failed")
    return {
        "status": "passed",
        "joint_rows": len(saved),
        "m3_rate_seed_pairs": manifest["counts"]["m3_rate_seed_pairs"],
        "m5_ablations": manifest["counts"]["m5_ablations"],
        "m5_permutation_measurements": manifest["counts"]["m5_permutation_measurements"],
        "component_verifiers_checked": check_components,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_m3_m5")
    parser.add_argument("--m3-root", type=Path, default=ROOT / "results/m3")
    parser.add_argument("--m5-root", type=Path, default=ROOT / "results/final_m5")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.output) if args.verify else generate(args.output, args.m3_root, args.m5_root)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
