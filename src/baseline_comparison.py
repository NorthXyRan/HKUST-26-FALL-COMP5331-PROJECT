"""Optional comparison of completed runs against published Table III values.

This command is deliberately separate from the experiment pipeline. It reads
metrics produced by ``src.pipeline evaluate`` and never changes experiment data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PAPER_ROC_AUC = {
    1: {"H1000": 0.9580, "combined_H": 0.9857},
    5: {"H1000": 0.9635, "combined_H": 0.9876},
    10: {"H1000": 0.9646, "combined_H": 0.9872},
}


def compare(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("run_config.json and summary.json are required; complete inject/extract/evaluate first")
    config = json.loads(config_path.read_text())
    summary = json.loads(summary_path.read_text())
    percent = float(config["anomaly_percent_of_original"])
    rounded_percent = int(round(percent))
    if rounded_percent not in PAPER_ROC_AUC or percent != rounded_percent:
        raise ValueError("published Table III references are available at 1%, 5%, and 10% only")
    results = summary["results"]
    comparison = {
        "injection_percent": percent,
        "metric": "ROC-AUC",
        "setting": "random_to_random",
        "reference": "TGF paper Table III",
        "values": {},
        "caveat": "The generated injection realization and local dataset version may differ from the paper.",
    }
    for feature_set in PAPER_ROC_AUC[rounded_percent]:
        matches = [row for row in results
                   if row["feature_set"] == feature_set and row["setting"] == "random_to_random"]
        if matches:
            observed = float(matches[0]["roc_auc"])
            expected = PAPER_ROC_AUC[rounded_percent][feature_set]
            comparison["values"][feature_set] = {
                "paper_roc_auc": expected,
                "run_roc_auc": observed,
                "difference": observed - expected,
            }
    if not comparison["values"]:
        raise ValueError("no H1000 or combined_H random_to_random metrics found in the evaluated run")

    output_json = run_dir / "baseline_comparison.json"
    output_json.write_text(json.dumps(comparison, indent=2) + "\n")
    lines = [
        "# Optional paper baseline comparison", "",
        f"Reference: TGF paper Table III, {rounded_percent}% injection, random→random ROC-AUC.", "",
        "| Feature configuration | Paper ROC-AUC | This run | Difference |",
        "|---|---:|---:|---:|",
    ]
    for name, values in comparison["values"].items():
        lines.append(
            f"| {name} | {values['paper_roc_auc']:.4f} | {values['run_roc_auc']:.6f} | {values['difference']:+.6f} |"
        )
    lines.extend(["", comparison["caveat"], ""])
    (run_dir / "BASELINE_COMPARISON.md").write_text("\n".join(lines))
    print(pd.DataFrame.from_dict(comparison["values"], orient="index").to_string())
    print(f"Saved optional comparison under {run_dir}")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="completed output directory from src.pipeline evaluate")
    args = parser.parse_args(argv)
    compare(args.run_dir)


if __name__ == "__main__":
    main()
