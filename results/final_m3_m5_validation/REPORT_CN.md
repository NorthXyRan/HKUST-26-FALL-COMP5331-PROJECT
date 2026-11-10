# Final M3–M5 项目内环境复验报告

复验日期：2026-11-10（HKT）

## 结论

在仓库根目录内的 `.venv/` 和 `TGF/` 上重新执行后，结果与此前隔离环境一致：

- 完整 `unittest`：45 项通过，0 失败，0 跳过，用时 12.679 秒；
- M3 verifier：通过；
- Final M5 verifier：通过；
- M3–M5 联合 verifier：通过，并递归检查两个组件 verifier；
- 官方 TGF 公式对照：实际执行并通过，没有因 checkout 缺失而跳过。

这次复验没有重新训练模型，也没有覆盖任何冻结实验输入或结果。正式研究结果仍位于：

- `results/m3/`；
- `results/final_m5/`；
- `results/final_m3_m5/`。

本目录只记录“相同结果可以在项目内环境复验”的证据。

## 项目内环境

- Python：3.11.18；
- NumPy：1.26.4；
- pandas：1.5.3；
- scikit-learn：1.6.1；
- SciPy：1.17.1；
- Matplotlib：3.9.4；
- 环境路径：项目根目录 `.venv/`；
- 上游实现路径：项目根目录 `TGF/`；
- TGF commit：`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b`。

`.venv/` 和 `TGF/` 已由 `.gitignore` 排除，不会进入项目提交。它们可以按 README 的命令重建。

## 验证结果

M3：18 条流、57,390 个异常、688,680 行诊断、39 个输入文件和 7 张图全部通过检查。

Final M5：216 条消融、324 条指标漂移检查、54 条预测漂移检查、810 条 permutation、
4,284 条 inventory 和 6,426 条 MDI 全部通过检查。

联合包：18 行 rate/seed/feature-set 证据、9 个 M3 rate/seed 配置、216 条 M5 消融和
810 条 permutation 全部通过检查。

机器可读的环境、哈希、命令与计数见 `verification.json`。
