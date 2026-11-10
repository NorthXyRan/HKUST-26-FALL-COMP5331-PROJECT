# M3 更新后的 Final M5 测试完成报告

日期：2026-11-10（HKT）

工作分支：`final-m5`

M3 来源：`origin/codex/m3-diagnostics`，commit `39c894c94f25853df08e3c2c477f62b083ac848a`

## 1. 结论先行

当前仓库中 **M1–M5 已声明的核心自动化测试项目均已完成**：完整测试套件为
`45 passed / 0 failed / 0 skipped`，M3、Final M5 和 M3–M5 联合包的独立验证器全部通过。

这不等于整个课程项目的所有交付已经结束。M6 最终报告整合、M7 幻灯片/展示分工仍是后续工作；
更多数据集、更多分类器和独立 classifier-seed 交叉实验属于研究增强项，也尚未执行。

## 2. 本次接入了什么

M3 更新来自上面的固定 commit，新增内容包括：

- 18 条流：3 个注入率 × 3 个 injection seed × 2 个生成器；
- 57,390 个异常和 688,680 行展开诊断；
- test/train/validation、100/1,000/10,000 窗口、observed/benign-only history 对照；
- activity/degree、common neighbors、Jaccard、pair history、recency、concentration 和 burstiness；
- 输入、源码、依赖、输出哈希与独立 verifier。

M3 与 Final M5 的 `results/final_m2_m4/protocol.json` 和 `manifest.json` 哈希完全一致，
因此可以在同一受控实验中按 rate/seed 对齐。新增联合包位于
[`results/final_m3_m5/`](../results/final_m3_m5/)，没有重写任何冻结输入或已有 M3/M5 结果。

## 3. 完整测试结果

执行命令：

```bash
python -m unittest discover -s tests -v
```

结果：

| 测试组 | 数量 | 结果 | 主要覆盖 |
| --- | ---: | --- | --- |
| 冻结 M2/M4 产物 | 4 | 全部通过 | 文件/历史源码哈希、流约束、指标重算、特征与切分 |
| Final M5 合同 | 6 | 全部通过 | run matrix、特征分组、消融配对、置换、预测漂移 |
| 严格 pipeline 与上游公式 | 10 | 全部通过 | past-only、同时间不可见、阈值、切分、TGF 公式一致性 |
| 生成器行为 | 7 | 全部通过 | 可复现、新边/非自环、窗口、覆盖失败、无随机降级 |
| M3 行为 | 9 | 全部通过 | 滑动窗口、时间边界、burstiness、重放、配对汇总、篡改拒绝 |
| M3–M5 联合合同 | 5 | 全部通过 | 公共协议、完整 join、逐项原值、语义边界、产物哈希 |
| 旧 M5 回归 | 4 | 全部通过 | 分组、置换、多窗口、共享 modeling 兼容性 |
| **总计** | **45** | **45 通过，0 失败，0 跳过** | |

原先唯一会跳过的上游公式对照，本次使用官方 TGF commit
`af9d575e07c7bc3fd0b4a0b8d0249e7cdd2eb88b` 实际执行并通过。checkout 只用于本地测试，
不属于本仓库提交内容。

测试输出出现两类非失败 warning：旧版 pandas 对 NumPy API 的 deprecation warning，以及 M3
极小合成样本中用于验证“未定义 burstiness 保持缺失”的除零 warning。两者都没有改变测试结果。

## 4. 三个独立验证器

### M3

```text
status=passed, streams=18, anomalies=57390,
diagnostic_rows=688680, input_files=39, figures=7
```

### Final M5

```text
status=passed, ablations=216, baseline_checks=324,
baseline_prediction_checks=54, permutation_measurements=810,
inventory_rows=4284, mdi_rows=6426
```

### M3–M5 联合包

```text
status=passed, joint_rows=18, m3_rate_seed_pairs=9,
m5_ablations=216, m5_permutation_measurements=810,
component_verifiers_checked=true
```

联合 verifier 会重新检查公共协议哈希、两个结果包输入哈希、联合矩阵、摘要公式、源码哈希和输出哈希。

## 5. 新增联合测试具体回答什么

1. **能不能联合？** 能。M3 和 Final M5 使用相同冻结输入、协议、rates 与 seeds。
2. **join 是否完整？** 完整。9 个 rate/injection-seed 配置与 2 个 feature set 对齐为 18 行，无缺失或重复。
3. **活动匹配是否有效？** 有效。九个配置的 activity error 都下降；degree、common-neighbor、
   Jaccard KS 也在每个 rate/seed 中下降。
4. **M5 数值有没有在 join 时被改变？** 没有。联合表的每个 AP drop 都逐项等于原始
   `ablations.csv`。
5. **有没有过度解释特征？** 测试强制保留 `global_graph` 的真实语义，不允许把它说成
   neighborhood 组；也明确记录 common-neighbor/Jaccard/burstiness 并非当前模型预测列。

## 6. 随机性口径

- **Injection-seed variation**：M3 与冻结输入使用 42、43、44。
- **Classifier-seed variation**：Final M5 的 Random Forest `random_state` 与对应 injection seed
  使用相同数值，没有独立交叉；所以当前跨-seed 波动不能拆分成纯注入波动与纯模型波动。
- **Permutation repeats**：每个上下文 5 次 shuffle，固定 permutation seed 31415；这是置换波动，
  不是 5 个新数据集或 5 个新 injection seeds。

若最终报告需要分别估计 injection 与 classifier 随机性，应新增
`injection_seed × classifier_seed` 交叉实验。这是统计增强项，不是当前 M5 完成标准中的缺失测试。

## 7. 完成与未完成边界

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| M3 冻结诊断 | 完成 | 哈希、18 流、7 图、独立 verifier 均通过 |
| Final M5 核心实验 | 完成 | 216 消融、810 置换、6,426 MDI、漂移审计 |
| M3–M5 联合测试 | 完成 | 5 项联合测试与联合 verifier 通过 |
| 完整仓库自动化测试 | 完成 | 45/45，通过且无跳过 |
| M6 最终报告 | 未完成 | 需把 RQ、主图表、限制和贡献整合为提交稿 |
| M7 展示材料 | 未完成 | 需制作幻灯片、安排六人讲解并排练 |
| 更多数据集/真实异常 | 未完成（可选增强） | 当前结论仅限 UCI Messages 与人工异常 |
| 独立 classifier-seed 网格 | 未完成（可选增强） | 当前模型 seed 与 injection seed 绑定 |

因此，准确说法是：**M5 及其依赖的 M3 核心测试已经完成；M1–M5 技术实验包可进入 M6，
但整个课程项目尚未完成。**

## 8. 复现前置依赖

1. 从仓库根目录运行，使用 `requirements.txt` 固定版本；M3 的补充数值依赖记录在
   `configs/m3_environment.txt`。
2. 保留 `results/final_m2_m4/`、`results/m3/` 和 `results/final_m5/` 不变。
3. 如需执行上游公式对照，按 README 稀疏 checkout TGF，并固定到上述 commit。
4. 新分析必须写入新的 `results/<milestone>/` 目录；生成器 seed、分类器 seed 与 permutation
   repeats 必须分别报告。

首次复验时主目录空间不足，因此锁定环境和 TGF checkout 临时放在 `/tmp`。空间释放后，已经
按 README 在项目根目录建立 `.venv/` 和 `TGF/`，再次得到 `45/45` 通过、0 跳过，并通过三个
verifier。项目内复验的机器可读记录见
[`results/final_m3_m5_validation/`](../results/final_m3_m5_validation/)。`.venv/` 和 `TGF/` 均被
`.gitignore` 排除，不会进入提交，也不会改变冻结结果。
