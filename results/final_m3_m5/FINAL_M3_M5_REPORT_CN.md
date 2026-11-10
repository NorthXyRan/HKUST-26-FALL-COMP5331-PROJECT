# Final M3–M5 联合测试与研究结论

## 1. 联合测试结论

M3 与 Final M5 使用同一个 `strict_m2_m4_v1` 冻结输入包，协议和基础 manifest 的 SHA-256 完全一致；注入率、注入 seed、历史定义和数据切分均可对齐。因此，当前 M3 诊断可以与 Final M5 消融作联合解释。旧版 `results/full_pipeline` 和 `results/m5` 仍只作为历史产物。

本联合包对齐了 9 个 rate/seed 诊断配置与两个特征集合，共 18 行证据；同时复核 216 条消融、810 条置换测量和 6426 条 MDI 记录。

## 2. M3 告诉我们：匹配是否真的有效

在 test、observed history、1,000-event window 的预先声明主分析中，activity matching 在九个 rate/seed 配置中都降低了端点活动误差，平均降低 56.3700%（范围 53.7159%–59.8681%）。

平均 KS 距离也显示匹配后的结构更接近对应正常边：

| 诊断 | Random | Activity-matched |
| --- | ---: | ---: |
| degree mean | 0.8545 | 0.4170 |
| common neighbors | 0.1255 | 0.0713 |
| Jaccard | 0.1255 | 0.0771 |
| recency | 0.8440 | 0.5308 |
| burstiness（仅定义样本） | 0.1926 | 0.0732 |
| pair count | 0.8332 | 0.8350 |

`pair_count` 几乎没有改善：两种生成器都强制异常边是从未出现过的新边，因此异常的历史 pair count 全为 0，而对应正常 reference edge 通常有重复历史。这是活动匹配没有消除的明显残余差异。

## 3. M5 告诉我们：模型实际用了什么

下表给出完整模型 AP 减去删除特征组后 AP 的均值；正数越大，说明重新训练后缺少该组造成的损失越大。均值覆盖 3 个注入率、3 个 seed 和 2 个特征集合。

| 设置 | Endpoint activity | Pair history | Global graph |
| --- | ---: | ---: | ---: |
| R→R | 0.3379 | -0.0059 | 0.0090 |
| R→M | 0.0285 | 0.0928 | 0.0039 |
| M→M | 0.1873 | 0.2204 | -0.0062 |

联合起来可以得到较稳妥的解释：随机异常留下很强的端点活动捷径，所以 R→R 高度依赖 endpoint activity；活动匹配削弱了该捷径，R→M 中 endpoint 组的平均作用明显变小。由于所有异常仍是历史新边，pair-history 残差没有被匹配掉，R→M 和 M→M 对 pair-history 的依赖更强。

这不是完整的因果证明。消融组 `global_graph` 是窗口级全图统计，不能改名为 neighborhood 组；M3 的 common-neighbor、Jaccard 和 burstiness 也不是当前分类器的预测列。因此只能说这些残差存在，不能声称模型直接使用了它们。

## 4. 随机性与统计范围

- M3 的 seeds 42、43、44 只代表注入 seed 变化，不训练分类器。
- M5 在每个冻结注入 run 上令 Random Forest `random_state` 使用同一个数值 seed；没有把注入 seed 与独立分类器 seed 做完整交叉实验。
- grouped permutation 每个上下文做 5 次 shuffle，固定 permutation seed 31415；这些 shuffle 只刻画置换波动，不是新的数据集或独立注入重复。

所以当前的跨-seed 波动混合了注入样本变化和与之绑定的模型随机性。若最终报告需要把两者分别估计，仍需额外运行 injection-seed × classifier-seed 的交叉实验；这属于统计增强，不影响当前预先声明的 M5 完成标准。

## 5. 完成范围

M3–M5 的核心实验、联合协议检查、结果哈希、完整矩阵和机制边界均已覆盖。项目整体仍需 M6 最终报告整合与 M7 展示材料；单数据集、单分类器、无真实异常外部验证仍是研究限制，而不是已经通过的测试。

机器可读的逐 rate/seed/feature-set 证据见 `joint_evidence.csv`，汇总与随机性说明见 `joint_summary.json`，输入、源码、依赖和输出哈希见 `manifest.json`。
