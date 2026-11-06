# Final M5 中文执行计划

## 1. 目标与验收标准

M5 的目标不是重新选择生成器或修改 M2/M4，而是在冻结的最终协议上解释模型为什么在随机异常与活动匹配异常之间出现性能变化。

最终验收需要同时满足：

1. 只读取 `results/final_m2_m4/`，不修改任何冻结输入或旧实验产物。
2. 对 endpoint activity、pair history、global graph 三组特征分别做“删除该组后重新训练”的消融实验。
3. 每个消融模型都使用训练条件对应的 validation 集重新校准 1% FPR 阈值，然后在固定 test 集评估。
4. 覆盖 1%/5%/10% 注入率、seeds 42/43/44、H1000/combined-H，以及 R→R、R→M、M→M 三种设置。
5. 同时提供 grouped permutation importance；MDI 只作为辅助证据，不能单独支持因果结论。
6. 先在每个 seed 内计算配对差值，再跨 seed 汇总均值和样本标准差。
7. 生成可追溯的 CSV、中文报告、图和 manifest；所有结论都能定位到输入、代码版本和随机种子。
8. 无消融模型必须通过最终 M4 的跨平台漂移审计，且完整测试套件通过：test event/时间/标签逐行严格一致；预测分数相关性、平均绝对差、分类一致率和指标漂移都进入 manifest/报告。冻结 M4 只作协议审计，M5 的配对比较必须全部来自同一当前环境。

## 2. 数据与代码边界

以下目录和旧入口保持只读：

- `results/final_m2_m4/`：正式 M2/M4 输入。
- `results/full_pipeline/`：旧协议结果，仅供历史复现。
- `results/m5/`：旧 seed-42 M5 原型，不作为最终结论。
- `src/m5.py`、`src/m5_report.py`、`src/m5_verify.py`：旧 M5 复现入口。
- `configs/m5_groups.json`：旧 M5 manifest 已记录其哈希。

新增实现使用独立命名，避免破坏旧产物：

- `src/final_m5.py`
- `src/final_m5_report.py`
- `src/final_m5_verify.py`
- `configs/final_m5_groups.json`
- `tests/test_final_m5.py`
- `results/final_m5/`

临时输出只能写入已被 Git 忽略的 `work/`，正式 runner 应拒绝覆盖已有最终目录。

## 3. 前置依赖状态

### 已具备

- 最终 M2/M4 的 9 个运行目录：3 个注入率 × 3 个 seed。
- 每个运行的 random/matched feature table、split、predictor JSON、metrics 和 predictions。
- 固定的 Random Forest、训练集平衡和 validation 阈值实现：`src.final_pipeline.fit_score`。
- 三组现有语义映射：endpoint activity、pair history、global graph。

### 执行前需要解决

- 建立项目隔离环境并安装锁定依赖；当前系统默认 Python 不具备 pandas。
- 修复冻结产物测试：manifest 的源码哈希属于历史 base revision，不能与后来合并后的当前文件直接比较。
- 团队确认最终报告使用 `global_graph`，而不是把它误称为 neighborhood。当前模型没有 common-neighbor/Jaccard predictor。

### 尚缺、会限制最终解释的前置实验

- M3 正式诊断产物尚不存在。当前只能完成 M5 的模型侧证据，暂时不能完成以下解释：
  - 正常边、随机异常、匹配异常的完整分布比较；
  - degree/activity 匹配改善及残余差异；
  - neighborhood overlap/Jaccard 差异；
  - prior-pair/repeated-interaction 差异；
  - temporal burstiness 差异；
  - 上述诊断在 rate/seed 之间的稳定性。
- 可选的 pinned TGF checkout 不存在，因此与上游实现逐公式对照的测试会跳过；这不阻塞使用已冻结特征表完成 M5。

在 M3 完成前，报告只能写“观察到某组特征影响发生变化”，不能把变化归因于某个生成机制。

## 4. 正式实验矩阵

### 4.1 Retrained ablation

每个上下文训练完整模型及三个 leave-one-family-out 模型：

| 维度 | 数量 | 取值 |
|---|---:|---|
| 注入率 | 3 | 1%、5%、10% |
| injection/model seed | 3 | 42、43、44 |
| 特征集合 | 2 | H1000、combined-H |
| 训练→测试设置 | 3 | R→R、R→M、M→M |
| 模型变体 | 4 | full、去 endpoint、去 pair、去 global |

预期共 `3 × 3 × 2 × 3 × 4 = 216` 条模型结果，其中 full-model 基线为 54 条。

每个变体记录：

- AP（仓库中命名为 PR-AUC）；
- ROC-AUC；
- recall at validation-calibrated 1% FPR；
- actual validation FPR、actual test FPR 和 threshold；
- train/validation/test 行数、异常数、特征数；
- full minus ablated 的指标变化。

### 4.2 Retraining recovery

对 full 和每个消融变体，在相同 rate/seed/feature set 内计算：

- cross-condition change：`R→M - R→R`；
- matched retraining recovery：`M→M - R→M`；
- feature-family contribution：full 与消融版本的上述配对差值之差。

先保留每个 seed 的 paired delta，再报告三个 seed 的 mean、sample SD、min 和 max。

### 4.3 Grouped permutation importance

对 54 个 full model，在固定 test rows 上联合打乱一个特征组的全部列；combined-H 中同一组跨所有 horizon 使用同一个行排列。

若每组重复 5 次，预期测量数为 `54 × 3 × 5 = 810`。shuffle SD 只表示排列波动，不能代替跨 injection seed 的不确定性。

### 4.4 MDI

保存 full model 的逐列 impurity importance，用于定位具体列，但主结论以 retrained ablation 和 grouped permutation 为准。MDI 可能偏向高基数特征，并受相关特征分摊影响。

## 5. 需要补充的测试

### P0：合并前必须通过

1. **输入契约**
   - 自动发现且仅发现 9 个最终 run。
   - 每个 run 的 rate、seed、文件集合和 predictor JSON 完整。
   - random 与 matched 表 schema 一致。
   - 只使用保存的 `split`，不重新按行数切分。

2. **防数据泄漏**
   - predictor 中不得出现 `t`、`u`、`v`、`event_id`、`split`、label。
   - train、validation、test 的 event_id/行集合在所有消融变体中保持不变。

3. **分组契约**
   - 每个 predictor 恰好属于一个组，不能遗漏或重叠。
   - H1000 应有 34 个 predictor，combined-H 应有 204 个。
   - 删除一组后特征数准确；combined-H 必须同时删除该组的全部 horizon 列。
   - 未知列、空组或删除后无特征时 fail fast。

4. **训练与阈值协议**
   - 每个消融模型重新训练并使用自己的 validation scores 校准阈值。
   - actual validation FPR 不超过 1%。
   - R→R 与 R→M 共享相同 random-trained 模型/阈值，但使用各自 test rows。
   - M→M 使用 matched train/validation。

5. **M4 协议一致性与跨平台漂移审计**
   - 54 条 full-model 结果与冻结 predictions/metrics 对齐；event_id、时间和 label 必须逐行严格一致。
   - 记录分数相关性、平均/最大绝对差、分类一致率，以及 AP、ROC-AUC、recall、actual validation/test FPR 和 threshold 的实际差值。
   - 冻结 M4 来自 macOS ARM/Python 3.9，Linux x86/Python 3.11 重训不要求 bitwise 一致；超出事先声明的预测或指标灾难性漂移护栏则失败。

6. **结果完整性与聚合**
   - ablation 恰有 216 条；permutation 在 5 repeats 时恰有 810 条。
   - metric drop、cross-condition change 和 retraining recovery 公式正确。
   - aggregation 使用 seed 间 sample SD，不混入 permutation repeats。

7. **可复现性与 provenance**
   - 相同输入/种子重复运行结果一致。
   - manifest 覆盖输入哈希、源码哈希、分组、环境、seed 和实验参数。
   - verifier 能发现缺文件、哈希变化、重复/缺失上下文和数值关系错误。

### P1：建议完成

- grouped permutation 的组内联合排列、确定性和输入不可变测试。
- 缺少类别、validation 无负例、schema 不一致等失败路径。
- 中文报告和图片生成 smoke test。
- 输出目录已存在时拒绝覆盖测试。
- 用小型 synthetic fixture 完成端到端测试，避免单元测试依赖完整实验时长。

## 6. 执行阶段与命令

### 阶段 A：仓库基线

```bash
python -m unittest discover -s tests -v
python -m src.m5_verify
```

记录通过、失败和跳过项；环境错误与代码错误分开报告。

### 阶段 B：实现前协议测试

先增加 `tests/test_final_m5.py`，以小型 fixture 固定输入选择、分组、split、threshold 和聚合契约，再实现 runner。

### 阶段 C：单 run smoke test

```bash
python -m src.final_m5 \
  --input results/final_m2_m4 \
  --output work/final_m5_smoke \
  --rates 1 --seeds 42
```

首先确认 24 条 ablation 结果及 6 条 full baseline parity，再扩大实验。

### 阶段 D：完整实验

```bash
python -m src.final_m5 \
  --input results/final_m2_m4 \
  --output results/final_m5

python -m src.final_m5_verify results/final_m5
python -m unittest discover -s tests -v
```

### 阶段 E：M3 联合解释

在 M3 产物冻结并核验后，按 rate/seed 对齐诊断和 M5 effect，补充机制解释。若 M3 尚未完成，报告必须保留明确的 pending 标记。

## 7. 中文报告结构

最终报告建议包含：

1. 一页结论摘要；
2. 数据、协议和三类特征的通俗定义；
3. full-model 的跨条件性能变化与 retraining recovery；
4. retrained ablation、grouped permutation 和 MDI 的一致与不一致之处；
5. 跨 seed 稳定性；
6. 与 M3 diagnostics 的联合解释；
7. 限制：单数据集、单分类器、三个 seed、相关特征、后验解释；
8. 复现命令、输入/代码哈希和产物索引。

任何结论都应限制在当前 UCI Messages、固定 Random Forest 和已测试特征内，不使用“证明了因果关系”等表述。

## 8. 完成定义

只有在以下条件同时成立时，Final M5 才标记完成：

- P0 测试全部通过；
- 54 条 full baseline 与最终 M4 的输入行严格对齐，并通过声明的预测/指标跨平台漂移审计；
- 216 条 ablation 与 810 条 permutation 结果完整；
- 多 seed 配对汇总、中文报告、图和 manifest 已生成；
- M3 证据已接入，或所有依赖 M3 的结论明确标记为未完成；
- `git status` 只包含经过审查、计划提交的代码、测试、文档和正式产物。
