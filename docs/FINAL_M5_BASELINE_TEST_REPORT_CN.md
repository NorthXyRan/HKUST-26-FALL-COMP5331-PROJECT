# Final M5 阶段 A：基线测试报告

> 本文件保留 2026-11-06 阶段 A 的历史快照。2026-11-10 的当前状态是 45/45 测试通过、
> 无失败或跳过，M3/Final M5/联合 verifier 全部通过。请以
> [`M5_TEST_COMPLETION_CN.md`](M5_TEST_COMPLETION_CN.md) 和
> [`results/final_m3_m5/FINAL_M3_M5_REPORT_CN.md`](../results/final_m3_m5/FINAL_M3_M5_REPORT_CN.md)
> 为当前结论。

日期：2026 年 11 月 6 日  
分支：`final-m5`  
起点：`main@e5c2e73`

## 1. 测试环境

- Python：3.11.18
- NumPy：1.26.4
- pandas：1.5.3
- scikit-learn：1.6.1
- Matplotlib：3.9.4
- SciPy：1.17.1
- 依赖来源：`configs/m5_environment.txt`

项目目录内的 Conda prefix 在创建后发生事务回滚，因此本次测试改用 `/tmp/hkust-final-m5-env`。这不改变代码或实验输入；仓库内 `.venv/` 本来就被 Git 忽略。后续应为组员提供一条稳定的环境创建命令。

## 2. 旧 M5 产物核验

命令：

```bash
python -m src.m5_verify
```

结果：通过。

| 旧 M5 run | 模型变体 | baseline checks | permutation measurements | inventoried columns |
|---|---:|---:|---:|---:|
| 1p_seed_42 | 24 | 24 | 90 | 476 |
| 5p_seed_42 | 24 | 24 | 90 | 476 |
| 10p_seed_42 | 24 | 24 | 90 | 476 |

这证明旧 M5 文件内部一致，但不改变其“旧协议、单 seed”的定位，不能直接作为 Final M5 结果。

## 3. 完整测试结果

首次执行结果：23 通过、1 失败、1 跳过。

失败项是冻结 M2/M4 provenance 测试。原测试比较 manifest 中记录的所有 `src/*.py` 与当前工作树；M5 分支后来修改了 `src/modeling.py`，但最终 M2/M4 runner 从未导入该文件，因此产物本身没有失效。

修复方式：

- 不修改 `results/final_m2_m4/` 或其 manifest；
- 继续核验冻结目录内全部产物哈希；
- 当前源码哈希只检查 `src.final_pipeline` 的真实 import dependency closure；
- manifest 中额外记录的源码哈希仍检查为合法 SHA-256 格式。

修复后重新执行：

```bash
python -m unittest discover -s tests -v
```

最终结果：

- 共运行 25 项测试；
- 24 项通过；
- 0 项失败；
- 1 项跳过；
- 用时约 11.6 秒。

跳过项为 pinned TGF checkout 不存在时的上游公式逐项对照测试。最终 strict-history 特征、冻结产物、所有指标和其余协议测试均已通过，因此该跳过项不阻塞 M5。

## 4. 已确认正常的范围

- 最终 M2/M4 保存文件和实际 artifact 哈希；
- random/matched stream 约束、共享时间戳和 reference alignment；
- strict pre-timestamp 特征与 equal-time 隔离；
- train/validation/test split；
- predictor metadata/label 排除；
- validation-only threshold 和 test-label independence；
- 54 条 M4 predictions/metrics 的重新计算；
- 旧 M5 的分组覆盖、联合置换和内部数值关系。

## 5. 尚未执行的测试与实验

以下项目尚不存在对应 Final M5 实现，因此本阶段不能运行：

- canonical Final M5 输入契约测试；
- 54 条 full-model parity；
- 216 条正式 retrained ablation；
- 810 条 grouped permutation measurements；
- 多 seed paired delta 和 sample-SD 汇总；
- Final M5 manifest/verifier；
- 中文报告和图的 smoke test。

这些是执行计划的阶段 B–D，不应被当前“全套旧测试通过”替代。

## 6. 当时受前置实验限制的部分（现已解除）

阶段 A 当时尚无 M3 正式诊断。现在 M3 已完成并与 strict Final M5 按公共协议哈希、rate/seed
和 feature set 对齐；联合结果支持 endpoint-activity 捷径削弱与 pair-history 残差解释。
common-neighbor/Jaccard/burstiness 仍不是当前分类器输入，因而继续保留非因果解释边界。

## 7. 当前判断

仓库基线健康，冻结 M2/M4 输入可以作为 Final M5 的可信起点。下一步可以安全进入阶段 B：先增加 `tests/test_final_m5.py`，再实现独立的 `src/final_m5.py`，而不修改旧 M5 入口与产物。
