# 实验系统更新日志

## 2025-12-13

### 优化：实验结果文件管理

**问题**:
- 之前每完成10个测试就保存一次结果，每次保存都生成新的时间戳
- 导致产生多个中间文件（10个测试、20个测试、30个测试...）
- 最终只有36个测试的完整文件有用，其他都是冗余

**解决方案**:
1. 在实验初始化时固定时间戳 (`self.experiment_timestamp`)
2. 所有保存操作都使用这个固定时间戳
3. 每次保存都覆盖同一个文件，而不是创建新文件
4. 报告文件也使用相同的时间戳，方便对应

**效果**:
- 每次实验只生成2个文件：
  - `experiment1_results_TIMESTAMP.json` - 实验结果
  - `experiment1_report_TIMESTAMP.md` - 实验报告
- 中间保存作为增量更新，不产生额外文件
- 如果实验中断，可以看到已完成的部分结果

**修改文件**:
- [experiments/test_01_model_comparison.py](experiments/test_01_model_comparison.py#L100)
  - Line 100: 添加 `self.experiment_timestamp` 固定时间戳
  - Line 589: `save_results()` 使用固定时间戳
  - Line 609: `generate_report()` 使用固定时间戳

---

## LLM评分系统

### 功能：使用LLM作为评委进行质量评分

**背景**:
- 规则评分系统（关键词匹配）存在严重缺陷：
  - 完整性：100%得10分（只要>500字）
  - 精确性：100%得7分（默认"general"类型）
  - 无法检测语义正确性
  - 无法识别幻觉内容

**解决方案**:
- 创建 [llm_judge_rescorer.py](experiments/llm_judge_rescorer.py)
- 使用强模型 (qwen3-235b-a22b-instruct-2507) 作为评委
- 结构化评分标准（Rubric）+ JSON输出
- 4个维度：准确性、完整性、推理质量、专业性
- 并发评分（默认8线程）

**核心发现**:
- LLM评分平均比规则评分高10分 (76.74 vs 66.74)
- **qwen3-8b + RAG 才是最优配置** (84.00分)
- 规则评分严重低估了小模型的能力
- 大模型在无RAG时容易产生幻觉

**使用方法**:
```bash
# 对已有结果重新评分
uv run python experiments/llm_judge_rescorer.py \
  --input outputs/experiment1_results_20251213_200059.json \
  --judge-model qwen3-235b-a22b-instruct-2507 \
  --workers 8
```

**输出文件**:
- `experiment1_results_llm_scored_TIMESTAMP.json` - 重新评分的结果
- `experiment1_llm_scoring_analysis.md` - 深度分析报告

---

## 实验1：模型对比

### 测试配置
- 模型: qwen3-8b, qwen3-14b, qwen3-32b
- 场景: 6个（复杂多需求、技术对比、ROI计算、陷阱问题、近似匹配、近义词区分）
- 模式: 有/无 RAG
- 总测试数: 36 (6×3×2)

### 关键结论

#### 基于LLM评分的最新结论：

1. **最佳配置**: qwen3-8b + RAG (84.00分)
   - 质量最高，成本最低
   - 适用80%场景

2. **高并发场景**: qwen3-14b + RAG (83.25分)
   - 速度最快 (7.41 tokens/s)
   - 延迟最短 (21.64s)

3. **谨慎使用**: qwen3-32b
   - LLM评分反而最低 (82.25分)
   - 延迟长、成本高
   - 无RAG时容易幻觉

#### RAG效果：

| 模型 | 无RAG | 有RAG | 提升 |
|------|-------|-------|------|
| 8b | 68.75 | **84.00** | +15.25 |
| 14b | 71.25 | 83.25 | +12.00 |
| 32b | 70.92 | 82.25 | +11.33 |

**关键洞察**: 小模型从RAG获益最大

---

## 文件结构

```
tts/
├── experiments/
│   ├── test_01_model_comparison.py    # 实验1主程序
│   └── llm_judge_rescorer.py          # LLM评分工具
├── outputs/
│   ├── experiment1_results_TIMESTAMP.json           # 实验结果（规则评分）
│   ├── experiment1_results_llm_scored_TIMESTAMP.json # 实验结果（LLM评分）
│   ├── experiment1_report_TIMESTAMP.md              # 实验报告
│   ├── experiment1_analysis.md                      # 深度分析（规则评分）
│   ├── experiment1_llm_scoring_analysis.md          # 深度分析（LLM评分）
│   └── experiment1_run.log                          # 运行日志
├── run_experiment1.sh                 # 实验1运行脚本
└── CHANGELOG.md                       # 本文件
```

---

## 下一步计划

- [ ] 实现实验2（不同RAG策略对比）
- [ ] 实现实验3（Prompt工程优化）
- [ ] 添加人工评估对比
- [ ] 建立持续评估pipeline
