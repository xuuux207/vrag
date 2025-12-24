# 实验3：v3 → v4 升级日志

## 更新时间
2025-12-23

## 核心改进

### 新增方法4：渐进式总结 + 增量RAG（v3版本）

针对方法3（v2渐进式总结）发现的问题，开发了全新的v3版本实现。

## 问题分析

### 方法3（v2）的问题
1. **信息丢失严重**：只保留了最后一段的内容，前面所有信息全部丢失
2. **评分很低**：58.6/100，远低于其他方法（82-83分）
3. **RAG时间异常**：2.91秒，明显高于正常的0.3-0.5秒

### 根本原因
```python
# v2的prompt太弱
SIMPLE_SUMMARY_PROMPT = """
【之前的总结】：{previous_summary}
【新输入】：{new_segment}

请更新总结（过滤口语词、寒暄，保留关键信息）。
"""
```

虽然要求"保留关键信息"，但：
- **输入只有摘要 + 摘要**，没有完整上下文
- LLM容易"遗忘"之前的细节
- 每次总结都是基于简化的文本

## 解决方案：v3版本

### 核心思想
1. **总结输入 = 之前总结 + 当前段全文**（保证信息不丢失）
2. **每次输入都进行RAG检索**（实时发现相关文档）
3. **过滤低相关度文档**（只保留有效结果）
4. **文档去重**（避免重复检索）

### 技术实现

#### 新文件：`incremental_summarizer_v3.py`
```python
class IncrementalRAGSummarizer:
    """增量RAG总结器 - 边总结边检索"""

    def add_segment(self, segment_text: str, simulate_delay: bool = True):
        # 1. 总结：输入包含总结+全文
        prompt = INCREMENTAL_SUMMARY_PROMPT.format(
            previous_summary=self.current_summary,
            current_segment=segment_text  # ← 完整段落文本！
        )

        # 2. RAG检索（每个segment都执行）
        new_docs = self._rag_search(
            query_text=self.current_summary,  # 用总结检索
            segment_text=segment_text,        # 用全文计算相关度
            top_k=5
        )

        # 3. 累积相关文档（去重+过滤）
        self.relevant_docs.extend(new_docs)
```

#### 关键方法
1. **`_calculate_relevance()`**：使用余弦相似度计算相关度
2. **`_rag_search()`**：增量RAG，每次输入都检索
   - 使用总结作为query（简洁）
   - 使用全文计算相关度（准确）
   - 过滤低于阈值的文档（默认0.6）
   - 去重已检索的文档ID

## 实验更新

### 文件变更

#### 1. `test_03_v3.py`
```python
# 新增导入
from experiments.incremental_summarizer_v3 import IncrementalRAGSummarizer

# 新增方法4函数
def method4_incremental_rag(...):
    """方法4：边输入边总结+边检索，过滤低相关度文档（v3版本）"""
    summarizer = IncrementalRAGSummarizer(
        llm_client,
        embedding_service,
        vector_index,
        model_name=SUMMARY_MODEL,
        relevance_threshold=0.6  # 相关度阈值
    )
    # ... 处理逻辑

# 更新并行执行：从3个方法增加到4个
with ThreadPoolExecutor(max_workers=4) as executor:
    future_m1 = executor.submit(method1_baseline, ...)
    future_m2 = executor.submit(method2_batch_summary, ...)
    future_m3 = executor.submit(method3_incremental_summary, ...)
    future_m4 = executor.submit(method4_incremental_rag, ...)  # 新增
```

#### 2. `analyze_exp3_v3_results.py`
- 新增 `m4_data` 数据收集
- 所有表格增加 Method 4 列
- 更新关键发现部分，分析4个方法的对比
- 新增 Method 4 特有指标：检索文档数、增量RAG时间

### 输出示例
```
🚀 并行运行四个方法...
  ✓ 方法1完成（3.24秒）
  ✓ 方法2完成（4.15秒）
  ✓ 方法3完成（输入后: 2.87秒）
  ✓ 方法4完成（输入后: 2.31秒, 检索12个文档）
```

## 预期效果

### 方法4的优势
1. **信息保留率高**：每次输入包含完整段落，不会丢失信息
2. **RAG质量好**：增量检索，文档更相关
3. **延迟最低**：总结和RAG都在用户说话时完成
4. **去重过滤**：只保留高相关度文档，减少噪音

### 对比 v2 (方法3)
| 维度 | v2 (方法3) | v3 (方法4) |
|------|-----------|-----------|
| 总结输入 | 总结 + 总结 | **总结 + 全文** |
| RAG时机 | 最后一次 | **每次输入** |
| 文档过滤 | 无 | **相关度阈值** |
| 文档去重 | 无 | **ID去重** |
| 信息保留 | ❌ 丢失 | ✅ 完整 |

## 如何运行

### 1. 运行实验（并行测试4个方法）
```bash
uv run python experiments/test_03_v3.py
```

### 2. 分析结果
```bash
uv run python experiments/analyze_exp3_v3_results.py
```

### 3. 查看对比
分析脚本会显示4个方法的详细对比：
- LLM综合评分（0-100分）
- 延迟分析（RAG时间、TTFT、生成时间、总延迟）
- 输出质量（Query长度、压缩比、输出速度）
- 关键发现（综合评分、用户体验、特点分析）

## 文件清单

### 新增文件
- `experiments/incremental_summarizer_v3.py` - v3总结器实现

### 修改文件
- `experiments/test_03_v3.py` - 添加方法4，并行运行4个方法
- `experiments/analyze_exp3_v3_results.py` - 支持4个方法的对比分析

### 保留文件
- `experiments/incremental_summarizer_v2.py` - v2总结器（保留用于对比）
- `experiments/long_audio_test_cases_v2.json` - 测试用例
- `experiments/README_v3.md` - 实验说明

## 下一步

1. **运行完整实验**：测试所有5个测试用例
2. **分析结果**：查看方法4是否解决了信息丢失问题
3. **对比评分**：验证方法4的综合得分是否最高
4. **优化参数**：根据结果调整相关度阈值（0.6）

## 注意事项

1. **兼容性**：保留了方法3（v2），可以对比两个版本
2. **性能开销**：方法4每次都做RAG，会增加处理时间（但在用户说话时完成）
3. **阈值调整**：`relevance_threshold=0.6` 可以根据实际效果调整
4. **并行执行**：4个方法同时运行，总时间约等于最慢方法的时间

## 期待结果

如果v3实现正确，应该看到：
- ✅ 方法4的信息保留率显著提升
- ✅ 方法4的综合评分接近或超过方法1/2
- ✅ 方法4的用户感知延迟最低（总结+RAG都在输入时完成）
- ✅ 方法4的RAG相关性更高（增量检索+过滤）
