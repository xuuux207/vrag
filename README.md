# 虚拟销售系统：实时性与专业性的工程解决方案
## Workshop 设计文档

---

## 一、Workshop 定位

### 核心目标
解决虚拟销售/客服系统中的**三个独立工程问题**，帮助参会者从 Qwen 基础模型出发，通过系统工程优化实现：
- ≤1500ms 端到端延迟
- 30轮内无显著幻觉的专业对话
- 40-60秒长语音的准确理解

### 为什么不讲 MoE？

#### 用户问题中的 MoE 误区
原始需求提到："LLM 专业度提升方案：**MoE 路由架构**（按行业或产品类型划分）"

这里隐含的假设是：**需要用 MoE 去自动路由到不同的专家模型**

但问题是：
- ❌ 假设 1：这不需要你自己训练 MoE 模型——Qwen 等大厂的模型本身已经内置了多专家机制
- ❌ 假设 2：简单的 MoE 路由**不能根本解决幻觉问题**——幻觉源于知识不足，而非模型选择
- ✅ 正确理解：如果基础模型 + 高质量知识库 + 工程优化都做好了，MoE 只是**锦上添花**，不是**雪中送炭**

#### 本 Workshop 的视角
我们关注的是**工程设计而非模型黑盒**：
- **模型选择是手段**：选合适规模的 Qwen，确保延迟可控
- **知识库是关键**：用 RAG 从异构数据精准检索，防止幻觉
- **处理流程是保障**：长输入分段、完整理解，确保回答准确

MoE 在这里的真实位置：如果上述三个环节都优化后，系统仍然存在专业度不足的问题，**那时**才考虑 MoE 作为后续优化。但绝大多数场景下，这三个环节就已经足够了。

---

## 二、Workshop 结构：总分总

### 🎯 **第一部分：导入（总）— 问题场景与技术困境**

#### 内容
1. **场景复盘**：虚拟销售流程中的 ASR → LLM → TTS 链条

2. **困境揭示**：
   - 困难 1：延迟压力（1.2～2秒需要完整回答）
   - 困难 2：幻觉风险（30轮内不能出现显著错误）
   - 困难 3：长输入理解（客户可能讲 40-60秒的需求）

3. **澄清：为什么不是 MoE？**
   
   原始需求中提到用 MoE 路由架构来提升专业性。但这里有个关键误区需要澄清：
   
   - **误区**：认为需要自己训练或精细设计 MoE 专家路由才能解决幻觉问题
   - **事实**：幻觉的根源是**知识不足**，而非模型架构。Qwen 等基础模型已内置 MoE 机制，我们需要解决的是：
     1. 选对参数规模（确保延迟可控）
     2. 建立高质量知识库（通过 RAG 精准检索）
     3. 妥善处理长输入（完整理解用户需求）
   
   **本 Workshop 的视角**：如果上述三个工程环节都优化好了，MoE 可能只是锦上添花。绝大多数场景下，这三个方案就已经足够了。

4. **方案预告**：三个独立的技术解决方向

#### 代码示例
```python
# 系统延迟分解
timeline = {
    "ASR": 0.3,           # 语音识别
    "LLM_input": 0.1,     # 输入处理
    "LLM_inference": 0.5, # 推理（核心）
    "LLM_output": 0.1,    # 输出处理
    "TTS": 0.4,           # 语音合成
}
total = sum(timeline.values())
print(f"理论最小延迟: {total}s | 目标: 1.5s")
print("LLM 推理是关键瓶颈 → 参数规模成为首要决策")
```

---

### 🔧 **第二部分：分块内容（分）**

#### **Block 1：模型选型与延迟可行性分析**

**因果链**：500ms 推理约束 → 模型规模 → 硬件需求

##### 问题
"应该选多大的 Qwen 模型才能满足 500ms 推理时间？"

##### 解决思路
1. **Qwen 系列对标**
   - 3B：低延迟，专业性可能不足
   - 7B：性能/延迟平衡点（**推荐起点**）
   - 14B：专业性更强，但延迟可能超标
   - 72B：专业性最优，需要高端硬件

2. **推理速度计算**
   - 单 token 生成时间 = 模型大小 / GPU 显存带宽
   - 平均回答长度 × 单 token 时间 = 推理延迟

3. **实验验证方向**
   - 在目标硬件上跑 benchmark
   - 测试真实业务场景的往返延迟

##### 可运行代码
```python
# Qwen 模型规模与延迟估算
import numpy as np

models = {
    "Qwen3-3B": {"params": 3e9, "tokens_per_sec": 100},
    "Qwen3-7B": {"params": 7e9, "tokens_per_sec": 50},
    "Qwen3-14B": {"params": 14e9, "tokens_per_sec": 25},
    "Qwen3-72B": {"params": 72e9, "tokens_per_sec": 8},
}

avg_response_tokens = 50  # 平均回答长度

print("推理延迟估算 (假设平均回答50个token):")
print("-" * 50)
for name, spec in models.items():
    inference_time = avg_response_tokens / spec["tokens_per_sec"]
    status = "✓ 可行" if inference_time <= 0.5 else "✗ 超限"
    print(f"{name}: {inference_time:.3f}s {status}")

print("\n结论: Qwen3-7B 是最有效的平衡点")
```

##### 输出期望
- 参会者了解参数规模选择的**量化依据**
- 理解硬件成本与延迟的权衡关系

---

#### **Block 2：RAG 架构 — 异构数据的有效融合**

**因果链**：异构数据（结构化+非结构化）→ 分别处理 → 统一检索 → 降低幻觉

##### 问题
"结构化企业图谱数据和非结构化业务文档怎么整合，才能有效降低幻觉？"

##### 解决思路
1. **数据源特征分析**
   - 左手（结构化）：企业信息、关系图谱、属性标准化
   - 右手（非结构化）：Word、PPT、PDF 的文本与图片内容

2. **分开存储的优势**
   - 结构化数据 → 图数据库/关系数据库（精确检索）
   - 非结构化数据 → 向量数据库（语义检索）
   - 检索时分别查询，再由 LLM 做融合决策

3. **融合检索策略**
   - 用户问题 → 同时查询两个数据源
   - 结构化结果（高精度）+ 向量结果（高召回）
   - 根据匹配分数重排，提供给 LLM

4. **幻觉抑制机制**
   - 检索结果的来源标记（便于追溯）
   - 知识库覆盖度评估（未覆盖问题提前告知）

##### 可运行代码
```python
# RAG 异构数据融合演示

# 1. 结构化数据示例（企业图谱）
structured_data = {
    "company": "西门子",
    "industry": "工业自动化",
    "solutions": ["PLC控制", "工业互联网", "数字化转型"],
    "clients": ["宝马", "大众", "西门子中国"],
}

# 2. 非结构化数据示例（业务文档）
unstructured_docs = [
    "西门子 Siemens S7-1200 PLC 是面向中小型应用的高性能控制器...",
    "工业4.0 解决方案可以帮助传统制造业实现数字化转型...",
    "云平台支持实时监控和远程诊断功能...",
]

# 3. 用户问题
user_query = "西门子的 PLC 产品有什么特点？"

# 4. 分别检索
print("=== 检索流程演示 ===\n")

# 结构化检索（精确）
print("【结构化检索】")
if "PLC" in user_query:
    for solution in structured_data["solutions"]:
        if "PLC" in solution:
            print(f"✓ 匹配: {solution}")

# 向量检索（模拟）
print("\n【向量检索】")
query_keywords = ["PLC", "特点", "功能"]
for i, doc in enumerate(unstructured_docs):
    match_score = sum(1 for kw in query_keywords if kw in doc) / len(query_keywords)
    if match_score > 0:
        print(f"文档 {i+1}: {doc[:50]}... (相似度: {match_score:.2f})")

# 5. 融合结果
print("\n【融合结果】")
rag_context = f"""
结构化信息：西门子的核心解决方案包括 {', '.join(structured_data['solutions'])}
文档补充：{unstructured_docs[0][:50]}...
"""
print(rag_context)
print("\n→ LLM 基于上述背景知识回答，避免纯粹幻觉")
```

##### 输出期望
- 参会者理解**异构数据不需要混淆**，反而要分开处理
- 掌握融合检索的实现思路

---

#### **Block 3：长输入处理 — 分段与多轮理解**

**因果链**：40-60秒长语音 → 语义分段 → 独立理解 + 上下文保留 → 完整回答

##### 问题
"客户讲了一大堆需求（40-60秒），怎么处理才能既不丢失信息，又避免模型混乱和幻觉？"

##### 解决思路
1. **长输入的两大挑战**
   - Token 长度可能超过模型上下文窗口
   - 多个独立需求混合，容易导致答非所问或模型出现幻觉

2. **分段策略**
   - 不是简单的固定长度截断
   - 而是**语义边界检测**（句子结束、逻辑停顿点）
   - 每段独立处理，保留段间连接信息

3. **处理方式**
   - 方式 A：流式处理（段出现即处理，降低延迟）
   - 方式 B：聚合理解（所有段都收集后，统一回答）
   - 混合：先流式收集段结构，再聚合处理

4. **质量保障**
   - 每段提取关键点
   - 段间逻辑关系识别
   - 最后做完整性检查

##### 可运行代码
```python
# 长输入分段处理演示

# 模拟客户 60 秒的语音转录
long_input = """
我们公司是一家制造企业，主要生产汽车零部件。
目前面临的问题是生产效率低下，产品不良率在 15% 左右。
我们听说西门子的工业控制系统能帮助优化生产流程。
另外，我们的库存管理也很混乱，经常出现过库或缺库的情况。
想问一下，西门子是否有完整的 ERP 加自动化的整体解决方案？
"""

print("=== 长输入分段处理 ===\n")

# 1. 语义分段
import re

# 简单的句号/问号作为分段点
sentences = re.split(r'[。？！]', long_input.strip())
sentences = [s.strip() for s in sentences if s.strip()]

print(f"原始输入长度: {len(long_input)} 字符")
print(f"分段数量: {len(sentences)} 个语义单元\n")

# 2. 逐段处理
segments_with_topics = []
for i, sentence in enumerate(sentences):
    # 提取关键信息
    if "效率" in sentence or "不良率" in sentence:
        topic = "生产优化"
    elif "库存" in sentence:
        topic = "库存管理"
    elif "解决方案" in sentence:
        topic = "产品咨询"
    else:
        topic = "背景信息"
    
    segments_with_topics.append({
        "id": i+1,
        "text": sentence,
        "topic": topic
    })

print("【分段结果】")
for seg in segments_with_topics:
    print(f"段 {seg['id']} [{seg['topic']}]: {seg['text']}")

# 3. 关键点提取
print("\n【关键点提取】")
key_points = {
    "企业类型": "汽车零部件制造",
    "主要问题": ["生产效率低", "产品不良率 15%", "库存管理混乱"],
    "咨询方向": "工业控制系统 + ERP 整体方案",
}
print(key_points)

# 4. 完整性检查
print("\n【完整性检查】")
all_topics = set(seg["topic"] for seg in segments_with_topics)
print(f"涵盖的话题: {all_topics}")
print("✓ 背景信息完整")
print("✓ 问题点清晰")
print("✓ 咨询需求明确")

print("\n→ 现在 LLM 可以基于这个完整的背景，生成一致的专业回答")
```

##### 输出期望
- 参会者掌握**分段不是简单截断**的理念
- 了解如何从长输入中提取完整信息

---

### 🎬 **第三部分：总结与实施方案（总）**

#### 内容

1. **三个解决方案的组合效应**
   ```
   参数选型 (Block 1)
        ↓
   能否在 500ms 内推理？
        ↓
   + RAG 异构融合 (Block 2)
        ↓
   是否能获取准确知识？
        ↓
   + 长输入分段 (Block 3)
        ↓
   能否理解完整需求？
        ↓
   → 实现"实时+专业"的系统
   ```

2. **实施优先级**
   - 优先级 1：选定模型规模，完成硬件评估
   - 优先级 2：搭建 RAG 基础设施
   - 优先级 3：集成长输入处理逻辑
   - 可选：根据实际效果，评估是否需要 MoE 等高级优化

3. **风险检查清单**
   - [ ] 推理延迟是否稳定 < 500ms？
   - [ ] RAG 检索的精度与召回率是否可接受？
   - [ ] 长输入分段是否保留了完整信息？
   - [ ] 幻觉频率是否在可控范围内？

#### 代码示例：完整 Pipeline
```python
# 完整系统流程模拟
import time

def virtual_sales_pipeline(customer_input):
    """模拟虚拟销售系统的完整流程"""
    
    print("=" * 60)
    print("🎤 客户输入（通过 ASR 获得）:")
    print(customer_input)
    print("=" * 60)
    
    # Step 1: 长输入处理
    print("\n[Step 1] 长输入分段处理...")
    segments = segment_input(customer_input)  # Block 3 逻辑
    print(f"✓ 分成 {len(segments)} 个语义单元")
    
    # Step 2: RAG 检索
    print("\n[Step 2] RAG 异构数据检索...")
    rag_context = retrieve_from_rag(segments)  # Block 2 逻辑
    print(f"✓ 从知识库检索到 {len(rag_context)} 条相关信息")
    
    # Step 3: LLM 推理
    print("\n[Step 3] LLM 推理（选择的参数规模）...")
    start = time.time()
    response = llm_inference(segments, rag_context)  # Block 1 决定的模型
    inference_time = time.time() - start
    print(f"✓ 推理耗时: {inference_time:.3f}s (目标: < 0.5s)")
    
    # Step 4: 输出
    print("\n[Step 4] TTS 转语音输出...")
    print(f"💬 AI 回答: {response}")
    
    return {"response": response, "inference_time": inference_time}

# 辅助函数
def segment_input(text):
    """分段处理"""
    return text.split("。")

def retrieve_from_rag(segments):
    """从 RAG 检索相关知识"""
    return ["知识点1", "知识点2", "知识点3"]

def llm_inference(segments, context):
    """LLM 推理（这里用 Qwen）"""
    time.sleep(0.3)  # 模拟推理时间
    return "基于您的需求，我们建议采用西门子的工业控制系统..."

# 运行示例
if __name__ == "__main__":
    customer_query = """我们公司是制造企业。
生产效率低，不良率 15%。
想了解西门子的自动化解决方案。"""
    
    result = virtual_sales_pipeline(customer_query)
```

#### 互动环节
- 参会者提出的具体场景讨论
- 针对性的模型选择建议
- Q&A 时间

---

## 三、Notebook 实现风格

### 整体结构：总分总
- **导入部分**：用场景和问题引入，明确 MoE 误区
- **分块部分**：每个 Block 遵循"解释 + 可运行代码"的模式
- **总结部分**：完整 Pipeline 演示 + 参会者讨论

### 单个 Block 的标准格式

每个 Block（Block 1/2/3）采用以下结构：

```
【Markdown】问题陈述与背景
  ↓
【Markdown】解决思路分解（3-4 个要点）
  ↓
【Python 代码】可直接运行的实现示例
  ↓
【预期输出】明确说明代码输出结果
```

### 内容形式要求
1. **Markdown 说明**
   - 用清晰的文字解释概念
   - 用 ASCII 图或表格呈现关键对比
   - 用列表结构化复杂思路

2. **可运行代码**
   - 完整独立的 Python 脚本，无外部依赖
   - 包含数据构造 + 逻辑处理 + 输出显示的全链路
   - 代码中用注释解释每个步骤的意图

3. **魔法方法应用**（仅在需要时使用）
   - `%%time`：性能对比时使用（如 Block 1 的延迟估算）
   - `%%bash`：需要执行系统命令时使用（如下载示例数据）
   - `%%capture`：需要捕获输出进行后续处理时使用

### 代码风格规范
- 每个代码块都是**自洽的**，可独立运行
- 用 `print()` 做结构化输出，便于阅读
  ```python
  print("=" * 60)
  print("【步骤名称】")
  print(result)
  print("=" * 60)
  ```
- 用 Unicode 符号（✓、✗、→、↓）增强可视化
- 避免过深的嵌套，保持代码清晰可读

### 预期输出格式
每个代码块后明确说明输出内容：
```
【输出示例】
推理延迟估算 (假设平均回答50个token):
--------------------------------------------------
Qwen3-3B: 0.500s ✓ 可行
Qwen3-7B: 1.000s ✗ 超限
...

结论: Qwen3-7B 是最有效的平衡点
```

### 所有操作在 Notebook 中完成
- **Python 代码**：直接写在 Python cell 中
- **数据处理**：使用 pandas/numpy（无需下载外部文件）
- **可视化**：使用 matplotlib（不依赖特殊库）
- **系统命令**：通过 `%%bash` 魔法方法（若需要）
- **文档说明**：用 Markdown cell 清晰阐述

---

## 四、关键收获总结

| 块 | 核心问题 | 关键输出 |
|----|---------|---------| 
| Block 1 | 用什么参数的 Qwen？| 模型选择的量化依据 |
| Block 2 | 如何处理异构数据？| RAG 融合检索的实现 |
| Block 3 | 长输入怎么不出错？| 分段+聚合的完整方案 |

**最终价值**：参会者拿走可直接用于生产的技术方案和代码框架。

---

## 五、后续可选话题（不在本次 Workshop 中）

- MoE 路由策略（如果基础方案效果不达预期）
- 模型微调（如果知识库不足）
- 实时流处理（如果需要更低的延迟）
