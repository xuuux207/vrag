# 实验 2：企业图谱 + 业务文档的混合 RAG 检索

## 0. 问题背景与初衷

### 0.1 原始问题描述

**问题 2：RAG 与知识库方案 - 针对异构数据源的知识整合**

在虚拟销售 AI 场景中，销售人员需要同时掌握两类信息：

#### 左手：企业图谱数据（结构化）
- **数据来源**：类似天眼查、企查查的企业信息
- **数据结构**：实体-关系图谱
- **典型内容**：
  - 企业基本信息：注册资本、成立时间、行业分类、经营范围
  - 组织关系：股权结构、母子公司、法人/高管
  - 业务信息：资质证书、历史合作、行业地位
  - 关联关系：供应商、客户、竞争对手

**示例场景**：
```
销售：这家公司是谁？
系统：XX科技，成立于2015年，注册资本5000万，主营AI芯片研发
      母公司是YY集团（上市公司），CEO是张三（曾任华为高管）
      2023年获得B轮融资2亿，投资方包括红杉资本
```

#### 右手：业务文档（非结构化）
- **数据来源**：乙方（如 TechFlow）的业务材料
- **文档形式**：Word、PPT、PDF
- **内容类型**：文本、图片、图表
- **典型内容**：
  - 产品手册
  - 解决方案白皮书
  - 客户案例（成功案例、失败教训）
  - 技术文档、FAQ

**特点**：文档之间耦合性不强，每个文档相对独立

#### 核心挑战

1. **异构数据融合**：
   - 企业图谱是**结构化图数据**（实体 + 关系）
   - 业务文档是**非结构化文本**
   - 如何让 RAG 系统同时查询两者？

2. **检索策略**：
   - 企业名称、产品型号需要**精确匹配**（BM25 稀疏检索）
   - 需求描述需要**语义理解**（Dense 向量检索）
   - 如何设计混合检索策略？

3. **知识库设计**：
   - 如何组织企业图谱数据？
   - 如何与文档向量索引协同工作？
   - 如何形成"有效的炼丹炉"？

4. **实际应用场景**：
```
客户："我想了解XX公司的背景，他们之前有没有用过类似方案？
       适合推荐哪个产品？"

系统需要：
1. 查企业图谱 → 了解公司规模、行业、历史
2. 查合作记录 → 是否有类似项目经验
3. 查业务文档 → 匹配合适的产品和案例
4. 融合信息 → 给出综合推荐
```

### 0.2 实验目标

本实验要验证的核心问题：

1. ✅ **混合检索是否必要？**
   - 纯 Dense 向量 vs BM25 + Dense 混合
   - 在企业名称、产品型号等关键词匹配上的差异

2. ✅ **企业图谱如何有效融合？**
   - 图谱数据如何转化为 RAG 可用的格式
   - 关系查询（如"母公司是谁"）如何实现

3. ✅ **多源融合的价值？**
   - 单用企业图谱 vs 单用文档 vs 两者融合
   - 融合后回答的全面性和准确性提升

4. ✅ **实际工程可行性？**
   - 性能开销（混合检索、图谱查询）
   - 复杂度 vs 收益的权衡

---

## 1. 技术方案设计

### 1.1 混合检索架构（Hybrid Retrieval）

#### 为什么需要混合检索？

**场景 1：关键词精确匹配**
```
查询："DataStream Pro 的价格是多少？"

纯 Dense 向量：可能召回 "DataStream Lite" 或 "数据流处理平台" 的文档
BM25 稀疏检索：精确匹配 "DataStream Pro"，确保召回正确产品
```

**场景 2：企业名称匹配**
```
查询："XX科技之前有没有合作过？"

纯 Dense 向量：可能召回相似行业的公司
BM25：精确匹配 "XX科技"，定位到正确企业
```

**场景 3：语义理解**
```
查询："我们想降低成本，有什么方案？"

BM25：只能匹配 "成本" 关键词
Dense 向量：理解 "降低成本" ≈ "节省预算" ≈ "性价比高"
```

#### 混合检索流程

```python
# 第 1 步：BM25 稀疏检索（关键词精确匹配）
bm25_results = bm25_search(query, corpus, top_k=20)
# 优势：企业名、产品型号、技术名称等精确匹配

# 第 2 步：Dense 向量检索（语义匹配）
query_vector = embed(query)
dense_results = vector_search(query_vector, index, top_k=20)
# 优势：理解同义词、上下文、隐含需求

# 第 3 步：RRF 融合（Reciprocal Rank Fusion）
fused_results = rrf_fusion(bm25_results, dense_results, k=60)
# 公式：score(doc) = sum(1 / (k + rank_i)) for all rankings

# 第 4 步：Reranking 精排
final_results = reranking_service.rerank(query, fused_results, top_k=5)
# 使用 BAAI/bge-reranker-v2-m3 精确排序
```

**RRF 融合算法**：
```python
def rrf_fusion(rankings: List[List[str]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion

    参数：
        rankings: 多个排序结果 [[doc1, doc2, ...], [doc3, doc1, ...]]
        k: 平滑参数，避免头部文档权重过大

    返回：
        融合后的排序结果 [(doc_id, score), ...]
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 1.2 企业图谱融合方案

#### 方案 A：图谱数据向量化（推荐）✅

将企业图谱的**实体和关系**转化为文本描述，然后向量化索引：

```python
# 1. 企业实体转为文档
entity_doc = f"""
【企业】{company.name}
注册资本：{company.capital}
成立时间：{company.founded_date}
行业分类：{company.industry}
经营范围：{company.business_scope}
CEO：{company.ceo}
"""

# 2. 关系转为文档
relation_doc = f"""
【企业关系】{company.name}
母公司：{company.parent_company}
子公司：{', '.join(company.subsidiaries)}
投资方：{', '.join(company.investors)}
历史合作：{', '.join(company.past_projects)}
"""

# 3. 统一索引
all_docs = entity_docs + relation_docs + business_docs
vector_index.add_documents(all_docs)
```

**优势**：
- ✅ 统一检索流程，无需单独查询图数据库
- ✅ 支持混合检索（BM25 + Dense）
- ✅ 实现简单，适合原型验证

**劣势**：
- ⚠️ 复杂关系查询能力弱（如"3层股权关系"）
- ⚠️ 不适合超大规模图谱（>10万实体）

#### 方案 B：图数据库 + 文档索引并行（未来优化）

```python
# 1. 识别查询意图
if is_graph_query(query):
    # 查询 Neo4j 图数据库
    graph_results = neo4j_query(query)
else:
    # 查询文档索引
    doc_results = hybrid_search(query)

# 2. 融合结果
final_context = merge(graph_results, doc_results)
```

**本实验采用方案 A**（向量化），方案 B 留作后续优化方向

### 1.3 完整架构图

```
用户查询："XX科技适合推荐哪个产品？"
    ↓
【步骤 1】混合检索
    ├─ BM25 稀疏检索 → Top 20（精确匹配 "XX科技"）
    ├─ Dense 向量检索 → Top 20（语义理解 "推荐产品"）
    └─ RRF 融合 → Top 40
    ↓
【步骤 2】多源召回
    ├─ 企业图谱文档（XX科技的基本信息、关系）
    ├─ 业务文档（产品手册、案例）
    └─ 历史合作记录
    ↓
【步骤 3】Reranking 精排
    └─ BAAI/bge-reranker-v2-m3 → Top 5
    ↓
【步骤 4】上下文组织
    ├─ 企业背景（来自图谱）
    ├─ 相关产品（来自业务文档）
    └─ 相似案例（来自案例库）
    ↓
【步骤 5】LLM 生成回答
    └─ 基于 qwen3-8b + RAG（复用实验一结论）
```

### 1.4 Agent 式多步检索编排（复用原始设计亮点）

为保留原始设计中「基于 Agent 的多步检索」优势，在混合检索框架外再增加一个轻量 Agent 层，用于拆解复杂查询并按需触发不同数据源：

1. **意图识别**：LLM 先判断查询是否需要图谱关系、历史案例或产品策略等不同信息块。
2. **检索规划**：Agent 根据意图生成一个小型计划（如“先查企业背景→再拉案例→验证库存”）。
3. **多步执行**：每一步调用 `hybrid_search()`，但附带不同的系统提示/过滤条件（例如限定数据源、企业名称、时间范围）。
4. **上下文整合**：Agent 把每一步的成果写入记忆，直到覆盖查询中的所有槽位，再交给 LLM 生成最终回答。

伪代码示例：

```python
plan = agent.plan(query)
context_chunks = []
for step in plan:
    search_kwargs = step.to_kwargs()
    results, debug = hybrid_search(query=step.prompt, **search_kwargs)
    context_chunks.append({"step": step.name, "docs": results, "latency": debug["latency"]})

final_answer = llm.generate(query, context_chunks)
```

这种 Agent 式编排可以：
- ✅ 显式展示检索链路，方便结合延迟指标做逐步优化
- ✅ 支持“先结构化后非结构化”“图谱→文档→FAQ”一类多 Hop 检索
- ✅ 与实验一中的管控策略（如输出格式校验）保持一致，减少新增工程成本

---

## 2. 数据设计

### 2.0 数据总体原则（延续实验一主题）

- **行业背景沿用实验一**：继续围绕虚构厂商 TechFlow 与其虚拟客户生态，保证故事线一致、便于横向对比。
- **数据形态覆盖三类源**：结构化（企业图谱）、非结构化（业务文档）、知识图谱（显式关系边），与最初“左手 + 右手 + 图谱”设想一致。
- **完全虚构且可控**：所有实体、项目、关系均为虚构，避免命中真实训练语料；必要时通过脚本统一生成并打水印字段，如 `source: "synthetic_v2"`。
- **延迟可观测**：在生成数据时即记录 size、token 数等统计，方便后续估算检索与生成耗时。

### 2.1 虚构企业图谱（新增）

设计 **10 家虚构公司**，覆盖不同行业和规模：

```python
COMPANY_GRAPH = [
    {
        "id": "company_001",
        "name": "鼎盛科技有限公司",
        "industry": "智能制造",
        "founded_date": "2015-03-15",
        "capital": "5000万元",
        "business_scope": "工业自动化、数据采集、IoT平台",
        "employees": 500,
        "ceo": "李明",
        "parent_company": "鼎盛集团",
        "subsidiaries": ["鼎盛智能", "鼎盛数据"],
        "investors": ["红杉资本", "IDG"],
        "past_projects": [
            {
                "year": 2023,
                "project": "工厂数字化改造",
                "partner": "TechFlow",
                "product": "DataStream Lite",
                "result": "生产效率提升 35%"
            }
        ],
        "pain_points": ["生产数据孤岛", "设备故障率高", "库存管理混乱"]
    },
    {
        "id": "company_002",
        "name": "星辰金融集团",
        "industry": "金融科技",
        "founded_date": "2010-06-01",
        "capital": "2亿元",
        "business_scope": "证券交易、风控系统、支付平台",
        "employees": 2000,
        "ceo": "王芳",
        "parent_company": None,  # 独立集团
        "subsidiaries": ["星辰证券", "星辰支付", "星辰保险"],
        "investors": ["腾讯", "软银"],
        "past_projects": [],  # 未合作过
        "pain_points": ["实时风控延迟高", "交易数据处理瓶颈", "合规压力大"]
    },
    # ... 8 个其他公司（电商、物流、能源、医疗等行业）
]
```

**设计原则**：
- ✅ 行业多样化（制造、金融、电商、物流、能源、医疗等）
- ✅ 规模差异（初创、中型、大型）
- ✅ 合作历史差异（有合作、无合作、竞品客户）
- ✅ 痛点明确（可匹配 TechFlow 产品）

### 2.2 企业关系设计

```python
COMPANY_RELATIONS = [
    # 股权关系
    {"source": "鼎盛集团", "relation": "持股", "target": "鼎盛科技", "percentage": 80},
    {"source": "红杉资本", "relation": "投资", "target": "鼎盛科技", "round": "B轮", "amount": "2亿元"},

    # 合作关系
    {"source": "鼎盛科技", "relation": "使用产品", "target": "DataStream Lite", "year": 2023},
    {"source": "TechFlow", "relation": "服务客户", "target": "鼎盛科技", "satisfaction": 4.5},

    # 竞争关系
    {"source": "星辰金融", "relation": "竞品客户", "target": "XX竞品公司", "year": 2022},

    # 供应链关系
    {"source": "鼎盛科技", "relation": "供应商", "target": "华为云", "service": "云计算"},
]
```

### 2.3 业务文档（复用实验一）

```python
# 已有 16 个 TechFlow 虚构文档
# 位于 data/fictional_knowledge_base.py
FICTIONAL_DOCUMENTS = [
    # 产品手册（6个核心文档）
    # 干扰文档（6个近似产品）
    # 同义词文档（4个）
]
```

### 2.4 数据组织方式

将企业图谱转化为可检索的文档：

```python
def company_to_documents(company: Dict) -> List[Dict]:
    """将企业图谱转为多个检索文档"""

    docs = []

    # 文档 1：企业基本信息
    docs.append({
        "id": f"{company['id']}_basic",
        "type": "company_profile",
        "title": f"{company['name']} 企业档案",
        "content": f"""
【企业名称】{company['name']}
【行业】{company['industry']}
【成立时间】{company['founded_date']}
【注册资本】{company['capital']}
【员工规模】{company['employees']}人
【CEO】{company['ceo']}
【经营范围】{company['business_scope']}
        """
    })

    # 文档 2：企业关系
    docs.append({
        "id": f"{company['id']}_relations",
        "type": "company_relations",
        "title": f"{company['name']} 企业关系",
        "content": f"""
【企业名称】{company['name']}
【母公司】{company.get('parent_company', '无')}
【子公司】{', '.join(company.get('subsidiaries', []))}
【投资方】{', '.join(company.get('investors', []))}
        """
    })

    # 文档 3：合作历史
    if company.get('past_projects'):
        for project in company['past_projects']:
            docs.append({
                "id": f"{company['id']}_project_{project['year']}",
                "type": "project_history",
                "title": f"{company['name']} 合作案例（{project['year']}）",
                "content": f"""
【客户】{company['name']}
【项目】{project['project']}
【时间】{project['year']}年
【合作方】{project['partner']}
【使用产品】{project['product']}
【项目成果】{project['result']}
                """
            })

    # 文档 4：痛点分析
    docs.append({
        "id": f"{company['id']}_painpoints",
        "type": "company_needs",
        "title": f"{company['name']} 业务痛点",
        "content": f"""
【企业名称】{company['name']}
【行业】{company['industry']}
【当前痛点】
{chr(10).join(f"- {pain}" for pain in company['pain_points'])}
        """
    })

    return docs
```

### 2.5 多源耦合策略

- **结构化 → 文本化**：企业图谱（结构化表 + 知识图谱）先按 `company_to_documents()` 规则生成描述性段落，再与实验一复用的 FICTIONAL_DOCUMENTS 一起进入同一检索索引，保证“左手 + 右手”真正融合。
- **知识图谱保持显式边**：除文本化外，`COMPANY_RELATIONS` 仍以边列表方式存储，供 Agent 在需要“母子公司/投资链”推理时直接遍历，保留原始设计中“结构化 + 知识图谱”并存的能力。
- **主题对齐**：所有公司、产品均围绕 TechFlow 及其实验一虚构产品线展开，回答中可以自然引用实验一的评测结论（例如推荐 qwen3-8b）。
- **可扩展性**：将三类数据都标注 `source_type`（graph/doc/edge），便于统计不同数据源的贡献度和延迟，占位以支持后续接入真实企业数据。

---

## 3. 测试用例设计

### 测试用例 1：企业背景查询（图谱检索）
```python
{
    "id": "fusion_test_001",
    "category": "企业背景查询",
    "query": "鼎盛科技是做什么的？公司规模多大？",
    "expected_sources": {
        "company_profile": True,  # 企业档案
        "business_docs": False     # 不需要业务文档
    },
    "expected_key_points": [
        "智能制造行业",
        "成立于2015年",
        "注册资本5000万",
        "员工500人",
        "CEO 李明"
    ],
    "expected_bm25_advantage": True  # BM25 应精确匹配 "鼎盛科技"
}
```

### 测试用例 2：合作历史查询（图谱 + 关系）
```python
{
    "id": "fusion_test_002",
    "category": "合作历史查询",
    "query": "鼎盛科技之前有没有用过我们的产品？效果怎么样？",
    "expected_sources": {
        "project_history": True,   # 合作案例
        "company_profile": True,   # 企业基本信息
        "business_docs": True      # 产品文档
    },
    "expected_key_points": [
        "2023年合作过",
        "使用 DataStream Lite",
        "工厂数字化改造项目",
        "生产效率提升 35%"
    ]
}
```

### 测试用例 3：产品推荐（图谱 + 文档融合）
```python
{
    "id": "fusion_test_003",
    "category": "产品推荐",
    "query": "星辰金融集团想做实时风控，应该推荐哪个产品？",
    "expected_sources": {
        "company_profile": True,     # 了解客户（金融行业）
        "company_needs": True,       # 痛点（实时风控延迟高）
        "business_docs": True        # 产品文档
    },
    "expected_key_points": [
        "金融行业客户",
        "推荐 DataStream Pro",
        "适合实时风控场景",
        "低延迟（10ms）",
        "金融行业成功案例"
    ],
    "expected_fusion_value": True  # 需要融合企业信息 + 产品信息
}
```

### 测试用例 4：关系查询（图谱推理）
```python
{
    "id": "fusion_test_004",
    "category": "关系查询",
    "query": "鼎盛科技的母公司是谁？投资方有哪些？",
    "expected_sources": {
        "company_relations": True,  # 企业关系
        "company_profile": False
    },
    "expected_key_points": [
        "母公司是鼎盛集团",
        "投资方包括红杉资本、IDG",
        "B轮融资2亿元"
    ],
    "expected_bm25_advantage": True  # 需要精确匹配公司名称
}
```

### 测试用例 5：竞品客户识别（陷阱测试）
```python
{
    "id": "fusion_test_005",
    "category": "竞品客户识别",
    "query": "星辰金融之前有没有合作过？",
    "expected_sources": {
        "project_history": True,
        "company_relations": True
    },
    "expected_behavior": "识别出该客户使用竞品，谨慎推荐或说明差异化优势",
    "expected_key_points": [
        "星辰金融未合作过",
        "但使用过竞品 XX",
        "我们的产品相比竞品的优势..."
    ]
}
```

### 测试用例 6：混合检索对比（BM25 vs Dense）
```python
{
    "id": "fusion_test_006",
    "category": "混合检索效果验证",
    "query": "DataStream Pro 适合哪些行业？",
    "baseline_a": "仅 BM25 检索",
    "baseline_b": "仅 Dense 向量检索",
    "proposed": "BM25 + Dense 混合",
    "expected_result": "混合检索应召回更全面（产品文档 + 行业案例 + 客户档案）"
}
```

---

## 4. 评估指标

### 4.1 混合检索效果

| 指标 | 说明 | 计算方式 |
|------|------|----------|
| **召回率（Recall）** | 相关文档被召回的比例 | 召回相关文档数 / 总相关文档数 |
| **精确率（Precision）** | 召回文档中相关的比例 | 相关文档数 / 召回文档数 |
| **MRR（Mean Reciprocal Rank）** | 首个相关文档的排名 | 平均(1 / 首个相关文档排名) |
| **BM25 vs Dense 差异** | 两者召回文档的重叠度 | 交集 / 并集 |

### 4.2 多源融合质量

| 维度 | 说明 | 评分方式 |
|------|------|----------|
| **数据源覆盖** | 是否召回了所需的数据源 | 企业图谱/业务文档 命中情况 |
| **关系推理准确性** | 企业关系查询是否正确 | 母公司/投资方/合作历史 准确率 |
| **信息融合度** | 多源信息是否有机结合 | LLM-as-judge 评分 |
| **回答全面性** | 是否同时包含背景+推荐+案例 | 关键点覆盖率 |

### 4.3 对比实验

| 方案 | 数据源 | 检索方式 | 预期效果 |
|------|--------|----------|----------|
| **Baseline A** | 仅业务文档 | Dense 向量 | 缺少客户背景，推荐不精准 |
| **Baseline B** | 仅企业图谱 | BM25 | 有背景但无产品细节 |
| **Baseline C** | 图谱+文档 | Dense 向量 | 融合但关键词匹配弱 |
| **方案 D（推荐）** | 图谱+文档 | BM25 + Dense 混合 | 最全面准确 |

### 4.4 延迟可观测性（客户延时敏感需求）

- **分阶段打点**：`hybrid_search()` 输出的 `debug_info["latency"]` 记录 BM25、Dense、RRF、候选准备、Rerank 等耗时；Agent 层再额外记录规划、LLM 生成的耗时，形成“检索→生成”全链路时间线。
- **可视化方式**：实验脚本生成 `outputs/experiment2_latency_breakdown.json` 与折线/堆叠柱状图，展示不同方案、不同公司查询的 P50/P95 延迟。
- **告警阈值**：设置 500ms 软阈值；若某阶段超阈值即写入 log，并在回答附带“本次检索耗时偏高”提示，便于与客户同步预期。
- **采样方法**：每个测试用例跑 ≥5 次，附带上下文 token 统计，分析上下文长度与延迟的关系，为后续优化（如缓存、裁剪）提供依据。

---

## 5. 实验实施计划

### 阶段 1：数据准备（Day 1）
- [ ] 设计 10 家虚构企业图谱
- [ ] 设计企业关系网络
- [ ] 将图谱数据转为检索文档格式
- [ ] 保存到 `data/company_graph.py`

### 阶段 2：混合检索实现（Day 1-2）
- [ ] 实现 BM25 检索模块
- [ ] 实现 RRF 融合算法
- [ ] 集成到现有 RAG 流程
- [ ] 更新 `rag_utils.py`

### 阶段 3：实验执行（Day 2）
- [ ] 实现 `experiments/test_02_hybrid_rag.py`
- [ ] 运行 4 种方案对比
- [ ] 收集混合检索 vs 纯向量的差异数据
- [ ] 使用 LLM-as-judge 评分

### 阶段 4：分析与报告（Day 3）
- [ ] 分析混合检索的提升效果
- [ ] 分析企业图谱融合的价值
- [ ] 生成报告 `outputs/experiment2_hybrid_rag_analysis.md`

---

## 6. 核心技术模块

### 6.1 BM25 实现

```python
import math
from collections import Counter
from typing import List, Dict

class BM25:
    """BM25 算法实现"""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        """
        参数：
            corpus: 文档列表
            k1: 词频饱和参数（通常 1.2-2.0）
            b: 长度归一化参数（0-1，0.75 常用）
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus)
        self.doc_freqs = []
        self.idf = {}

        # 计算 IDF
        df = Counter()
        for doc in corpus:
            tokens = set(doc.split())
            df.update(tokens)

        for term, freq in df.items():
            self.idf[term] = math.log((len(corpus) - freq + 0.5) / (freq + 0.5) + 1)

        # 计算每个文档的词频
        for doc in corpus:
            self.doc_freqs.append(Counter(doc.split()))

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        检索最相关的文档

        返回：[(doc_idx, score), ...]
        """
        query_terms = query.split()
        scores = []

        for idx, doc_freq in enumerate(self.doc_freqs):
            score = 0
            for term in query_terms:
                if term not in doc_freq:
                    continue

                freq = doc_freq[term]
                idf = self.idf.get(term, 0)

                # BM25 公式
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                score += idf * (numerator / denominator)

            scores.append((idx, score))

        # 排序返回 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

### 6.2 混合检索主函数

```python
import time

def hybrid_search(
    query: str,
    bm25_index: BM25,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    reranking_service: RerankingService,
    bm25_top_k: int = 20,
    dense_top_k: int = 20,
    final_top_k: int = 5
) -> Tuple[List[Dict], Dict]:
    """
    混合检索主流程

    返回：
        - final_results: 最终 Top-K 结果
        - debug_info: 调试信息（BM25/Dense/RRF 各阶段结果）
    """

    latency = {}

    # 步骤 1：BM25 稀疏检索
    t0 = time.perf_counter()
    bm25_results = bm25_index.search(query, top_k=bm25_top_k)
    latency["bm25_ms"] = (time.perf_counter() - t0) * 1000
    bm25_doc_ids = [idx for idx, score in bm25_results]

    # 步骤 2：Dense 向量检索
    t0 = time.perf_counter()
    query_vector = embedding_service.embed(query)
    dense_results = vector_index.search(query_vector, top_k=dense_top_k)
    latency["dense_ms"] = (time.perf_counter() - t0) * 1000
    dense_doc_ids = [r['doc_id'] for r in dense_results]

    # 步骤 3：RRF 融合
    t0 = time.perf_counter()
    rrf_results = rrf_fusion([bm25_doc_ids, dense_doc_ids], k=60)
    latency["rrf_ms"] = (time.perf_counter() - t0) * 1000

    # 步骤 4：取 Top-40 候选文档
    t0 = time.perf_counter()
    candidate_doc_ids = [doc_id for doc_id, score in rrf_results[:40]]
    candidate_docs = [vector_index.get_document(doc_id) for doc_id in candidate_doc_ids]
    latency["candidate_fetch_ms"] = (time.perf_counter() - t0) * 1000

    # 步骤 5：Reranking 精排
    t0 = time.perf_counter()
    final_results = reranking_service.rerank(
        query=query,
        documents=candidate_docs,
        top_k=final_top_k
    )
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000
    latency["retrieval_total_ms"] = sum(latency.values())

    # 调试信息
    debug_info = {
        "bm25_results": bm25_doc_ids[:5],
        "dense_results": dense_doc_ids[:5],
        "rrf_fused": candidate_doc_ids[:10],
        "final_reranked": [r['doc_id'] for r in final_results],
        "overlap_bm25_dense": len(set(bm25_doc_ids) & set(dense_doc_ids)),
        "latency": latency
    }

    return final_results, debug_info
```

---

## 7. 预期产出

### 7.1 代码文件
- `data/company_graph.py`（企业图谱数据）
- `rag_utils.py`（新增 BM25、RRF 融合）
- `experiments/test_02_hybrid_rag.py`（实验脚本）

### 7.2 实验结果
- `outputs/experiment2_results_YYYYMMDD_HHMMSS.json`
- `outputs/experiment2_hybrid_rag_analysis.md`
- `outputs/experiment2_latency_breakdown.json`（分阶段耗时）
- `outputs/experiment2_latency_viz.png`（可视化图表）

### 7.3 关键发现（预期）
- 混合检索相比纯向量检索，召回率提升 XX%
- 企业图谱融合使回答全面性提升 XX%
- BM25 在企业名称、产品型号匹配上准确率 XX%
- 多源融合质量评分提升 XX 分

---

## 8. 技术亮点

### 8.1 混合检索的必要性
- BM25 保证关键词精确匹配（企业名、产品型号）
- Dense 向量补充语义理解（同义词、隐含需求）
- RRF 融合平衡两者，避免单一算法的局限性

### 8.2 企业图谱轻量化融合
- 无需引入图数据库，降低技术复杂度
- 将实体和关系转为文本文档，统一检索流程
- 适合中小规模场景（< 1万企业）

### 8.3 回到问题本质
- **左手（企业图谱）+ 右手（业务文档）** 真正融合
- 虚拟销售既能了解客户背景，又能精准推荐产品
- 形成"有效的炼丹炉"：数据 → 知识 → 智能推荐

---

## 9. 与实验一的关联

| 维度 | 实验一 | 实验二 |
|------|--------|--------|
| **核心问题** | 验证 RAG 减少幻觉 | 验证多源融合 + 混合检索 |
| **数据源** | 仅非结构化文档 | 企业图谱 + 业务文档 |
| **检索方式** | Dense 向量 + Reranking | BM25 + Dense + RRF + Reranking |
| **模型选择** | 对比 3 个模型 | 复用 qwen3-8b（实验一结论） |
| **评分方式** | LLM-as-judge | 复用实验一评分器 |
| **技术复杂度** | 中 | 高 |

---

## 10. 后续优化方向

### 短期优化
- 支持中文分词（jieba）优化 BM25
- 调优 RRF 参数 k（当前 60）
- 添加查询改写（Query Rewrite）

### 中期优化
- 引入 Elasticsearch 加速 BM25
- 支持实体链接（Entity Linking）
- 添加时间维度（企业信息更新）

### 长期优化
- 引入图数据库（Neo4j）支持复杂关系查询
- 实现动态检索策略（LLM 决定用 BM25 还是 Dense）
- 多模态支持（图表、PPT 内容提取）

---

**文档版本**：v2.0（重新设计）
**创建日期**：2025-12-14
**实验状态**：⏳ 设计完成，待实施
**关键变更**：回归问题初衷（企业图谱 + 业务文档融合），增加混合检索设计
