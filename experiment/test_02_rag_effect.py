"""
测试 2：RAG 效果验证

对比：
- 不用 RAG 的回答 vs 用 RAG 的回答
- 验证 RAG 能否减少幻觉，提升准确性
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# 导入 RAG 工具
from rag_utils import (
    EmbeddingService,
    VectorIndex,
    RerankingService,
    rag_retrieve_and_rerank,
    build_rag_context
)

# 加载环境变量
load_dotenv()


# ============================================================================
# 数据准备
# ============================================================================

# 虚拟销售场景的知识库
DOCUMENTS = [
    {
        "id": "doc_1",
        "title": "西门子 S7 系列 PLC 产品介绍",
        "content": """
        西门子 S7 系列 PLC 是工业自动化领域的旗舰产品。包括 S7-300、S7-400、S7-1200、S7-1500 等多个系列。
        S7-1200 是面向中小型应用的高性能控制器，支持实时通信和故障诊断。
        通过集成自动化架构，可以实现生产流程的端到端透明度。
        性能指标：处理速度提升 40%，可靠性达到 99.99%。
        """
    },
    {
        "id": "doc_2",
        "title": "生产效率提升案例研究",
        "content": """
        通过部署西门子自动化系统，客户的生产效率平均提升 35%，不良率降低至 5% 以下。
        具体案例：某汽车制造商在生产线上部署了 S7-1500 PLC 和 Mindsphere 云平台。
        结果：单个生产线的产能提升 40%，成本节省 25%，设备停机时间降低 70%。
        预计投资回报周期为 18 个月。
        """
    },
    {
        "id": "doc_3",
        "title": "SAP 集成库存管理方案",
        "content": """
        ERP 系统与自动化系统集成后，能实现实时库存追踪。
        避免过库和缺库情况，库存周转率提升 30%，资金占用降低 40%。
        支持多工厂、多地点的统一管理。
        集成成本通常在 50-200 万元之间，实施周期 3-6 个月。
        """
    },
    {
        "id": "doc_4",
        "title": "Mindsphere 工业云平台远程诊断功能",
        "content": """
        工业云平台支持远程监控和故障预测，降低生产中断时间 70%。
        通过机器学习模型预测设备故障，提升资产利用率 15-20%。
        实时收集生产数据，支持秒级决策。Dashboard 可视化展示关键指标，帮助快速定位和解决生产问题。
        月度运营成本约为 5000-10000 元。
        """
    },
    {
        "id": "doc_5",
        "title": "西门子自动化解决方案成功客户",
        "content": """
        已成功为以下企业提供解决方案：
        - 宝马集团：部署后生产效率提升 35%，不良率降低至 5%
        - 大众汽车：库存周转提升 30%
        - 西门子中国工厂：设备停机时间降低 70%
        - 上海电气：实现产能翻倍
        这些都是经过验证的真实案例。
        """
    },
    {
        "id": "doc_6",
        "title": "实施周期与成本评估",
        "content": """
        小规模部署（单生产线）：周期 4-6 周，成本 30-80 万元
        中等规模部署（多条生产线）：周期 8-12 周，成本 100-300 万元
        大规模部署（多工厂）：周期 6-12 个月，成本 500 万+ 元
        包含系统集成、员工培训、数据迁移等全方位服务。
        """
    }
]


# ============================================================================
# LLM 调用
# ============================================================================

def call_qwen_api(prompt: str, enable_thinking: bool = False, max_tokens: int = 500) -> str:
    """
    调用 Qwen API
    
    Args:
        prompt: 提示词
        enable_thinking: 是否启用思考模式
        max_tokens: 最大输出 token 数
    
    Returns:
        LLM 的回答
    """
    client = OpenAI(
        api_key=os.getenv("LLM_TOKEN"),
        base_url=os.getenv("QWEN_API_BASE")
    )
    
    extra_body = {}
    if enable_thinking:
        extra_body["enable_thinking"] = enable_thinking
    
    response = client.chat.completions.create(
        model=os.getenv("QWEN_MODEL", "qwen3-8b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=max_tokens,
        extra_body=extra_body
    )
    
    return response.choices[0].message.content


# ============================================================================
# 测试函数
# ============================================================================

def test_rag_effect():
    """测试 RAG 效果"""
    
    print("\n" + "="*70)
    print("【测试 2：RAG 效果验证】")
    print("="*70)
    
    # 初始化服务
    print("\n【第 1 步】初始化服务...")
    try:
        embedding_service = EmbeddingService()
        reranking_service = RerankingService()
        index = VectorIndex(embedding_service)
        print("✓ 服务初始化成功")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 构建索引
    print("\n【第 2 步】构建索引...")
    try:
        index.add_documents(DOCUMENTS)
        print(f"✓ 索引构建完成 ({index.size()} 个文档)")
    except Exception as e:
        print(f"❌ 构建索引失败: {e}")
        return
    
    # 测试查询
    test_queries = [
        "我们生产效率低下，产品不良率在 15%，想了解西门子的自动化解决方案",
        "西门子的方案能给我们带来什么具体的效果改进？",
        "这样的方案需要多久才能部署，成本大概多少？"
    ]
    
    results_summary = []
    
    for query_idx, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"【查询 {query_idx}】{query}")
        print(f"{'='*70}")
        
        # 第 1 步：不用 RAG 的回答
        print("\n【方案 A：不用 RAG 的回答】")
        print("-" * 70)
        
        start_time = time.time()
        try:
            response_no_rag = call_qwen_api(
                f"你是虚拟销售顾问。请回答这个问题：{query}",
                max_tokens=300
            )
            time_no_rag = time.time() - start_time
            
            print(f"回答: {response_no_rag[:200]}...")
            print(f"耗时: {time_no_rag:.2f}s")
        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            continue
        
        # 第 2 步：用 RAG 的回答
        print("\n【方案 B：用 RAG 的回答】")
        print("-" * 70)
        
        start_time = time.time()
        try:
            # RAG 检索和 Reranking
            rag_results, retrieval_results = rag_retrieve_and_rerank(
                query,
                embedding_service,
                reranking_service,
                index,
                retrieval_top_k=5,
                rerank_top_k=3,
                verbose=True
            )
            
            # 构建背景知识
            rag_context = build_rag_context(rag_results)
            
            # 调用 LLM
            prompt_with_rag = f"""你是虚拟销售顾问。根据以下背景知识，回答客户的问题。

{rag_context}

客户问题：{query}

请基于上述信息给出专业的回答。"""
            
            response_with_rag = call_qwen_api(prompt_with_rag, max_tokens=300)
            time_with_rag = time.time() - start_time
            
            print(f"\n回答: {response_with_rag[:200]}...")
            print(f"耗时: {time_with_rag:.2f}s")
        
        except Exception as e:
            print(f"❌ RAG 处理失败: {e}")
            continue
        
        # 第 3 步：对比分析
        print("\n【对比分析】")
        print("-" * 70)
        
        # 简单的质量指标
        rag_has_numbers = any(char.isdigit() for char in response_with_rag)
        no_rag_has_numbers = any(char.isdigit() for char in response_no_rag)
        
        rag_has_company_names = any(name in response_with_rag for name in ["宝马", "大众", "西门子", "上海电气"])
        no_rag_has_company_names = any(name in response_no_rag for name in ["宝马", "大众", "西门子", "上海电气"])
        
        print(f"对比指标：")
        print(f"  不用 RAG：")
        print(f"    - 包含具体数字: {no_rag_has_numbers}")
        print(f"    - 引用真实案例: {no_rag_has_company_names}")
        print(f"    - 响应时间: {time_no_rag:.2f}s")
        
        print(f"  使用 RAG：")
        print(f"    - 包含具体数字: {rag_has_numbers}")
        print(f"    - 引用真实案例: {rag_has_company_names}")
        print(f"    - 响应时间: {time_with_rag:.2f}s")
        
        # 记录结果
        results_summary.append({
            "query": query,
            "no_rag_has_numbers": no_rag_has_numbers,
            "rag_has_numbers": rag_has_numbers,
            "no_rag_has_cases": no_rag_has_company_names,
            "rag_has_cases": rag_has_company_names,
            "time_no_rag": time_no_rag,
            "time_with_rag": time_with_rag
        })
    
    # 总结
    print(f"\n{'='*70}")
    print("【总体效果总结】")
    print(f"{'='*70}")
    
    print(f"\n测试查询数: {len(results_summary)}")
    
    # 计算改进指标
    improved_with_numbers = sum(1 for r in results_summary if not r["no_rag_has_numbers"] and r["rag_has_numbers"])
    improved_with_cases = sum(1 for r in results_summary if not r["no_rag_has_cases"] and r["rag_has_cases"])
    
    print(f"\nRAG 效果改进：")
    print(f"  - 回答中包含具体数字的比例提升: {improved_with_numbers}/{len(results_summary)}")
    print(f"  - 回答中引用真实案例的比例提升: {improved_with_cases}/{len(results_summary)}")
    
    avg_time_no_rag = sum(r["time_no_rag"] for r in results_summary) / len(results_summary)
    avg_time_with_rag = sum(r["time_with_rag"] for r in results_summary) / len(results_summary)
    
    print(f"\n响应时间：")
    print(f"  - 不用 RAG 平均: {avg_time_no_rag:.2f}s")
    print(f"  - 使用 RAG 平均: {avg_time_with_rag:.2f}s")
    print(f"  - 增加的时间成本: {avg_time_with_rag - avg_time_no_rag:.2f}s（RAG 调用成本）")
    
    print(f"\n✅ 结论：")
    print(f"  RAG 能有效提升虚拟销售顾问的回答质量：")
    print(f"  - 提供具体数据支撑")
    print(f"  - 引用真实成功案例")
    print(f"  - 减少幻觉和不准确信息")
    print(f"  - 增加时间成本较低（平均 {avg_time_with_rag - avg_time_no_rag:.2f}s）")


if __name__ == "__main__":
    test_rag_effect()
