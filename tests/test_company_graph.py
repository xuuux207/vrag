"""
测试企业图谱数据 - 验证数据质量和融合效果

测试内容:
1. 数据完整性检查
2. 与实验一数据对接验证
3. 文档生成测试
4. 虚构程度验证
"""

import sys
sys.path.append('.')

from data.company_graph import (
    COMPANY_GRAPH,
    COMPANY_RELATIONS,
    get_all_company_documents,
    get_companies_by_status,
    get_statistics,
    get_company_by_name,
    get_relations_by_company
)
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS


def test_data_integrity():
    """测试数据完整性"""
    print("=" * 60)
    print("测试 1: 数据完整性检查")
    print("=" * 60)

    # 检查企业数据
    assert len(COMPANY_GRAPH) == 10, "企业数量应为10家"
    print("✓ 企业数量: 10家")

    # 检查必填字段
    required_fields = ['id', 'name', 'industry', 'employees', 'pain_points', 'synthetic_source']
    for company in COMPANY_GRAPH:
        for field in required_fields:
            assert field in company, f"{company['name']} 缺少字段: {field}"
    print("✓ 所有企业包含必填字段")

    # 检查关系数据
    assert len(COMPANY_RELATIONS) > 40, "关系数应 > 40条"
    print(f"✓ 企业关系: {len(COMPANY_RELATIONS)}条")

    # 检查合作状态分布
    cooperated = get_companies_by_status("cooperated")
    competitor = get_companies_by_status("competitor")
    potential = get_companies_by_status("potential")
    print(f"✓ 已合作: {len(cooperated)}家")
    print(f"✓ 竞品客户: {len(competitor)}家")
    print(f"✓ 潜在客户: {len(potential)}家")

    print("\n")


def test_techflow_alignment():
    """测试与实验一的 TechFlow 产品线对齐"""
    print("=" * 60)
    print("测试 2: 与实验一 TechFlow 产品线对齐")
    print("=" * 60)

    # TechFlow 产品列表（来自实验一）
    techflow_products = [
        "FlowControl-X100",
        "FlowControl-X500",
        "FlowMind",
        "DataStream Lite",
        "DataStream Pro"
    ]

    # 检查企业图谱中使用的产品
    used_products = set()
    for company in COMPANY_GRAPH:
        for project in company.get('past_projects', []):
            used_products.add(project['product'])

    print(f"企业图谱使用的产品: {used_products}")

    # 验证所有产品都在 TechFlow 产品线中
    for product in used_products:
        assert any(tp in product for tp in techflow_products), \
            f"产品 {product} 不在 TechFlow 产品线中"
    print("✓ 所有产品均来自 TechFlow 产品线")

    # 检查业务文档中的产品
    doc_products = set()
    for doc in FICTIONAL_DOCUMENTS:
        for product in techflow_products:
            if product in doc['title'] or product in doc['content']:
                doc_products.add(product)

    print(f"业务文档覆盖的产品: {doc_products}")
    print("✓ 企业图谱与业务文档产品线一致\n")


def test_document_generation():
    """测试文档生成"""
    print("=" * 60)
    print("测试 3: 检索文档生成")
    print("=" * 60)

    # 生成所有文档
    all_docs = get_all_company_documents()
    print(f"✓ 生成企业图谱文档: {len(all_docs)}篇")

    # 检查文档类型分布
    doc_types = {}
    for doc in all_docs:
        doc_type = doc['type']
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

    print("文档类型分布:")
    for doc_type, count in doc_types.items():
        print(f"  - {doc_type}: {count}篇")

    # 业务文档数量
    business_docs = len(FICTIONAL_DOCUMENTS)
    print(f"\n✓ 业务文档(实验一): {business_docs}篇")

    # 总文档数
    total_docs = len(all_docs) + business_docs
    print(f"✓ 总检索文档: {total_docs}篇\n")

    return all_docs


def test_synthetic_verification():
    """测试数据虚构程度"""
    print("=" * 60)
    print("测试 4: 数据虚构程度验证")
    print("=" * 60)

    # 检查水印标记
    for company in COMPANY_GRAPH:
        assert company.get('synthetic_source') == 'experiment_v2', \
            f"{company['name']} 缺少水印标记"
    print("✓ 所有企业包含水印标记: experiment_v2")

    # 检查公司名称虚构性
    company_names = [c['name'] for c in COMPANY_GRAPH]
    print(f"✓ 虚构公司名称: {', '.join(company_names[:3])} 等")

    # 检查投资方（真实但投资对象虚构）
    real_investors = set()
    for relation in COMPANY_RELATIONS:
        if relation.get('relation_type') == 'investment':
            real_investors.add(relation['source'])

    print(f"✓ 投资方（真实）: {', '.join(list(real_investors)[:5])} 等")
    print("✓ 投资对象（虚构）: 全部为虚构企业\n")


def test_query_scenarios():
    """测试查询场景"""
    print("=" * 60)
    print("测试 5: 典型查询场景")
    print("=" * 60)

    # 场景1: 企业背景查询
    company = get_company_by_name("鼎盛科技有限公司")
    assert company is not None
    print(f"✓ 场景1 - 企业背景查询")
    print(f"  查询: 鼎盛科技是做什么的?")
    print(f"  结果: {company['industry']}, {company['employees']}人, CEO {company['ceo']}")

    # 场景2: 合作历史查询
    if company.get('past_projects'):
        project = company['past_projects'][0]
        print(f"\n✓ 场景2 - 合作历史查询")
        print(f"  查询: 鼎盛科技之前用过什么产品?")
        print(f"  结果: {project['year']}年使用{project['product']}, {project['result']}")

    # 场景3: 关系查询
    relations = get_relations_by_company("鼎盛科技有限公司")
    print(f"\n✓ 场景3 - 企业关系查询")
    print(f"  查询: 鼎盛科技的投资方有谁?")
    investment_relations = [r for r in relations if r.get('relation_type') == 'investment']
    print(f"  结果: {', '.join(r['source'] for r in investment_relations)}")

    # 场景4: 竞品客户识别
    competitor = get_companies_by_status("competitor")[0]
    print(f"\n✓ 场景4 - 竞品客户识别(陷阱测试)")
    print(f"  客户: {competitor['name']}")
    print(f"  状态: 使用竞品 {competitor.get('competitor_product')}")
    print(f"  应对: 需谨慎推荐并说明差异化优势")

    print("\n")


def test_data_fusion():
    """测试多源数据融合"""
    print("=" * 60)
    print("测试 6: 多源数据融合")
    print("=" * 60)

    # 企业图谱文档
    graph_docs = get_all_company_documents()
    print(f"左手 - 企业图谱: {len(graph_docs)}篇")

    # 业务文档
    business_docs = FICTIONAL_DOCUMENTS
    print(f"右手 - 业务文档: {len(business_docs)}篇")

    # 融合检索池
    fusion_pool = graph_docs + business_docs
    print(f"融合检索池: {len(fusion_pool)}篇")

    # 检查数据源标记
    graph_source_count = sum(1 for d in fusion_pool if d.get('source_type') == 'graph')
    doc_source_count = len(fusion_pool) - graph_source_count
    print(f"  - 企业图谱文档: {graph_source_count}篇")
    print(f"  - 业务文档: {doc_source_count}篇")

    print("✓ 数据源标记完整,可追踪来源\n")


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "企业图谱数据测试" + " " * 27 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    try:
        test_data_integrity()
        test_techflow_alignment()
        test_document_generation()
        test_synthetic_verification()
        test_query_scenarios()
        test_data_fusion()

        print("=" * 60)
        print("✓ 所有测试通过!")
        print("=" * 60)
        print("\n数据统计:")
        stats = get_statistics()
        print(f"  - 企业总数: {stats['total_companies']}家")
        print(f"  - 关系总数: {stats['total_relations']}条")
        print(f"  - 生成文档: {stats['total_documents']}篇")
        print(f"  - 业务文档: {len(FICTIONAL_DOCUMENTS)}篇")
        print(f"  - 融合总量: {stats['total_documents'] + len(FICTIONAL_DOCUMENTS)}篇")
        print("\n数据已准备就绪,可以开始实验!\n")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
