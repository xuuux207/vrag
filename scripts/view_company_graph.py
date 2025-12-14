"""
企业图谱数据快速查看工具

用法:
  uv run python scripts/view_company_graph.py --company "鼎盛科技"
  uv run python scripts/view_company_graph.py --relations
  uv run python scripts/view_company_graph.py --stats
"""

import sys
sys.path.append('.')

import argparse
import json
from data.company_graph import (
    COMPANY_GRAPH,
    COMPANY_RELATIONS,
    get_company_by_name,
    get_relations_by_company,
    get_companies_by_status,
    get_statistics,
    company_to_documents
)


def view_company_detail(company_name: str):
    """查看企业详情"""
    company = get_company_by_name(company_name)
    if not company:
        print(f"未找到企业: {company_name}")
        return

    print("\n" + "=" * 60)
    print(f"企业档案: {company['name']}")
    print("=" * 60)
    print(f"ID: {company['id']}")
    print(f"行业: {company['industry']}")
    print(f"成立时间: {company['founded_date']}")
    print(f"注册资本: {company['capital']}")
    print(f"员工规模: {company['employees']}人")
    print(f"CEO: {company['ceo']}")
    print(f"母公司: {company.get('parent_company', '无')}")
    print(f"子公司: {', '.join(company.get('subsidiaries', []))}")
    print(f"投资方: {', '.join(company.get('investors', []))}")

    print(f"\n【业务痛点】")
    for pain in company['pain_points']:
        print(f"  - {pain}")

    if company.get('past_projects'):
        print(f"\n【合作历史】")
        for project in company['past_projects']:
            print(f"  • {project['year']}年: {project['project']}")
            print(f"    合作方: {project['partner']}")
            print(f"    使用产品: {project['product']}")
            print(f"    项目成果: {project['result']}")
            print(f"    投资金额: {project.get('investment', '未公开')}")

    if company.get('competitor_product'):
        print(f"\n【竞品信息】")
        print(f"  使用竞品: {company['competitor_product']}")

    if company.get('note'):
        print(f"\n【备注】")
        print(f"  {company['note']}")

    # 企业关系
    relations = get_relations_by_company(company['name'])
    if relations:
        print(f"\n【企业关系】({len(relations)}条)")
        relation_types = {}
        for rel in relations:
            rel_type = rel.get('relation_type', '其他')
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1

        for rel_type, count in relation_types.items():
            print(f"  - {rel_type}: {count}条")

    # 生成的检索文档
    docs = company_to_documents(company)
    print(f"\n【检索文档】({len(docs)}篇)")
    for doc in docs:
        print(f"  - {doc['type']}: {doc['title']}")

    print("=" * 60 + "\n")


def view_all_relations():
    """查看所有关系"""
    print("\n" + "=" * 60)
    print("企业关系网络")
    print("=" * 60)

    relation_types = {}
    for rel in COMPANY_RELATIONS:
        rel_type = rel.get('relation_type', '其他')
        if rel_type not in relation_types:
            relation_types[rel_type] = []
        relation_types[rel_type].append(rel)

    for rel_type, rels in relation_types.items():
        print(f"\n【{rel_type}】({len(rels)}条)")
        for rel in rels[:5]:  # 只显示前5条
            source = rel.get('source', '')
            target = rel.get('target', '')
            relation = rel.get('relation', '')
            print(f"  {source} --[{relation}]--> {target}")
        if len(rels) > 5:
            print(f"  ... 还有 {len(rels) - 5} 条")

    print("=" * 60 + "\n")


def view_statistics():
    """查看统计信息"""
    stats = get_statistics()

    print("\n" + "=" * 60)
    print("数据统计总览")
    print("=" * 60)
    print(f"企业总数: {stats['total_companies']}家")
    print(f"关系总数: {stats['total_relations']}条")
    print(f"生成文档: {stats['total_documents']}篇")

    print(f"\n【合作状态分布】")
    print(f"  已合作: {stats['cooperated_companies']}家")
    print(f"  竞品客户: {stats['competitor_companies']}家")
    print(f"  潜在客户: {stats['potential_companies']}家")

    print(f"\n【关系类型分布】")
    for rel_type, count in stats['relation_types'].items():
        print(f"  {rel_type}: {count}条")

    print(f"\n【行业分布】")
    for industry in stats['industries']:
        print(f"  - {industry}")

    print(f"\n【企业列表】")
    for i, company in enumerate(COMPANY_GRAPH, 1):
        status = "已合作" if company.get('past_projects') else \
                 "竞品客户" if company.get('competitor_product') else "潜在客户"
        print(f"  {i}. {company['name']:<15} | {company['industry']:<10} | {status}")

    print("=" * 60 + "\n")


def list_companies_by_status(status: str):
    """按状态列出企业"""
    companies = get_companies_by_status(status)

    status_map = {
        "cooperated": "已合作企业",
        "competitor": "竞品客户",
        "potential": "潜在客户",
        "all": "所有企业"
    }

    print("\n" + "=" * 60)
    print(status_map.get(status, status))
    print("=" * 60)

    for i, company in enumerate(companies, 1):
        print(f"{i}. {company['name']}")
        print(f"   行业: {company['industry']}")
        print(f"   规模: {company['employees']}人")
        if company.get('past_projects'):
            products = [p['product'] for p in company['past_projects']]
            print(f"   使用产品: {', '.join(products)}")
        print()

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="企业图谱数据查看工具")
    parser.add_argument("--company", "-c", help="查看指定企业详情")
    parser.add_argument("--relations", "-r", action="store_true", help="查看所有关系")
    parser.add_argument("--stats", "-s", action="store_true", help="查看统计信息")
    parser.add_argument("--status", choices=["cooperated", "competitor", "potential", "all"],
                       help="按合作状态列出企业")

    args = parser.parse_args()

    if args.company:
        view_company_detail(args.company)
    elif args.relations:
        view_all_relations()
    elif args.stats:
        view_statistics()
    elif args.status:
        list_companies_by_status(args.status)
    else:
        # 默认显示统计信息
        view_statistics()


if __name__ == "__main__":
    main()
