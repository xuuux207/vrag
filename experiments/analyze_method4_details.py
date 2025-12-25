"""
深入分析方法4的结果和LLM评分理由
"""
import json
from pathlib import Path

# 加载最新结果
output_dir = Path(__file__).parent.parent / "outputs"
result_files = list(output_dir.glob("experiment3_dual_model_results_*.json"))
latest_file = max(result_files, key=lambda p: p.stat().st_mtime)

print(f"📂 加载文件: {latest_file.name}\n")

with open(latest_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

# 分析每个测试用例
for idx, test_case in enumerate(results, 1):
    print(f"\n{'='*80}")
    print(f"测试用例 {idx}: {test_case['test_case_id']}")
    print(f"类别: {test_case['category']}")
    print('='*80)

    # 对比方法1和方法4
    m1 = test_case.get('method1_baseline', {})
    m4 = test_case.get('method4_incremental_rag', {})

    if not m4:
        print("方法4数据缺失")
        continue

    print("\n【方法1 (Baseline) vs 方法4 (Incremental RAG)】")
    print(f"\n方法1评分: {m1.get('evaluation', {}).get('total_score', 0):.1f}/100")
    print(f"方法4评分: {m4.get('evaluation', {}).get('total_score', 0):.1f}/100")

    # 详细评分对比
    print("\n评分维度对比:")
    eval_keys = ['info_retention_score', 'noise_filtering_score', 'rag_relevance_score',
                 'response_quality_score', 'conciseness_score']
    eval_names = ['信息保留', '噪音过滤', 'RAG相关性', '回复质量', '简洁度']

    for key, name in zip(eval_keys, eval_names):
        m1_score = m1.get('evaluation', {}).get(key, 0)
        m4_score = m4.get('evaluation', {}).get(key, 0)
        diff = m4_score - m1_score
        print(f"  {name:8s}: M1={m1_score:5.1f}, M4={m4_score:5.1f}, 差值={diff:+6.1f}")

    # 检索文档对比
    print(f"\n检索文档数: M1={len(m1.get('rag_results', []))}个, M4={len(m4.get('rag_results', []))}个")

    print("\n方法1检索的文档:")
    for i, doc in enumerate(m1.get('rag_results', [])[:3], 1):
        print(f"  {i}. {doc.get('title', '无标题')}")

    print("\n方法4检索的文档:")
    for i, doc in enumerate(m4.get('rag_results', [])[:3], 1):
        print(f"  {i}. {doc.get('title', '无标题')}")

    # LLM评分理由
    print(f"\n【方法4的LLM评分理由】:")
    reasoning = m4.get('evaluation', {}).get('reasoning', '无理由')
    # 只显示前500字
    print(reasoning[:500] if len(reasoning) > 500 else reasoning)

    # 总结长度对比
    m4_summary = m4.get('summary', '')
    print(f"\n【方法4总结】(长度: {len(m4_summary)}字):")
    print(m4_summary[:200] + "..." if len(m4_summary) > 200 else m4_summary)

    # 回复对比
    m1_response = m1.get('final_response', '')
    m4_response = m4.get('final_response', '')
    print(f"\n【回复长度】: M1={len(m1_response)}字, M4={len(m4_response)}字")

    print(f"\n【方法4回复】(前300字):")
    print(m4_response[:300] if len(m4_response) > 300 else m4_response)

    if idx >= 2:  # 只看前2个case
        print("\n(后续案例省略...)")
        break

print("\n" + "="*80)
print("分析完成")
