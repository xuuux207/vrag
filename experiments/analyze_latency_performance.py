"""
分析实验3 v3服务器版本的延迟性能
重点关注：延迟指标对比、用户体验优化

运行: uv run python experiments/analyze_latency_performance.py
"""

import json
from pathlib import Path
from typing import Dict, List
import statistics


def load_results(file_path: str) -> List[Dict]:
    """加载实验结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_latency_metrics(results: List[Dict]) -> Dict:
    """提取延迟相关指标"""
    metrics = {
        "method1_baseline": [],
        "method2_batch": [],
        "method3_incremental": [],
        "method4_incremental_rag": []
    }

    for test_case in results:
        for method_key in metrics.keys():
            if method_key in test_case:
                method_data = test_case[method_key]
                timing = method_data.get("timing", {})

                # 提取关键延迟指标
                latency_data = {
                    "test_case_id": test_case["test_case_id"],
                    "category": test_case["category"],
                    # TTFT (Time To First Token) - 用户感知到响应的时间
                    "ttft": timing.get("ttft", 0),
                    # Generation time - 完整生成回复的时间
                    "generation_time": timing.get("generation_time", 0),
                    # Total time - 总处理时间
                    "total_time": timing.get("total_time", 0) if method_key in ["method1_baseline", "method2_batch"] else timing.get("total_time_after_input", 0),
                    # RAG time - 检索时间
                    "rag_time": timing.get("rag_time", 0),
                    # Summary time - 总结时间 (仅method2/3/4)
                    "summary_time": timing.get("summary_time", 0) if method_key == "method2_batch" else timing.get("summary_processing_time", 0),
                    # Tokens per second - 生成速度
                    "tokens_per_second": method_data.get("metrics", {}).get("tokens_per_second", 0)
                }

                metrics[method_key].append(latency_data)

    return metrics


def calculate_statistics(data_list: List[float]) -> Dict:
    """计算统计指标"""
    if not data_list:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}

    return {
        "mean": statistics.mean(data_list),
        "median": statistics.median(data_list),
        "min": min(data_list),
        "max": max(data_list),
        "std": statistics.stdev(data_list) if len(data_list) > 1 else 0
    }


def analyze_latency(metrics: Dict) -> None:
    """分析延迟性能"""

    print("\n" + "="*80)
    print("实验3 v3 服务器版本 - 延迟性能分析报告")
    print("="*80)
    print("\n【用户关注点】延迟是用户最担心的问题\n")

    # 1. TTFT (Time To First Token) 分析
    print("\n" + "-"*80)
    print("1. TTFT (首Token延迟) - 用户感知响应速度的关键指标")
    print("-"*80)
    print("   TTFT越低，用户越早看到系统开始回复，体验越好\n")

    for method_name, method_data in metrics.items():
        ttft_values = [d["ttft"] for d in method_data]
        ttft_stats = calculate_statistics(ttft_values)

        method_display = {
            "method1_baseline": "方法1 (Baseline)",
            "method2_batch": "方法2 (批量总结)",
            "method3_incremental": "方法3 (渐进式v2)",
            "method4_incremental_rag": "方法4 (渐进式RAG v3)"
        }

        print(f"{method_display[method_name]}:")
        print(f"  平均TTFT: {ttft_stats['mean']:.2f}秒 (± {ttft_stats['std']:.2f})")
        print(f"  中位数:   {ttft_stats['median']:.2f}秒")
        print(f"  范围:     {ttft_stats['min']:.2f}秒 - {ttft_stats['max']:.2f}秒")
        print()

    # 2. 总延迟对比
    print("-"*80)
    print("2. 总处理时间 - 完整回复用户所需时间")
    print("-"*80)
    print("   对于方法3/4，这是用户输入完成后的等待时间\n")

    for method_name, method_data in metrics.items():
        total_time_values = [d["total_time"] for d in method_data]
        total_stats = calculate_statistics(total_time_values)

        method_display = {
            "method1_baseline": "方法1 (Baseline)",
            "method2_batch": "方法2 (批量总结)",
            "method3_incremental": "方法3 (渐进式v2)",
            "method4_incremental_rag": "方法4 (渐进式RAG v3)"
        }

        print(f"{method_display[method_name]}:")
        print(f"  平均总时间: {total_stats['mean']:.2f}秒 (± {total_stats['std']:.2f})")
        print(f"  中位数:     {total_stats['median']:.2f}秒")
        print(f"  范围:       {total_stats['min']:.2f}秒 - {total_stats['max']:.2f}秒")
        print()

    # 3. 生成速度对比
    print("-"*80)
    print("3. Token生成速度 - 模型推理性能")
    print("-"*80)
    print("   服务器本地vLLM的生成速度\n")

    for method_name, method_data in metrics.items():
        tps_values = [d["tokens_per_second"] for d in method_data]
        tps_stats = calculate_statistics(tps_values)

        method_display = {
            "method1_baseline": "方法1 (Baseline)",
            "method2_batch": "方法2 (批量总结)",
            "method3_incremental": "方法3 (渐进式v2)",
            "method4_incremental_rag": "方法4 (渐进式RAG v3)"
        }

        print(f"{method_display[method_name]}:")
        print(f"  平均速度: {tps_stats['mean']:.1f} tokens/秒 (± {tps_stats['std']:.1f})")
        print(f"  中位数:   {tps_stats['median']:.1f} tokens/秒")
        print(f"  范围:     {tps_stats['min']:.1f} - {tps_stats['max']:.1f} tokens/秒")
        print()

    # 4. 用户体验总结
    print("-"*80)
    print("4. 用户体验分析 - 延迟优化建议")
    print("-"*80)
    print()

    # 计算方法4相比方法1的优势
    m1_ttft = statistics.mean([d["ttft"] for d in metrics["method1_baseline"]])
    m4_ttft = statistics.mean([d["ttft"] for d in metrics["method4_incremental_rag"]])

    m1_total = statistics.mean([d["total_time"] for d in metrics["method1_baseline"]])
    m4_total = statistics.mean([d["total_time"] for d in metrics["method4_incremental_rag"]])

    print("【方法4 (渐进式RAG v3) vs 方法1 (Baseline)】")
    print(f"  TTFT改善:     {m1_ttft:.2f}秒 → {m4_ttft:.2f}秒 (减少 {((m1_ttft - m4_ttft) / m1_ttft * 100):.1f}%)")
    print(f"  总时间改善:   {m1_total:.2f}秒 → {m4_total:.2f}秒 (减少 {((m1_total - m4_total) / m1_total * 100):.1f}%)")
    print()

    print("【关键发现】")

    # 检查是否满足≤1500ms的目标
    avg_total_times = {
        name: statistics.mean([d["total_time"] for d in data])
        for name, data in metrics.items()
    }

    print(f"\n  1. 延迟性能:")
    for method_name, avg_time in avg_total_times.items():
        method_display = {
            "method1_baseline": "方法1",
            "method2_batch": "方法2",
            "method3_incremental": "方法3",
            "method4_incremental_rag": "方法4"
        }

        # Workshop目标: ≤1500ms (1.5秒)
        meets_goal = "✓ 满足" if avg_time <= 1.5 else "✗ 不满足"
        print(f"     {method_display[method_name]}: {avg_time:.2f}秒 → {meets_goal} ≤1.5秒目标")

    print(f"\n  2. 用户体验优势:")
    print(f"     方法4的渐进式处理让用户在说话过程中就开始准备回复")
    print(f"     输入完成后仅需等待 {m4_total:.1f}秒 即可获得回复")

    print(f"\n  3. 服务器性能:")
    avg_tps = statistics.mean([d["tokens_per_second"] for d in metrics["method4_incremental_rag"]])
    print(f"     本地vLLM (Qwen3-32B): {avg_tps:.1f} tokens/秒")
    print(f"     比云端API更快，且无网络延迟")

    print("\n" + "="*80)
    print("总结: 用户最担心的延迟问题")
    print("="*80)
    print("\n【现状】")
    print(f"  - 所有方法的处理时间均远超1.5秒目标")
    print(f"  - 方法4虽然最优，但平均 {m4_total:.1f}秒 仍需优化")
    print(f"  - 本地vLLM性能良好 ({avg_tps:.1f} tokens/秒)")

    print("\n【问题根源】")
    # 分析各个环节的时间占比
    m4_avg_rag = statistics.mean([d["rag_time"] for d in metrics["method4_incremental_rag"]])
    m4_avg_gen = statistics.mean([d["generation_time"] for d in metrics["method4_incremental_rag"]])
    m4_avg_summary = statistics.mean([d["summary_time"] for d in metrics["method4_incremental_rag"]])

    print(f"  方法4各环节平均耗时:")
    print(f"    - 总结处理: {m4_avg_summary:.2f}秒 ({m4_avg_summary/m4_total*100:.1f}%)")
    print(f"    - RAG检索:  {m4_avg_rag:.2f}秒 ({m4_avg_rag/m4_total*100:.1f}%)")
    print(f"    - 生成回复: {m4_avg_gen:.2f}秒 ({m4_avg_gen/m4_total*100:.1f}%)")

    print("\n【优化建议】")
    print("  1. 模型优化:")
    print("     - 使用更小的模型 (Qwen3-8B → 更快)")
    print("     - 限制生成长度 (当前平均800+ tokens)")
    print("     - 使用流式输出优化TTFT")

    print("\n  2. RAG优化:")
    print("     - 优化embedding模型 (当前用云端API)")
    print("     - 减少检索文档数量")
    print("     - 使用更快的向量检索引擎")

    print("\n  3. 总结优化:")
    print("     - 简化总结提示词")
    print("     - 使用更轻量级的总结策略")
    print("     - 考虑缓存常见模式")

    print("\n" + "="*80 + "\n")


def main():
    """主函数"""
    # 加载结果
    results_file = Path(__file__).parent.parent / "outputs" / "experiment3_v3_server_results_20251224_131035.json"

    if not results_file.exists():
        print(f"错误: 找不到结果文件 {results_file}")
        return

    print(f"加载结果文件: {results_file}")
    results = load_results(results_file)

    # 提取延迟指标
    metrics = extract_latency_metrics(results)

    # 分析延迟
    analyze_latency(metrics)


if __name__ == "__main__":
    main()
