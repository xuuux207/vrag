"""
实验3 v3：长时间语音输入的渐进式总结测试（优化版 - 并行处理）

改进点：
1. 简化总结数据结构，只保留summary
2. 所有评估指标都交给LLM评分
3. 模拟真实延迟（边说边总结）
4. 使用快速模型做总结（qwen3-8b）
5. 使用14B模型生成回复（qwen3-14b）
6. 流式输出最终回复，记录首token延迟
7. 评估时间从用户输入完成开始计算
8. 并行运行三个方法（用户只说一次，三个agent同时处理）
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from rag_utils import EmbeddingService, VectorIndex
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS
from data.company_graph import convert_all_companies_to_documents
from experiments.incremental_summarizer_v2 import SimpleSummarizer
from experiments.incremental_summarizer_v3 import IncrementalRAGSummarizer

load_dotenv()

# 模型配置
API_KEY = os.getenv("QWEN_TOKEN")
BASE_URL = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SUMMARY_MODEL = "qwen3-8b"  # 8B模型用于总结（快速）
RESPONSE_MODEL = "qwen3-14b"  # 14B模型用于生成回复（高质量）
EVAL_MODEL = "qwen3-14b"  # 14B模型用于评估


# ========== 加载测试用例 ==========

def load_test_cases() -> List[Dict]:
    """加载测试用例"""
    test_file = Path(__file__).parent / "long_audio_test_cases_v2.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data["test_cases"]


# ========== LLM评估函数 ==========

def llm_evaluate_all(
    method_name: str,
    original_input: str,
    summary: str,
    rag_results: List[Dict],
    final_response: str,
    ground_truth: Dict,
    llm_client: OpenAI
) -> Dict:
    """
    使用LLM一次性评估所有指标
    """
    eval_prompt = f"""请评估以下语音助手的处理质量。

【原始用户输入】（{len(original_input)}字）：
{original_input[:200]}...

【总结结果】（{len(summary)}字）：
{summary}

【检索到的文档】：
{chr(10).join([f"{i+1}. {doc.get('title', '无标题')}" for i, doc in enumerate(rag_results[:3])])}

【最终回复】：
{final_response[:300]}...

【标准答案参考】：
- 关键信息点：{', '.join(ground_truth.get('key_points', [])[:5])}
- 关键实体：{', '.join(ground_truth.get('entities', []))}
- 原文噪音项数量：{ground_truth.get('total_noise_count', 0)}

请按以下标准评分（0-100分）：

1. 信息保留率（0-100分）：总结是否保留了所有关键信息
2. 噪音过滤率（0-100分）：是否有效过滤了口语词、寒暄等无用信息
3. RAG相关性（0-100分）：检索的文档是否与用户需求相关
4. 回复质量（0-100分）：回复是否准确、全面、专业
5. 简洁度（0-100分）：总结是否简洁，无冗余

请根据实际情况给出真实的分数，然后计算总分（5项平均值）。

返回JSON格式（不要markdown标记，不要示例数字，给出你的真实评分）：
{{
  "info_retention_score": <你的评分>,
  "noise_filtering_score": <你的评分>,
  "rag_relevance_score": <你的评分>,
  "response_quality_score": <你的评分>,
  "conciseness_score": <你的评分>,
  "total_score": <5项平均值>,
  "reasoning": "<详细的评分理由>"
}}
"""

    try:
        response_obj = llm_client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            stream=False,
            extra_body={"enable_thinking": False}
        )

        content = response_obj.choices[0].message.content.strip()

        # 清理markdown标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content.strip())
        return result

    except Exception as e:
        print(f"LLM评估失败: {e}")
        return {
            "info_retention_score": 50,
            "noise_filtering_score": 50,
            "rag_relevance_score": 50,
            "response_quality_score": 50,
            "conciseness_score": 50,
            "total_score": 50,
            "reasoning": f"评估失败: {str(e)}"
        }


# ========== 辅助函数 ==========

def search_with_query_text(
    query_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    top_k: int = 5
) -> List[Dict]:
    """使用文本query进行向量检索"""
    query_vector = embedding_service.embed_single(query_text)
    results = vector_index.search(query_vector, top_k=top_k)
    return [{"id": r["doc_id"], "title": r["title"], "content": r["content"]} for r in results]


def generate_response_streaming(
    prompt: str,
    llm_client: OpenAI
) -> Tuple[str, float, float, int]:
    """
    流式生成回复

    Returns:
        (完整回复, 首token延迟, 总生成时间, token数)
    """
    start_time = time.time()
    first_token_time = None
    full_response = ""
    token_count = 0

    try:
        stream = llm_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
            extra_body={"enable_thinking": False}
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time() - start_time

                content = chunk.choices[0].delta.content
                full_response += content
                token_count += 1

        total_time = time.time() - start_time

        return full_response, first_token_time or 0, total_time, token_count

    except Exception as e:
        print(f"流式生成失败: {e}")
        return f"生成失败: {str(e)}", 0, 0, 0


# ========== 方法1：Baseline（直接RAG） ==========

def method1_baseline(
    full_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法1：直接使用完整的800字文本进行RAG"""
    # 用户输入完成时刻
    input_complete_time = time.time()

    # 1. 向量检索
    rag_start = time.time()
    results = search_with_query_text(full_text, vector_index, embedding_service, top_k)
    rag_time = time.time() - rag_start

    # 2. 构建prompt
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    prompt = f"""用户查询：
{full_text}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    # 3. 流式生成回复
    gen_start = time.time()
    final_response, ttft, gen_time, token_count = generate_response_streaming(prompt, llm_client)

    # 4. LLM评估
    eval_result = llm_evaluate_all(
        "baseline",
        full_text,
        full_text,  # baseline没有summary，用原文
        results,
        final_response,
        ground_truth,
        llm_client
    )

    total_time = time.time() - input_complete_time

    return {
        "method": "baseline",
        "summary": full_text,  # 没有总结
        "rag_results": results,
        "final_response": final_response,
        "timing": {
            "rag_time": rag_time,
            "ttft": ttft,  # time to first token
            "generation_time": gen_time,
            "total_time": total_time
        },
        "metrics": {
            "query_length": len(full_text),
            "response_length": len(final_response),
            "token_count": token_count,
            "tokens_per_second": token_count / gen_time if gen_time > 0 else 0
        },
        "evaluation": eval_result
    }


# ========== 方法2：完整总结后RAG ==========

def method2_batch_summary(
    full_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法2：等待完整输入后一次性总结，然后RAG"""
    # 用户输入完成时刻
    input_complete_time = time.time()

    # 1. 一次性总结
    summary_start = time.time()
    summary_prompt = f"""用户进行了一段语音输入，请总结关键信息，过滤口语词和寒暄。

用户输入（{len(full_text)}字）：
{full_text}

只返回一行简洁的总结，不要JSON，不要markdown："""

    response = llm_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.1,
        stream=False,
        extra_body={"enable_thinking": False}
    )

    summary = response.choices[0].message.content.strip()
    summary_time = time.time() - summary_start

    # 2. 使用总结进行RAG
    rag_start = time.time()
    results = search_with_query_text(summary, vector_index, embedding_service, top_k)
    rag_time = time.time() - rag_start

    # 3. 生成回复
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    prompt = f"""用户原始输入：
{full_text[:200]}...

总结：
{summary}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    gen_start = time.time()
    final_response, ttft, gen_time, token_count = generate_response_streaming(prompt, llm_client)

    # 4. LLM评估
    eval_result = llm_evaluate_all(
        "batch_summary",
        full_text,
        summary,
        results,
        final_response,
        ground_truth,
        llm_client
    )

    total_time = time.time() - input_complete_time

    return {
        "method": "batch_summary",
        "summary": summary,
        "rag_results": results,
        "final_response": final_response,
        "timing": {
            "summary_time": summary_time,
            "rag_time": rag_time,
            "ttft": ttft,
            "generation_time": gen_time,
            "total_time": total_time
        },
        "metrics": {
            "query_length": len(summary),
            "compression_ratio": len(summary) / len(full_text),
            "response_length": len(final_response),
            "token_count": token_count,
            "tokens_per_second": token_count / gen_time if gen_time > 0 else 0
        },
        "evaluation": eval_result
    }


# ========== 方法3：渐进式总结+RAG ==========

def method3_incremental_summary(
    segments: List[Dict],
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法3：边输入边总结，最后进行RAG"""
    # 1. 渐进式总结（包含模拟延迟）
    summarizer = SimpleSummarizer(llm_client, model_name=SUMMARY_MODEL)
    segment_results = []

    summary_start_time = time.time()
    for segment_data in segments:
        seg_result = summarizer.add_segment(segment_data["text"], simulate_delay=True)
        segment_results.append(seg_result)

    # 用户输入完成时刻（包含说话时间）
    input_complete_time = time.time()

    summary = summarizer.get_final_summary()
    stats = summarizer.get_stats()

    # 2. 使用总结进行RAG
    rag_start = time.time()
    results = search_with_query_text(summary, vector_index, embedding_service, top_k)
    rag_time = time.time() - rag_start

    # 3. 生成回复
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    full_text = "".join([seg["text"] for seg in segments])

    prompt = f"""用户进行了{len(segments)}段语音输入。

总结：
{summary}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    gen_start = time.time()
    final_response, ttft, gen_time, token_count = generate_response_streaming(prompt, llm_client)

    # 4. LLM评估
    eval_result = llm_evaluate_all(
        "incremental_summary",
        full_text,
        summary,
        results,
        final_response,
        ground_truth,
        llm_client
    )

    # 注意：total_time从输入完成开始计算
    total_time_after_input = time.time() - input_complete_time

    return {
        "method": "incremental_summary",
        "summary": summary,
        "rag_results": results,
        "final_response": final_response,
        "segment_results": segment_results,
        "timing": {
            "summary_time_with_speech": time.time() - summary_start_time,  # 包含说话时间
            "summary_processing_time": stats["total_processing_time"],  # 纯处理时间
            "rag_time": rag_time,
            "ttft": ttft,
            "generation_time": gen_time,
            "total_time_after_input": total_time_after_input,  # 输入完成后的等待时间
        },
        "metrics": {
            "query_length": len(summary),
            "compression_ratio": stats["compression_ratio"],
            "response_length": len(final_response),
            "token_count": token_count,
            "tokens_per_second": token_count / gen_time if gen_time > 0 else 0,
            "avg_segment_processing": stats["avg_segment_time"]
        },
        "evaluation": eval_result
    }


# ========== 方法4：渐进式总结+增量RAG（v3） ==========

def method4_incremental_rag(
    segments: List[Dict],
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法4：边输入边总结+边检索，过滤低相关度文档（v3版本）"""
    # 1. 渐进式总结+RAG（包含模拟延迟）
    summarizer = IncrementalRAGSummarizer(
        llm_client,
        embedding_service,
        vector_index,
        model_name=SUMMARY_MODEL,
        relevance_threshold=0.6
    )
    segment_results = []

    summary_start_time = time.time()
    for segment_data in segments:
        seg_result = summarizer.add_segment(segment_data["text"], simulate_delay=True)
        segment_results.append(seg_result)

    # 用户输入完成时刻（包含说话时间）
    input_complete_time = time.time()

    summary = summarizer.get_final_summary()
    stats = summarizer.get_stats()

    # 获取累积的相关文档（已经去重和过滤）
    relevant_docs = summarizer.get_relevant_docs()

    # 2. 生成回复（使用累积的相关文档）
    rag_context = ""
    for i, doc in enumerate(relevant_docs[:top_k], 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    full_text = "".join([seg["text"] for seg in segments])

    prompt = f"""用户进行了{len(segments)}段语音输入。

总结：
{summary}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    gen_start = time.time()
    final_response, ttft, gen_time, token_count = generate_response_streaming(prompt, llm_client)

    # 3. LLM评估
    eval_result = llm_evaluate_all(
        "incremental_rag_v3",
        full_text,
        summary,
        relevant_docs[:top_k],
        final_response,
        ground_truth,
        llm_client
    )

    # 注意：total_time从输入完成开始计算
    total_time_after_input = time.time() - input_complete_time

    return {
        "method": "incremental_rag_v3",
        "summary": summary,
        "rag_results": relevant_docs[:top_k],
        "final_response": final_response,
        "segment_results": segment_results,
        "timing": {
            "summary_time_with_speech": time.time() - summary_start_time,  # 包含说话时间
            "summary_processing_time": stats["total_processing_time"],  # 纯处理时间
            "rag_time": stats["total_rag_time"],  # 累积RAG时间
            "avg_rag_time": stats["avg_rag_time"],  # 平均每段RAG时间
            "ttft": ttft,
            "generation_time": gen_time,
            "total_time_after_input": total_time_after_input,  # 输入完成后的等待时间
        },
        "metrics": {
            "query_length": len(summary),
            "compression_ratio": stats["compression_ratio"],
            "response_length": len(final_response),
            "token_count": token_count,
            "tokens_per_second": token_count / gen_time if gen_time > 0 else 0,
            "avg_segment_processing": stats["avg_segment_time"],
            "total_retrieved_docs": stats["total_retrieved_docs"],
            "total_relevant_docs": stats["total_relevant_docs"]
        },
        "evaluation": eval_result
    }


# ========== 主实验类 ==========

class Experiment3V3Runner:
    """实验3 v3运行器"""

    def __init__(self):
        print("初始化服务...")

        # LLM客户端
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        # Embedding服务
        self.embedding_service = EmbeddingService()

        # 初始化知识库
        print("构建知识库...")
        self.init_knowledge_base()

        print("✓ 初始化完成\n")

    def init_knowledge_base(self):
        """初始化知识库"""
        company_docs = convert_all_companies_to_documents()
        all_docs = FICTIONAL_DOCUMENTS + company_docs

        self.vector_index = VectorIndex(self.embedding_service)
        self.vector_index.add_documents(all_docs)

        print(f"知识库文档数: {len(all_docs)}")

    def run_single_test(self, test_case: Dict) -> Dict:
        """运行单个测试用例 - 并行执行四个方法"""
        print(f"\n{'='*70}")
        print(f"测试用例: {test_case['id']}")
        print(f"类别: {test_case['category']}")
        print(f"总长度: {test_case['total_length']} 字")
        print(f"分段数: {len(test_case['segments'])}")
        print(f"{'='*70}\n")

        full_text = "".join([seg["text"] for seg in test_case["segments"]])

        result = {
            "test_case_id": test_case["id"],
            "category": test_case["category"],
            "total_length": test_case["total_length"],
            "segment_count": len(test_case["segments"]),
            "ground_truth": test_case["ground_truth"]
        }

        print("🚀 并行运行四个方法...")
        start_parallel = time.time()

        # 使用线程池并行运行四个方法
        with ThreadPoolExecutor(max_workers=4) as executor:
            # 提交四个任务
            future_m1 = executor.submit(
                method1_baseline,
                full_text,
                self.vector_index,
                self.embedding_service,
                self.llm_client,
                test_case["ground_truth"]
            )

            future_m2 = executor.submit(
                method2_batch_summary,
                full_text,
                self.vector_index,
                self.embedding_service,
                self.llm_client,
                test_case["ground_truth"]
            )

            future_m3 = executor.submit(
                method3_incremental_summary,
                test_case["segments"],
                self.vector_index,
                self.embedding_service,
                self.llm_client,
                test_case["ground_truth"]
            )

            future_m4 = executor.submit(
                method4_incremental_rag,
                test_case["segments"],
                self.vector_index,
                self.embedding_service,
                self.llm_client,
                test_case["ground_truth"]
            )

            # 收集结果
            futures = {
                "method1": future_m1,
                "method2": future_m2,
                "method3": future_m3,
                "method4": future_m4
            }

            for method_name, future in futures.items():
                try:
                    result_data = future.result()
                    if method_name == "method1":
                        result["method1_baseline"] = result_data
                        print(f"  ✓ 方法1完成（{result_data['timing']['total_time']:.2f}秒）")
                    elif method_name == "method2":
                        result["method2_batch"] = result_data
                        print(f"  ✓ 方法2完成（{result_data['timing']['total_time']:.2f}秒）")
                    elif method_name == "method3":
                        result["method3_incremental"] = result_data
                        print(f"  ✓ 方法3完成（输入后: {result_data['timing']['total_time_after_input']:.2f}秒）")
                    else:
                        result["method4_incremental_rag"] = result_data
                        print(f"  ✓ 方法4完成（输入后: {result_data['timing']['total_time_after_input']:.2f}秒, 检索{result_data['metrics']['total_relevant_docs']}个文档）")
                except Exception as e:
                    print(f"  ✗ {method_name}失败: {e}")
                    import traceback
                    traceback.print_exc()

        parallel_time = time.time() - start_parallel
        print(f"\n并行总耗时: {parallel_time:.2f}秒")

        return result

    def run_all_tests(self):
        """运行所有测试"""
        test_cases = load_test_cases()
        results = []

        print(f"\n开始运行 {len(test_cases)} 个测试用例...")
        print(f"配置：总结模型={SUMMARY_MODEL}, 回复模型={RESPONSE_MODEL}")
        print(f"方法对比：")
        print(f"  1. Baseline - 直接RAG (800字原文)")
        print(f"  2. Batch Summary - 等输入完成后总结+RAG")
        print(f"  3. Incremental v2 - 边输入边总结，最后RAG")
        print(f"  4. Incremental v3 (新) - 边输入边总结+增量RAG，相关度过滤\n")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}]")
            try:
                result = self.run_single_test(test_case)
                results.append(result)
            except Exception as e:
                print(f"测试失败: {e}")
                import traceback
                traceback.print_exc()

        # 保存结果
        self.save_results(results)

        return results

    def save_results(self, results: List[Dict]):
        """保存结果到JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"experiment3_v3_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 结果已保存: {output_file}")


def main():
    """主函数"""
    runner = Experiment3V3Runner()
    results = runner.run_all_tests()

    print("\n" + "="*70)
    print("实验3 v3完成！")
    print("="*70)


if __name__ == "__main__":
    main()
