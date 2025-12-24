"""
实验3 v3 服务器版本：4种方法对比（使用服务器vLLM）

测试目标：
1. 对比四种方法：直接RAG、批量总结+RAG、渐进式总结v2+RAG、渐进式总结v3+增量RAG
2. 验证渐进式总结在800字长文本中的效果
3. 评估增量RAG的相关度过滤和去重效果

配置：
- LLM: 服务器本地 vLLM (Qwen/Qwen3-32B) - localhost:8000
- Embedding: 硅基流动 API (BAAI/bge-m3)
- Reranking: 硅基流动 API (BAAI/bge-reranker-v2-m3)
- 延迟模拟: 关闭（服务器不需要模拟）
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from rag_utils import EmbeddingService, VectorIndex
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS
from data.company_graph import convert_all_companies_to_documents
from experiments.incremental_summarizer_v2 import SimpleSummarizer
from experiments.incremental_summarizer_v3 import IncrementalRAGSummarizer

load_dotenv()

# 模型配置 - 服务器本地vLLM
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_MODEL = "Qwen/Qwen3-32B"

# 使用统一模型（服务器上的Qwen3-32B性能足够）
SUMMARY_MODEL = VLLM_MODEL
RESPONSE_MODEL = VLLM_MODEL

# Embedding配置（继续使用硅基流动）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBEDDING_TOKEN = os.getenv("EMBEDDING_TOKEN")


# ========== 加载测试用例 ==========

def load_test_cases() -> List[Dict]:
    """加载测试用例（v2版本，包含分段和干扰项）"""
    test_file = Path(__file__).parent / "long_audio_test_cases_v2.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data["test_cases"]


# ========== RAG工具函数 ==========

def search_with_query_text(
    query_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    top_k: int = 5
) -> List[Dict]:
    """使用文本查询进行向量检索"""
    query_vector = embedding_service.embed_single(query_text)
    results = vector_index.search(query_vector, top_k=top_k)
    return results


# ========== 流式生成 ==========

def generate_response_streaming(prompt: str, llm_client: OpenAI) -> tuple:
    """
    流式生成回复，返回(完整回复, TTFT, 生成时间, token数量)
    """
    start_time = time.time()
    ttft = None
    response_text = ""
    token_count = 0

    stream = llm_client.chat.completions.create(
        model=RESPONSE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=2000,
        temperature=0.7
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if ttft is None:
                ttft = time.time() - start_time
            response_text += content
            token_count += 1

    generation_time = time.time() - start_time

    if ttft is None:
        ttft = generation_time

    return response_text, ttft, generation_time, token_count


# ========== LLM评估函数 ==========

def llm_evaluate_all(
    method_name: str,
    full_text: str,
    summary: str,
    rag_results: List[Dict],
    final_response: str,
    ground_truth: Dict,
    llm_client: OpenAI
) -> Dict:
    """使用LLM评估所有维度"""
    eval_prompt = f"""你是一个专业的语音助手评估专家。请根据以下信息给出评分（0-100分）。

【原始输入】（用户的完整语音输入，约800字）：
{full_text}

【总结】（如果有）：
{summary if summary else "（无总结，直接使用原文）"}

【RAG检索结果】：
{len(rag_results)}个文档被检索到
{chr(10).join([f"- {doc.get('title', '无标题')}" for doc in rag_results[:3]])}

【最终回复】：
{final_response}

【Ground Truth】：
用户的关键需求：{ground_truth.get('key_points', [])}
应该避免的噪音：{ground_truth.get('noise_patterns', [])}

请从以下5个维度评分：

1. **信息保留率** (0-100分)
   - 最终回复是否覆盖了用户的所有关键需求点？
   - 重要的细节（预算、时间、技术要求等）是否都被保留？

2. **噪音过滤率** (0-100分)
   - 是否成功过滤了口语化表达、重复、无关闲聊？
   - 总结/回复是否简洁专业？

3. **RAG相关性** (0-100分)
   - 检索到的文档是否与用户需求高度相关？
   - 回复中的信息是否来自相关文档？

4. **回复质量** (0-100分)
   - 回复是否准确、专业、有针对性？
   - 是否回答了用户的核心问题？

5. **简洁度** (0-100分)
   - 回复是否简洁明了，没有冗余？
   - 信息密度是否合理？

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

    response = llm_client.chat.completions.create(
        model=RESPONSE_MODEL,
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.3,
        max_tokens=1000
    )

    result_text = response.choices[0].message.content.strip()

    # 去除markdown代码块标记
    if result_text.startswith("```"):
        lines = result_text.split('\n')
        result_text = '\n'.join(lines[1:-1])

    try:
        result = json.loads(result_text)
        return result
    except:
        print(f"[{method_name}] LLM评估返回格式错误: {result_text[:200]}")
        return {
            "info_retention_score": 0,
            "noise_filtering_score": 0,
            "rag_relevance_score": 0,
            "response_quality_score": 0,
            "conciseness_score": 0,
            "total_score": 0,
            "reasoning": "评估失败"
        }


# ========== 方法1：直接RAG（Baseline） ==========

def method1_baseline(
    full_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法1：直接使用完整文本进行RAG"""
    start_time = time.time()

    # 1. RAG检索
    rag_start = time.time()
    results = search_with_query_text(full_text, vector_index, embedding_service, top_k)
    rag_time = time.time() - rag_start

    # 2. 构建context
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    # 3. 生成回复
    prompt = f"""用户的语音输入：
{full_text}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    gen_start = time.time()
    final_response, ttft, gen_time, token_count = generate_response_streaming(prompt, llm_client)

    # 4. LLM评估
    eval_result = llm_evaluate_all(
        "baseline",
        full_text,
        "",
        results,
        final_response,
        ground_truth,
        llm_client
    )

    total_time = time.time() - start_time

    return {
        "method": "baseline",
        "summary": "",
        "rag_results": results,
        "final_response": final_response,
        "timing": {
            "rag_time": rag_time,
            "ttft": ttft,
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


# ========== 方法2：批量总结+RAG ==========

def method2_batch_summary(
    full_text: str,
    vector_index: VectorIndex,
    embedding_service: EmbeddingService,
    llm_client: OpenAI,
    ground_truth: Dict,
    top_k: int = 5
) -> Dict:
    """方法2：等待输入完成后，批量总结，然后RAG"""
    start_time = time.time()

    # 1. 批量总结
    summary_start = time.time()
    summary_prompt = f"""请总结以下语音输入的核心需求，过滤口语词和无关内容：

{full_text}

只返回简洁的总结（一段话）：
"""

    summary_response = llm_client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=500
    )
    summary = summary_response.choices[0].message.content.strip()
    summary_time = time.time() - summary_start

    # 2. RAG检索
    rag_start = time.time()
    results = search_with_query_text(summary, vector_index, embedding_service, top_k)
    rag_time = time.time() - rag_start

    # 3. 生成回复
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    prompt = f"""用户需求总结：
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

    total_time = time.time() - start_time

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
            "compression_ratio": len(summary) / len(full_text) if len(full_text) > 0 else 0,
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
    # 1. 渐进式总结（不模拟延迟）
    summarizer = SimpleSummarizer(llm_client, model_name=SUMMARY_MODEL)
    segment_results = []

    summary_start_time = time.time()
    for segment_data in segments:
        seg_result = summarizer.add_segment(segment_data["text"], simulate_delay=False)
        segment_results.append(seg_result)

    # 用户输入完成时刻
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
            "summary_time_total": time.time() - summary_start_time,
            "summary_processing_time": stats["total_processing_time"],
            "rag_time": rag_time,
            "ttft": ttft,
            "generation_time": gen_time,
            "total_time_after_input": total_time_after_input,
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
    # 1. 渐进式总结+RAG（不模拟延迟）
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
        seg_result = summarizer.add_segment(segment_data["text"], simulate_delay=False)
        segment_results.append(seg_result)

    # 用户输入完成时刻
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
            "summary_time_total": time.time() - summary_start_time,
            "summary_processing_time": stats["total_processing_time"],
            "rag_time": stats["total_rag_time"],
            "avg_rag_time": stats["avg_rag_time"],
            "ttft": ttft,
            "generation_time": gen_time,
            "total_time_after_input": total_time_after_input,
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
    """实验3 v3运行器（服务器版本）"""

    def __init__(self):
        print("\n初始化实验3 v3（服务器版本）...")
        print(f"LLM: {VLLM_MODEL} @ {VLLM_BASE_URL}")
        print(f"Embedding: {EMBEDDING_MODEL}")

        # 初始化LLM客户端（使用本地vLLM）
        self.llm_client = OpenAI(
            base_url=VLLM_BASE_URL,
            api_key="EMPTY"  # vLLM不需要API key
        )

        # 初始化Embedding服务（使用硅基流动）
        self.embedding_service = EmbeddingService(
            model_name=EMBEDDING_MODEL,
            api_key=EMBEDDING_TOKEN,
            base_url=EMBEDDING_URL
        )

        # 初始化向量索引
        self.vector_index = VectorIndex(self.embedding_service)

        # 加载知识库
        all_docs = FICTIONAL_DOCUMENTS + convert_all_companies_to_documents()
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
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"experiment3_v3_server_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 实验完成！结果已保存到: {output_file}")
        return results


def main():
    runner = Experiment3V3Runner()
    results = runner.run_all_tests()


if __name__ == "__main__":
    main()
