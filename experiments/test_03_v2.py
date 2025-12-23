"""
实验3 v2：长时间语音输入的渐进式总结测试

测试目标：
1. 对比三种方法：直接RAG、完整总结+RAG、渐进式总结+RAG
2. 验证渐进式总结在800字长文本中的效果
3. 评估噪音过滤能力和信息保留率

配置：
- LLM: 通义千问 API (qwen-plus)
- Embedding: 云端 API (BAAI/bge-m3)
- Reranking: 云端 API (BAAI/bge-reranker-v2-m3)
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from rag_utils import (
    EmbeddingService,
    VectorIndex,
    RerankService,
    BM25Index,
    get_jieba
)
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS
from data.company_graph import convert_all_companies_to_documents
from experiments.incremental_summarizer import IncrementalSummarizer

load_dotenv()

# 模型配置 - 使用通义千问API
API_KEY = os.getenv("QWEN_TOKEN")
BASE_URL = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen-plus")


# ========== 加载测试用例 ==========

def load_test_cases() -> List[Dict]:
    """加载测试用例（v2版本，包含分段和干扰项）"""
    test_file = Path(__file__).parent / "long_audio_test_cases_v2.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data["test_cases"]


# ========== 评估函数 ==========

def calculate_info_retention(extracted_points: List[str], ground_truth_points: List[str]) -> float:
    """
    计算信息保留率
    """
    if not ground_truth_points:
        return 1.0

    matched = 0
    for gt_point in ground_truth_points:
        for ext_point in extracted_points:
            # 简单的包含匹配
            if gt_point in ext_point or ext_point in gt_point:
                matched += 1
                break

    return matched / len(ground_truth_points)


def calculate_noise_filtering_rate(filtered_count: int, total_noise_count: int) -> float:
    """计算噪音过滤率"""
    if total_noise_count == 0:
        return 1.0
    return min(filtered_count / total_noise_count, 1.0)


def calculate_rag_recall(rag_results: List[Dict], expected_docs: List[str]) -> float:
    """计算RAG检索召回率"""
    if not expected_docs:
        return 1.0

    retrieved_titles = [doc.get("title", "") for doc in rag_results]

    matched = 0
    for expected in expected_docs:
        for title in retrieved_titles:
            if expected in title or title in expected:
                matched += 1
                break

    return matched / len(expected_docs)


def evaluate_response_quality(
    response: str,
    ground_truth: Dict,
    llm_client: OpenAI
) -> Tuple[float, str]:
    """使用LLM评估回复质量"""
    eval_prompt = f"""请评估以下AI助手的回复质量。

用户的关键需求：
{', '.join(ground_truth.get('key_points', []))}

AI回复：
{response}

评分标准（1-10分）：
- 是否回答了所有关键需求（40%）
- 信息是否准确、具体（30%）
- 是否引用了相关案例或数据（20%）
- 语言是否专业、友好（10%）

请返回JSON格式（不要markdown代码块）：
{{
  "score": 8.5,
  "reasoning": "评分理由"
}}
"""

    try:
        response_obj = llm_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )

        content = response_obj.choices[0].message.content.strip()

        # 移除markdown标记
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content.strip())
        return result["score"], result["reasoning"]

    except Exception as e:
        print(f"LLM评分失败: {e}")
        return 5.0, "评分失败"


# ========== 方法1：Baseline（直接RAG） ==========

def method1_baseline(
    full_text: str,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    rerank_service: RerankService,
    llm_client: OpenAI,
    top_k: int = 5
) -> Dict:
    """
    方法1：直接使用完整的800字文本进行RAG
    """
    start_time = time.time()

    # 1. 检索
    vector_results = vector_index.search(full_text, top_k=top_k * 2)
    bm25_results = bm25_index.search(full_text, top_k=top_k * 2)

    # 合并去重
    all_results = {}
    for doc in vector_results:
        all_results[doc["id"]] = doc
    for doc in bm25_results:
        if doc["id"] not in all_results:
            all_results[doc["id"]] = doc

    results = list(all_results.values())

    # Rerank
    if len(results) > top_k:
        results = rerank_service.rerank(full_text, results, top_k=top_k)
    else:
        results = results[:top_k]

    rag_time = time.time() - start_time

    # 2. 生成回复
    gen_start = time.time()
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

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    final_response = response.choices[0].message.content.strip()
    gen_time = time.time() - gen_start

    return {
        "method": "baseline",
        "rag_results": results,
        "final_response": final_response,
        "timing_rag": rag_time,
        "timing_generate": gen_time,
        "timing_total": time.time() - start_time,
        "query_length": len(full_text)
    }


# ========== 方法2：完整总结后RAG ==========

def method2_batch_summary(
    full_text: str,
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    rerank_service: RerankService,
    llm_client: OpenAI,
    top_k: int = 5
) -> Dict:
    """
    方法2：等待完整输入后一次性总结，然后RAG
    """
    start_time = time.time()

    # 1. 一次性总结
    summary_start = time.time()
    summary_prompt = f"""用户进行了一段长时间的语音输入，请总结关键信息。

用户输入（{len(full_text)}字）：
{full_text}

请返回JSON（不要markdown标记）：
{{
  "concise_summary": "简洁的总结（100-200字）",
  "key_points": ["关键点1", "关键点2"],
  "filtered_noise_count": 估计过滤的干扰项数量
}}
"""

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.1
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    summary_result = json.loads(content.strip())
    summary_time = time.time() - summary_start

    # 2. 使用总结进行RAG
    concise_query = summary_result["concise_summary"]

    rag_start = time.time()
    vector_results = vector_index.search(concise_query, top_k=top_k * 2)
    bm25_results = bm25_index.search(concise_query, top_k=top_k * 2)

    all_results = {}
    for doc in vector_results:
        all_results[doc["id"]] = doc
    for doc in bm25_results:
        if doc["id"] not in all_results:
            all_results[doc["id"]] = doc

    results = list(all_results.values())

    if len(results) > top_k:
        results = rerank_service.rerank(concise_query, results, top_k=top_k)
    else:
        results = results[:top_k]

    rag_time = time.time() - rag_start

    # 3. 生成回复
    gen_start = time.time()
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    prompt = f"""用户原始输入：
{full_text[:200]}...

提取的关键信息：
{concise_query}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    final_response = response.choices[0].message.content.strip()
    gen_time = time.time() - gen_start

    return {
        "method": "batch_summary",
        "summary": summary_result,
        "concise_query": concise_query,
        "rag_results": results,
        "final_response": final_response,
        "key_points": summary_result.get("key_points", []),
        "filtered_noise_count": summary_result.get("filtered_noise_count", 0),
        "timing_summary": summary_time,
        "timing_rag": rag_time,
        "timing_generate": gen_time,
        "timing_total": time.time() - start_time,
        "query_length": len(concise_query)
    }


# ========== 方法3：渐进式总结+RAG ==========

def method3_incremental_summary(
    segments: List[Dict],
    vector_index: VectorIndex,
    bm25_index: BM25Index,
    rerank_service: RerankService,
    llm_client: OpenAI,
    top_k: int = 5
) -> Dict:
    """
    方法3：边输入边总结，最后进行RAG
    """
    start_time = time.time()

    # 1. 渐进式总结
    summarizer = IncrementalSummarizer(llm_client, model_name=MODEL)
    segment_results = []

    for segment_data in segments:
        seg_result = summarizer.add_segment(segment_data["text"])
        segment_results.append(seg_result)

    summary_time = time.time() - start_time

    # 2. 最终结构化提取
    final_result = summarizer.finalize()

    # 3. 使用结构化query进行RAG
    concise_query = final_result["structured_query"]["concise_query"]

    rag_start = time.time()
    vector_results = vector_index.search(concise_query, top_k=top_k * 2)
    bm25_results = bm25_index.search(concise_query, top_k=top_k * 2)

    all_results = {}
    for doc in vector_results:
        all_results[doc["id"]] = doc
    for doc in bm25_results:
        if doc["id"] not in all_results:
            all_results[doc["id"]] = doc

    results = list(all_results.values())

    if len(results) > top_k:
        results = rerank_service.rerank(concise_query, results, top_k=top_k)
    else:
        results = results[:top_k]

    rag_time = time.time() - rag_start

    # 4. 生成回复
    gen_start = time.time()
    rag_context = ""
    for i, doc in enumerate(results, 1):
        rag_context += f"\n[文档{i}] {doc.get('title', '无标题')}\n"
        rag_context += f"{doc.get('content', '无内容')[:500]}\n"

    prompt = f"""用户进行了{len(segments)}段语音输入。

最终总结：
{final_result['final_summary']}

关键信息：
{', '.join(final_result['structured_query']['key_points'])}

相关信息：
{rag_context}

请给出专业、准确的回复：
"""

    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    final_response = response.choices[0].message.content.strip()
    gen_time = time.time() - gen_start

    return {
        "method": "incremental_summary",
        "segment_results": segment_results,
        "final_result": final_result,
        "concise_query": concise_query,
        "rag_results": results,
        "final_response": final_response,
        "key_points": final_result["structured_query"]["key_points"],
        "filtered_noise_count": final_result["total_filtered_noise"],
        "timing_summary": summary_time,
        "timing_rag": rag_time,
        "timing_generate": gen_time,
        "timing_total": time.time() - start_time,
        "query_length": len(concise_query),
        "compression_ratio": final_result["compression_ratio"]
    }


# ========== 主实验类 ==========

class Experiment3V2Runner:
    """实验3 v2运行器"""

    def __init__(self):
        print("初始化服务...")

        # LLM - 使用通义千问API
        self.llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

        # Embedding & Reranking
        self.embedding_service = EmbeddingService()
        self.rerank_service = RerankService()

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

        jieba = get_jieba()
        self.bm25_index = BM25Index(jieba=jieba)
        self.bm25_index.add_documents(all_docs)

        print(f"知识库文档数: {len(all_docs)}")

    def run_single_test(self, test_case: Dict) -> Dict:
        """运行单个测试用例"""
        print(f"\n{'='*70}")
        print(f"测试用例: {test_case['id']}")
        print(f"类别: {test_case['category']}")
        print(f"总长度: {test_case['total_length']} 字")
        print(f"分段数: {len(test_case['segments'])}")
        print(f"{'='*70}\n")

        # 拼接完整文本
        full_text = "".join([seg["text"] for seg in test_case["segments"]])

        result = {
            "test_case_id": test_case["id"],
            "category": test_case["category"],
            "total_length": test_case["total_length"],
            "segment_count": len(test_case["segments"]),
            "ground_truth": test_case["ground_truth"]
        }

        # 方法1: Baseline
        print("方法1: Baseline（直接RAG）")
        method1_result = method1_baseline(
            full_text,
            self.vector_index,
            self.bm25_index,
            self.rerank_service,
            self.llm_client
        )
        result["method1_baseline"] = method1_result

        # 方法2: 完整总结
        print("方法2: 完整总结后RAG")
        method2_result = method2_batch_summary(
            full_text,
            self.vector_index,
            self.bm25_index,
            self.rerank_service,
            self.llm_client
        )
        result["method2_batch"] = method2_result

        # 方法3: 渐进式总结
        print("方法3: 渐进式总结+RAG")
        method3_result = method3_incremental_summary(
            test_case["segments"],
            self.vector_index,
            self.bm25_index,
            self.rerank_service,
            self.llm_client
        )
        result["method3_incremental"] = method3_result

        # 评估
        print("\n评估中...")
        result["evaluation"] = self.evaluate(test_case, method1_result, method2_result, method3_result)

        return result

    def evaluate(self, test_case: Dict, m1: Dict, m2: Dict, m3: Dict) -> Dict:
        """评估三种方法"""
        gt = test_case["ground_truth"]
        evaluation = {}

        # 信息保留率（仅方法2和3）
        if "key_points" in m2:
            evaluation["method2_info_retention"] = calculate_info_retention(
                m2["key_points"], gt["key_points"]
            )
        if "key_points" in m3:
            evaluation["method3_info_retention"] = calculate_info_retention(
                m3["key_points"], gt["key_points"]
            )

        # 噪音过滤率
        if "total_noise_count" in gt:
            if "filtered_noise_count" in m2:
                evaluation["method2_noise_filtering"] = calculate_noise_filtering_rate(
                    m2["filtered_noise_count"], gt["total_noise_count"]
                )
            if "filtered_noise_count" in m3:
                evaluation["method3_noise_filtering"] = calculate_noise_filtering_rate(
                    m3["filtered_noise_count"], gt["total_noise_count"]
                )

        # RAG召回率
        if "expected_docs" in gt:
            evaluation["method1_rag_recall"] = calculate_rag_recall(
                m1.get("rag_results", []), gt["expected_docs"]
            )
            evaluation["method2_rag_recall"] = calculate_rag_recall(
                m2.get("rag_results", []), gt["expected_docs"]
            )
            evaluation["method3_rag_recall"] = calculate_rag_recall(
                m3.get("rag_results", []), gt["expected_docs"]
            )

        # 回复质量评分
        for method_name, method_result in [("method1", m1), ("method2", m2), ("method3", m3)]:
            score, reason = evaluate_response_quality(
                method_result.get("final_response", ""),
                gt,
                self.llm_client
            )
            evaluation[f"{method_name}_response_score"] = score
            evaluation[f"{method_name}_score_reason"] = reason

        # 延迟对比
        evaluation["method1_latency"] = m1.get("timing_total", 0)
        evaluation["method2_latency"] = m2.get("timing_total", 0)
        evaluation["method3_latency"] = m3.get("timing_total", 0)

        # Query长度对比
        evaluation["method1_query_length"] = m1.get("query_length", 0)
        evaluation["method2_query_length"] = m2.get("query_length", 0)
        evaluation["method3_query_length"] = m3.get("query_length", 0)

        return evaluation

    def run_all_tests(self):
        """运行所有测试"""
        test_cases = load_test_cases()
        results = []

        print(f"\n开始运行 {len(test_cases)} 个测试用例...\n")

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
        self.generate_report(results)

        return results

    def save_results(self, results: List[Dict]):
        """保存结果到JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"experiment3_v2_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 结果已保存: {output_file}")

    def generate_report(self, results: List[Dict]):
        """生成分析报告"""
        # TODO: 实现详细报告生成
        pass


def main():
    """主函数"""
    runner = Experiment3V2Runner()
    results = runner.run_all_tests()

    print("\n" + "="*70)
    print("实验3 v2完成！")
    print("="*70)


if __name__ == "__main__":
    main()
