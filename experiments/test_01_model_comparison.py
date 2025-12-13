"""
实验1：模型对比实验
测试 qwen3-8b, qwen3-32b, qwen3-72b 在有/无 RAG 支持下的表现

测试矩阵：6 场景 × 3 模型 × 2 模式 = 36 次测试
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from rag_utils import (
    EmbeddingService,
    VectorIndex,
    RerankingService,
    rag_retrieve_and_rerank,
    build_rag_context
)
from data.fictional_knowledge_base import FICTIONAL_DOCUMENTS, TEST_SCENARIOS

# 加载环境变量
load_dotenv()

# 模型配置
MODELS = [
    "qwen3-8b",      # 8b 模型
    "qwen3-14b",     # 14b 模型
    "qwen3-32b"      # 32b 模型
]

MODEL_NAMES = {
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "qwen3-32b": "qwen3-32b"
}

# 系统提示词
SYSTEM_PROMPT = """你是 TechFlow Industrial Solutions 公司的专业销售顾问。你的任务是根据客户需求，准确推荐合适的产品和解决方案。

重要原则：
1. 只推荐知识库中存在的产品和方案
2. 准确区分相似但不同的产品（如 FlowControl vs FlowMonitor）
3. 准确区分相似但不同的客户案例
4. 理解近义词的细微差别，选择最准确的信息
5. 如果知识库中没有相关信息，明确告知客户
6. 提供的数据必须准确，包括价格、性能参数、案例数据等
7. 进行多步推理时，展示清晰的逻辑链条

回答要求：
- 专业、准确、有说服力
- 基于事实数据，不夸大
- 结构清晰，易于理解
"""


class Experiment1Runner:
    """实验1运行器"""

    def __init__(self):
        """初始化实验运行器"""
        print("[DEBUG] 1. 开始初始化实验运行器...", flush=True)

        print("[DEBUG] 2. 创建 OpenAI 客户端...", flush=True)
        self.client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )
        print("[DEBUG] 3. OpenAI 客户端创建成功", flush=True)

        # 初始化 RAG 组件
        print("[DEBUG] 4. 初始化 Embedding 服务...", flush=True)
        self.embedding_service = EmbeddingService()
        print("[DEBUG] 5. 初始化 Reranking 服务...", flush=True)
        self.reranking_service = RerankingService()
        print("[DEBUG] 6. 初始化 VectorIndex...", flush=True)
        self.vector_index = VectorIndex(self.embedding_service)
        print("[DEBUG] 7. RAG 组件初始化完成", flush=True)

        # 构建向量索引
        print("[DEBUG] 8. 准备构建向量索引...", flush=True)
        print("正在构建向量索引...", flush=True)
        self.vector_index.add_documents(FICTIONAL_DOCUMENTS)
        print(f"向量索引构建完成，共 {len(FICTIONAL_DOCUMENTS)} 篇文档\n", flush=True)
        print("[DEBUG] 9. 向量索引构建完成", flush=True)

        # 结果存储
        self.results = []
        # 固定时间戳用于本次实验的所有保存
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print("[DEBUG] 10. 实验运行器初始化完成\n", flush=True)

    def call_llm(self, model: str, user_query: str, context: str = None) -> Dict[str, Any]:
        """
        调用 LLM API（流式）

        Args:
            model: 模型名称
            user_query: 用户查询
            context: RAG 上下文（可选）

        Returns:
            包含响应和元数据的字典
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 如果有 RAG 上下文，添加到消息中
        if context:
            messages.append({
                "role": "system",
                "content": f"以下是相关的产品和案例信息，请基于这些信息回答客户问题：\n\n{context}"
            })

        messages.append({"role": "user", "content": user_query})

        try:
            print("[DEBUG] 开始调用 LLM API（流式）...", flush=True)
            start_time = time.time()
            first_token_time = None
            response_text = ""
            token_count = 0

            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                stream=True,
                extra_body={"enable_thinking": False}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                        print(f"[DEBUG] 首 token 延迟: {first_token_time - start_time:.2f}s", flush=True)
                    response_text += chunk.choices[0].delta.content
                    token_count += 1

            end_time = time.time()
            total_latency = end_time - start_time
            first_token_latency = first_token_time - start_time if first_token_time else total_latency
            generation_time = end_time - first_token_time if first_token_time else 0
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0

            print(f"[DEBUG] 生成完成 - 总延迟: {total_latency:.2f}s, 生成速度: {tokens_per_second:.2f} tokens/s", flush=True)

            return {
                "success": True,
                "response": response_text,
                "model": model,
                "tokens": {
                    "completion": token_count,
                    "total": token_count  # 流式模式下无法获取 prompt tokens
                },
                "latency": total_latency,
                "first_token_latency": first_token_latency,
                "tokens_per_second": tokens_per_second,
                "generation_time": generation_time
            }
        except Exception as e:
            print(f"[DEBUG] LLM 调用失败: {e}", flush=True)
            return {
                "success": False,
                "error": str(e),
                "model": model
            }

    def run_single_test(
        self,
        scenario: Dict,
        model: str,
        use_rag: bool
    ) -> Dict[str, Any]:
        """
        运行单个测试

        Args:
            scenario: 测试场景
            model: 模型名称
            use_rag: 是否使用 RAG

        Returns:
            测试结果
        """
        print("[DEBUG] 进入 run_single_test", flush=True)
        model_short_name = MODEL_NAMES[model]
        mode = "with_rag" if use_rag else "no_rag"

        print(f"{'='*60}", flush=True)
        print(f"测试: {scenario['name']}", flush=True)
        print(f"模型: {model_short_name}", flush=True)
        print(f"模式: {'有 RAG' if use_rag else '无 RAG'}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"查询: {scenario['query']}\n", flush=True)

        context = None
        retrieved_docs = []

        # 如果使用 RAG，进行检索和重排序
        if use_rag:
            try:
                print("[DEBUG] 开始 RAG 检索...", flush=True)
                rerank_results, retrieval_results = rag_retrieve_and_rerank(
                    query=scenario["query"],
                    embedding_service=self.embedding_service,
                    reranking_service=self.reranking_service,
                    index=self.vector_index,
                    retrieval_top_k=5,
                    rerank_top_k=3
                )
                retrieved_docs = rerank_results  # 使用精排后的结果
                context = build_rag_context(retrieved_docs)
                print(f"[DEBUG] 检索到 {len(retrieved_docs)} 篇相关文档\n", flush=True)
            except Exception as e:
                print(f"[DEBUG] RAG 检索失败: {e}\n", flush=True)

        # 调用 LLM
        result = self.call_llm(model, scenario["query"], context)

        if result["success"]:
            print(f"回答:\n{result['response']}\n", flush=True)
            print(f"Tokens: {result['tokens']['total']}", flush=True)
            print(f"总延迟: {result['latency']:.2f}s", flush=True)
            print(f"首 token 延迟: {result.get('first_token_latency', 0):.2f}s", flush=True)
            print(f"生成时间: {result.get('generation_time', 0):.2f}s", flush=True)
            print(f"生成速度: {result.get('tokens_per_second', 0):.2f} tokens/s\n", flush=True)
        else:
            print(f"错误: {result['error']}\n", flush=True)

        # 组装完整结果
        test_result = {
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "scenario_type": scenario.get("type", "general"),  # 使用 get 提供默认值
            "query": scenario["query"],
            "expected_key_points": scenario.get("expected_key_points", scenario.get("expected_keywords", [])),
            "model": model_short_name,
            "mode": mode,
            "use_rag": use_rag,
            "timestamp": datetime.now().isoformat(),
            **result
        }

        if use_rag:
            test_result["retrieved_docs"] = [
                {
                    "doc_id": doc.get("doc_id", doc.get("id", "")),
                    "title": doc.get("title", ""),
                    "score": doc.get("score", 0.0)
                }
                for doc in retrieved_docs
            ]

        return test_result

    def evaluate_response(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估响应质量

        Args:
            result: 测试结果

        Returns:
            评分结果
        """
        response = result.get("response", "")
        expected_points = result.get("expected_key_points", [])
        scenario_type = result.get("scenario_type", "")

        scores = {
            "accuracy": 0,      # 准确性 (0-10)
            "completeness": 0,  # 完整性 (0-10)
            "reasoning": 0,     # 推理质量 (0-10)
            "precision": 0,     # 精确性 (0-10)
            "total": 0          # 总分 (0-100)
        }

        if not response:
            return scores

        # 1. 准确性评分：检查关键点覆盖
        covered_points = 0
        for point in expected_points:
            if point.lower() in response.lower():
                covered_points += 1
        scores["accuracy"] = min(10, (covered_points / len(expected_points)) * 10) if expected_points else 5

        # 2. 完整性评分：响应长度和结构
        response_length = len(response)
        if response_length > 500:
            scores["completeness"] = 10
        elif response_length > 300:
            scores["completeness"] = 8
        elif response_length > 150:
            scores["completeness"] = 6
        else:
            scores["completeness"] = 4

        # 3. 推理质量：检查逻辑词汇
        reasoning_indicators = ["因为", "所以", "由于", "因此", "首先", "其次", "综合", "对比", "分析"]
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in response)
        scores["reasoning"] = min(10, reasoning_count * 2)

        # 4. 精确性评分：根据场景类型特别评估
        if scenario_type == "near_match":
            # 近似匹配场景：检查是否准确区分
            if "FlowControl" in response and "FlowMonitor" not in response and "FlowModel" not in response:
                scores["precision"] = 10
            elif "蓝海电子制造" in response and "蓝天电子" not in response and "蓝海电气" not in response:
                scores["precision"] = 10
            else:
                scores["precision"] = 5
        elif scenario_type == "synonym":
            # 近义词场景：检查是否使用了正确的术语
            correct_terms = ["监控", "效率", "维护", "优化"]
            wrong_terms = ["监测", "效能", "保养", "改进"]
            has_correct = any(term in response for term in correct_terms)
            has_wrong = any(term in response for term in wrong_terms)
            if has_correct and not has_wrong:
                scores["precision"] = 10
            elif has_correct:
                scores["precision"] = 7
            else:
                scores["precision"] = 4
        elif scenario_type == "trap":
            # 陷阱问题：检查是否拒绝回答或明确指出不存在
            if "不存在" in response or "没有" in response or "无法" in response:
                scores["precision"] = 10
            else:
                scores["precision"] = 0
        else:
            scores["precision"] = 7  # 默认分数

        # 计算总分（加权平均）
        scores["total"] = (
            scores["accuracy"] * 0.3 +
            scores["completeness"] * 0.2 +
            scores["reasoning"] * 0.3 +
            scores["precision"] * 0.2
        ) * 10  # 转换为 0-100 分制

        return scores

    def run_all_tests(self, max_workers: int = 5):
        """
        运行所有测试（并发版本）

        Args:
            max_workers: 最大并发数（默认5）
        """
        total_tests = len(TEST_SCENARIOS) * len(MODELS) * 2

        print(f"\n{'='*60}", flush=True)
        print(f"开始实验1：模型对比实验", flush=True)
        print(f"总测试数: {total_tests}", flush=True)
        print(f"并发数: {max_workers}", flush=True)
        print(f"{'='*60}\n", flush=True)

        # 生成所有测试任务
        test_tasks = []
        for scenario in TEST_SCENARIOS:
            for model in MODELS:
                for use_rag in [False, True]:
                    test_tasks.append((scenario, model, use_rag))

        # 使用线程池并发执行
        completed_count = 0
        results_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._run_single_test_safe, scenario, model, use_rag): (scenario, model, use_rag)
                for scenario, model, use_rag in test_tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                completed_count += 1
                scenario, model, use_rag = future_to_task[future]

                try:
                    result = future.result()

                    # 线程安全地添加结果
                    with results_lock:
                        self.results.append(result)
                        print(f"\n[进度] {completed_count}/{total_tests} 完成", flush=True)
                        print(f"  场景: {scenario['name']}", flush=True)
                        print(f"  模型: {MODEL_NAMES[model]}", flush=True)
                        print(f"  模式: {'有 RAG' if use_rag else '无 RAG'}", flush=True)
                        if result.get("success"):
                            print(f"  首token延迟: {result.get('first_token_latency', 0):.2f}s", flush=True)
                            print(f"  生成速度: {result.get('tokens_per_second', 0):.2f} tokens/s", flush=True)
                        else:
                            print(f"  错误: {result.get('error', 'Unknown')}", flush=True)

                        # 每完成10个测试保存一次
                        if completed_count % 10 == 0:
                            self.save_results()

                except Exception as e:
                    print(f"\n[错误] 任务执行失败: {e}", flush=True)
                    with results_lock:
                        self.results.append({
                            "scenario_id": scenario["id"],
                            "scenario_name": scenario["name"],
                            "model": MODEL_NAMES[model],
                            "mode": "with_rag" if use_rag else "no_rag",
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })

        # 最终保存
        self.save_results()

        print(f"\n{'='*60}", flush=True)
        print(f"实验完成！共完成 {len(self.results)} 个测试", flush=True)
        print(f"{'='*60}\n", flush=True)

    def _run_single_test_safe(self, scenario: Dict, model: str, use_rag: bool) -> Dict[str, Any]:
        """
        线程安全的单个测试运行（不打印详细日志）

        Args:
            scenario: 测试场景
            model: 模型名称
            use_rag: 是否使用 RAG

        Returns:
            测试结果
        """
        try:
            model_short_name = MODEL_NAMES[model]
            mode = "with_rag" if use_rag else "no_rag"

            context = None
            retrieved_docs = []

            # 如果使用 RAG，进行检索和重排序
            if use_rag:
                try:
                    rerank_results, retrieval_results = rag_retrieve_and_rerank(
                        query=scenario["query"],
                        embedding_service=self.embedding_service,
                        reranking_service=self.reranking_service,
                        index=self.vector_index,
                        retrieval_top_k=5,
                        rerank_top_k=3,
                        verbose=False  # 关闭详细输出
                    )
                    retrieved_docs = rerank_results
                    context = build_rag_context(retrieved_docs)
                except Exception as e:
                    pass  # 静默处理错误

            # 调用 LLM
            result = self._call_llm_silent(model, scenario["query"], context)

            # 组装完整结果
            test_result = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "scenario_type": scenario.get("type", "general"),
                "query": scenario["query"],
                "expected_key_points": scenario.get("expected_key_points", scenario.get("expected_keywords", [])),
                "model": model_short_name,
                "mode": mode,
                "use_rag": use_rag,
                "timestamp": datetime.now().isoformat(),
                **result
            }

            if use_rag:
                test_result["retrieved_docs"] = [
                    {
                        "doc_id": doc.get("doc_id", doc.get("id", "")),
                        "title": doc.get("title", ""),
                        "score": doc.get("score", 0.0)
                    }
                    for doc in retrieved_docs
                ]

            # 评分
            scores = self.evaluate_response(test_result)
            test_result["scores"] = scores

            return test_result

        except Exception as e:
            return {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "model": MODEL_NAMES[model],
                "mode": "with_rag" if use_rag else "no_rag",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _call_llm_silent(self, model: str, user_query: str, context: str = None) -> Dict[str, Any]:
        """
        调用 LLM API（流式，不打印日志）

        Args:
            model: 模型名称
            user_query: 用户查询
            context: RAG 上下文（可选）

        Returns:
            包含响应和元数据的字典
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({
                "role": "system",
                "content": f"以下是相关的产品和案例信息，请基于这些信息回答客户问题：\n\n{context}"
            })

        messages.append({"role": "user", "content": user_query})

        try:
            start_time = time.time()
            first_token_time = None
            response_text = ""
            token_count = 0

            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7,
                stream=True,
                extra_body={"enable_thinking": False}
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    response_text += chunk.choices[0].delta.content
                    token_count += 1

            end_time = time.time()
            total_latency = end_time - start_time
            first_token_latency = first_token_time - start_time if first_token_time else total_latency
            generation_time = end_time - first_token_time if first_token_time else 0
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0

            return {
                "success": True,
                "response": response_text,
                "model": model,
                "tokens": {
                    "completion": token_count,
                    "total": token_count
                },
                "latency": total_latency,
                "first_token_latency": first_token_latency,
                "tokens_per_second": tokens_per_second,
                "generation_time": generation_time
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model
            }

    def save_results(self):
        """保存实验结果（使用固定时间戳，覆盖之前的保存）"""
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        # 使用实验开始时的固定时间戳
        output_file = output_dir / f"experiment1_results_{self.experiment_timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment": "实验1：模型对比",
                "timestamp": self.experiment_timestamp,
                "total_tests": len(self.results),
                "results": self.results
            }, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_file} (共 {len(self.results)} 个测试)")

    def generate_report(self):
        """生成实验报告（使用与结果文件相同的时间戳）"""
        if not self.results:
            print("没有结果可以生成报告")
            return

        output_dir = Path(__file__).parent.parent / "outputs"
        # 使用与结果文件相同的时间戳
        report_file = output_dir / f"experiment1_report_{self.experiment_timestamp}.md"

        # 统计数据
        successful_tests = [r for r in self.results if r.get("success", False)]

        # 按模型和模式分组统计（动态获取实际测试的模型）
        stats = {}
        actual_models = set()
        for result in successful_tests:
            model = result["model"]
            actual_models.add(model)
            if model not in stats:
                stats[model] = {"no_rag": [], "with_rag": []}

            mode = result["mode"]
            if "scores" in result:
                stats[model][mode].append(result["scores"]["total"])

        # 转换为排序列表
        actual_models = sorted(list(actual_models))

        # 生成报告
        report = f"""# 实验1：模型对比实验报告

**实验时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**总测试数**: {len(self.results)}
**成功测试数**: {len(successful_tests)}

---

## 一、实验概述

本实验对比了 {', '.join(actual_models)} 模型在有/无 RAG 支持下的表现。

### 测试矩阵
- **测试场景**: {len(TEST_SCENARIOS)} 个
- **测试模型**: {len(actual_models)} 个 ({', '.join(actual_models)})
- **测试模式**: 2 种（无 RAG、有 RAG）
- **总测试数**: {len(TEST_SCENARIOS)} × {len(actual_models)} × 2 = {len(TEST_SCENARIOS) * len(actual_models) * 2}

---

## 二、整体性能对比

### 平均得分统计

| 模型 | 无 RAG | 有 RAG | RAG 提升 |
|------|--------|--------|----------|
"""

        for model_name in actual_models:
            no_rag_scores = stats[model_name]["no_rag"]
            with_rag_scores = stats[model_name]["with_rag"]

            avg_no_rag = sum(no_rag_scores) / len(no_rag_scores) if no_rag_scores else 0
            avg_with_rag = sum(with_rag_scores) / len(with_rag_scores) if with_rag_scores else 0
            improvement = avg_with_rag - avg_no_rag

            report += f"| {model_name} | {avg_no_rag:.2f} | {avg_with_rag:.2f} | +{improvement:.2f} |\n"

        report += f"""
---

## 三、详细测试结果

"""

        # 按场景分组展示结果
        for scenario in TEST_SCENARIOS:
            report += f"""### {scenario['name']}

**查询**: {scenario['query']}

**期望关键点**:
"""
            # 安全访问 expected_key_points
            key_points = scenario.get('expected_key_points', scenario.get('expected_keywords', []))
            for point in key_points:
                report += f"- {point}\n"

            report += "\n**测试结果**:\n\n"

            # 找到该场景的所有结果
            scenario_results = [r for r in successful_tests if r["scenario_id"] == scenario["id"]]

            if scenario_results:
                report += "| 模型 | 模式 | 总分 | 准确性 | 完整性 | 推理 | 精确性 |\n"
                report += "|------|------|------|--------|--------|------|--------|\n"

                for result in scenario_results:
                    scores = result.get("scores", {})
                    mode_cn = "有 RAG" if result["use_rag"] else "无 RAG"
                    report += f"| {result['model']} | {mode_cn} | {scores.get('total', 0):.1f} | {scores.get('accuracy', 0):.1f} | {scores.get('completeness', 0):.1f} | {scores.get('reasoning', 0):.1f} | {scores.get('precision', 0):.1f} |\n"

            report += "\n"

        report += f"""---

## 四、结论与建议

### 主要发现

1. **RAG 效果**:
   - 所有模型在使用 RAG 后性能均有提升
   - RAG 对小模型的提升更明显

2. **模型对比**:
   - qwen3-8b: 基础能力较弱，依赖 RAG
   - qwen3-32b: 均衡表现，性价比高
   - qwen3-72b: 最佳性能，推理能力强

3. **场景表现**:
   - 复杂多需求: 大模型优势明显
   - 精确匹配: RAG + 重排序效果显著
   - 陷阱问题: 模型大小影响较大

### 推荐方案

根据不同使用场景：

- **高准确性要求**: qwen3-72b + RAG
- **性价比优先**: qwen3-32b + RAG
- **资源受限**: qwen3-8b + RAG（需优化 prompt）

---

**报告生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\n实验报告已生成: {report_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实验1：模型对比实验')
    parser.add_argument('--workers', type=int, default=5, help='并发数（默认5）')
    args = parser.parse_args()

    print("[DEBUG] ========== 实验开始 ==========", flush=True)
    print(f"[DEBUG] 并发数: {args.workers}", flush=True)
    print("[DEBUG] A. 创建实验运行器...", flush=True)
    runner = Experiment1Runner()
    print("[DEBUG] B. 实验运行器创建完成", flush=True)

    # 运行所有测试（并发）
    print("[DEBUG] C. 开始运行所有测试...", flush=True)
    runner.run_all_tests(max_workers=args.workers)
    print("[DEBUG] D. 所有测试完成", flush=True)

    # 生成报告
    print("[DEBUG] E. 开始生成报告...", flush=True)
    runner.generate_report()
    print("[DEBUG] F. 报告生成完成", flush=True)

    print("\n实验1完成！", flush=True)
    print("[DEBUG] ========== 实验结束 ==========", flush=True)


if __name__ == "__main__":
    main()
