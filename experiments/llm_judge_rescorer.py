"""
实验1改进版：使用 LLM 作为评委进行质量评分
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 评委模型配置（使用最强的模型作为评委）
JUDGE_MODEL = "qwen3-235b-a22b-instruct-2507"  # 或者使用外部API如GPT-4

# 评分标准（Rubric）
SCORING_RUBRIC = """
你是一位专业的AI系统评估专家。请根据以下标准对销售AI助手的回答进行评分：

## 评分维度（每项0-10分）

### 1. 准确性（Accuracy）
- 10分：所有推荐的产品、价格、参数完全准确，与知识库一致
- 7-9分：大部分信息准确，有1-2处小错误或遗漏
- 4-6分：部分信息准确，但有明显错误或张冠李戴
- 0-3分：严重错误，推荐不存在的产品或严重误导客户

### 2. 完整性（Completeness）
- 10分：完整回答了所有子问题，提供了所需的所有信息
- 7-9分：回答了主要问题，但有1-2个细节遗漏
- 4-6分：只回答了部分问题，遗漏较多
- 0-3分：大量遗漏，几乎没有实质内容

### 3. 推理质量（Reasoning）
- 10分：逻辑清晰，推理链条完整，有充分的论据支持
- 7-9分：推理基本合理，有少量跳跃
- 4-6分：推理不够充分，缺少关键步骤
- 0-3分：逻辑混乱或缺乏推理

### 4. 专业性（Professionalism）
- 10分：专业术语准确，表述清晰，结构良好
- 7-9分：整体专业，有小瑕疵
- 4-6分：专业性一般，表述不够清晰
- 0-3分：不专业或表述混乱

## 评分格式（必须返回有效的JSON）

{
  "accuracy": <0-10的分数>,
  "accuracy_reason": "<简要说明为什么给这个分数>",
  "completeness": <0-10的分数>,
  "completeness_reason": "<简要说明>",
  "reasoning": <0-10的分数>,
  "reasoning_reason": "<简要说明>",
  "professionalism": <0-10的分数>,
  "professionalism_reason": "<简要说明>",
  "hallucination_detected": <true/false>,
  "hallucination_details": "<如果发现幻觉，说明具体内容>",
  "overall_comment": "<总体评价，2-3句话>"
}

## 注意事项
1. 如果回答中提到了知识库中不存在的产品、价格、案例，标记为幻觉
2. 严格按照JSON格式输出，不要添加任何额外文字
3. 评分要客观公正，有理有据
"""


class LLMJudge:
    """LLM 评委"""

    def __init__(self, model: str = JUDGE_MODEL):
        """初始化评委"""
        self.client = OpenAI(
            api_key=os.getenv("QWEN_TOKEN"),
            base_url=os.getenv("QWEN_API_BASE")
        )
        self.model = model

    def evaluate(
        self,
        query: str,
        response: str,
        knowledge_base_context: str,
        expected_points: list = None
    ) -> Dict[str, Any]:
        """
        使用 LLM 评估回答质量

        Args:
            query: 用户问题
            response: AI回答
            knowledge_base_context: 知识库相关内容
            expected_points: 期望的关键点（可选）

        Returns:
            评分结果
        """

        # 构建评分提示
        eval_prompt = f"""
请评估以下销售AI助手的回答质量。

【客户问题】
{query}

【知识库相关内容】
{knowledge_base_context}

【AI助手的回答】
{response}
"""

        if expected_points:
            eval_prompt += f"""
【期望包含的关键点】
{', '.join(expected_points)}
"""

        eval_prompt += """

请严格按照评分标准进行评分，并以JSON格式输出结果。
"""

        try:
            # 调用评委模型
            judge_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SCORING_RUBRIC},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.3,  # 低温度保证评分稳定性
                max_tokens=1000,
                extra_body={"enable_thinking": False}
            )

            # 解析评分结果
            judge_text = judge_response.choices[0].message.content.strip()

            # 尝试提取JSON
            # 处理可能的markdown代码块
            if "```json" in judge_text:
                judge_text = judge_text.split("```json")[1].split("```")[0].strip()
            elif "```" in judge_text:
                judge_text = judge_text.split("```")[1].split("```")[0].strip()

            scores = json.loads(judge_text)

            # 计算总分
            scores["total"] = (
                scores["accuracy"] * 0.3 +
                scores["completeness"] * 0.25 +
                scores["reasoning"] * 0.25 +
                scores["professionalism"] * 0.2
            ) * 10  # 转换为0-100分制

            scores["success"] = True
            return scores

        except json.JSONDecodeError as e:
            print(f"警告：评委返回的不是有效JSON: {judge_text[:200]}")
            return {
                "success": False,
                "error": "JSON解析失败",
                "raw_response": judge_text[:500],
                "total": 0
            }
        except Exception as e:
            print(f"评分失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "total": 0
            }


def rescore_existing_results(results_file: str, output_file: str = None, judge_model: str = JUDGE_MODEL, max_workers: int = 8):
    """
    对已有的实验结果重新评分

    Args:
        results_file: 原始结果JSON文件路径
        output_file: 输出文件路径（可选）
        judge_model: 评委模型名称
        max_workers: 并发数（默认8）
    """
    print(f"正在加载结果文件: {results_file}")

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']
    judge = LLMJudge(judge_model)

    print(f"共 {len(results)} 个测试结果需要重新评分")
    print(f"使用评委模型: {judge_model}")
    print(f"并发数: {max_workers}")
    print("开始使用 LLM 评委重新评分...\n")

    # 准备评分任务
    def evaluate_single_result(i: int, result: dict):
        """评估单个结果"""
        if not result.get('success'):
            print(f"[{i}/{len(results)}] 跳过失败的测试")
            return i, None

        print(f"[{i}/{len(results)}] 评分: {result['scenario_name']} - {result['model']} - {result['mode']}")

        # 获取知识库上下文（如果有RAG结果）
        kb_context = ""
        if result.get('use_rag') and result.get('retrieved_docs'):
            kb_context = "检索到的相关文档：\n"
            for doc in result['retrieved_docs']:
                kb_context += f"- {doc['title']}\n"

        # 使用LLM评委评分
        scores = judge.evaluate(
            query=result['query'],
            response=result['response'],
            knowledge_base_context=kb_context,
            expected_points=result.get('expected_key_points', [])
        )

        if scores.get('success'):
            print(f"  总分: {scores['total']:.2f}")
            print(f"  准确性: {scores['accuracy']:.1f} - {scores.get('accuracy_reason', '')[:50]}...")
        else:
            print(f"  评分失败: {scores.get('error', 'Unknown')}")
        print()

        return i, scores

    # 并发评分
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(evaluate_single_result, i, result): i
            for i, result in enumerate(results, 1)
        }

        # 收集结果
        scores_by_index = {}
        for future in as_completed(future_to_index):
            idx, scores = future.result()
            if scores is not None:
                scores_by_index[idx] = scores

    # 按顺序更新结果
    for i, result in enumerate(results, 1):
        if i in scores_by_index:
            scores = scores_by_index[i]
            result['llm_scores'] = scores
            result['original_scores'] = result.get('scores', {})  # 保留原始评分用于对比
            result['scores'] = scores  # 用LLM评分替换

    # 保存结果
    if output_file is None:
        timestamp = Path(results_file).stem.split('_')[-1]
        output_file = f"outputs/experiment1_results_llm_scored_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n重新评分完成！")
    print(f"结果已保存到: {output_file}")

    return output_file


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='使用LLM评委重新评分')
    parser.add_argument('--input', type=str, required=True, help='输入的结果JSON文件')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    parser.add_argument('--judge-model', type=str, default=JUDGE_MODEL, help='评委模型')
    parser.add_argument('--workers', type=int, default=8, help='并发数（默认8）')

    args = parser.parse_args()

    rescore_existing_results(args.input, args.output, args.judge_model, args.workers)


if __name__ == "__main__":
    main()
