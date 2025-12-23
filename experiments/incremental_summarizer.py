"""
渐进式总结模块 - 实现边输入边总结的功能
模拟真实语音场景：用户一段一段说话，系统实时处理
"""

import json
import time
from typing import Dict, List, Optional
from openai import OpenAI


# ========== Prompt模板 ==========

INCREMENTAL_SUMMARY_PROMPT = """你是一个专业的语音助手，正在实时处理用户的语音输入。

【当前累积总结】：
{previous_summary}

【新输入片段】：
{new_segment}

请更新总结，要求：
1. 提取新片段中的关键信息（意图、实体、需求、约束）
2. 与之前的总结合并，去除重复内容
3. 过滤口语化表达（嗯、那个、就是、对了、啊等）
4. 过滤无关寒暄和闲聊
5. 保留所有有价值的关键信息

返回JSON格式（不要markdown代码块标记）：
{{
  "updated_summary": "更新后的简洁总结（仅包含关键信息）",
  "new_key_points": ["本段新增的关键点1", "关键点2"],
  "filtered_noise": ["本段过滤的干扰项1", "干扰项2"]
}}

注意：
- updated_summary 应该是完整的、自包含的总结
- 重复的信息只保留一次
- 口语化表达必须过滤掉
"""

FINAL_STRUCTURE_EXTRACTION_PROMPT = """你是一个专业的语音助手。用户刚刚完成了一段长时间的语音输入，你已经进行了实时总结。

【最终总结】：
{final_summary}

【原始输入段数】：{segment_count}
【总字数】：{total_length}

现在请进行最终的结构化提取：

返回JSON格式（不要markdown代码块标记）：
{{
  "main_intent": "主要意图（产品咨询/技术咨询/客户查询/问题反馈等）",
  "key_points": ["关键点1", "关键点2", "关键点3"],
  "entities": ["实体1", "实体2"],
  "constraints": ["约束条件1", "约束条件2"],
  "is_multi_question": false,
  "concise_query": "用于RAG检索的简化query（50-150字）"
}}

要求：
1. main_intent 要准确反映用户的核心目的
2. key_points 应该全面覆盖所有重要信息
3. entities 包括公司名、产品名、人名、地名等
4. constraints 包括预算、时间、技术要求等约束
5. is_multi_question: 如果包含多个独立问题，设为true
6. concise_query 要简洁但包含所有关键信息，适合RAG检索
"""


# ========== 渐进式总结类 ==========

class IncrementalSummarizer:
    """渐进式总结器 - 边输入边总结"""

    def __init__(self, llm_client: OpenAI, model_name: str = "qwen-plus"):
        self.llm = llm_client
        self.model_name = model_name
        self.current_summary = ""
        self.all_key_points = []
        self.all_filtered_noise = []
        self.segment_count = 0
        self.total_length = 0

    def reset(self):
        """重置状态"""
        self.current_summary = ""
        self.all_key_points = []
        self.all_filtered_noise = []
        self.segment_count = 0
        self.total_length = 0

    def add_segment(self, segment_text: str) -> Dict:
        """
        添加新的输入片段，更新总结

        Args:
            segment_text: 新输入的文本片段

        Returns:
            本次更新的结果
        """
        self.segment_count += 1
        self.total_length += len(segment_text)

        # 第一段输入，直接总结
        if self.segment_count == 1:
            return self._first_segment_summary(segment_text)

        # 后续输入，增量更新
        return self._incremental_update(segment_text)

    def _first_segment_summary(self, segment_text: str) -> Dict:
        """处理第一段输入"""
        # 第一段直接提取关键信息
        prompt = f"""这是用户的第一段输入，请提取关键信息并过滤噪音。

用户输入：
{segment_text}

返回JSON（不要markdown标记）：
{{
  "summary": "简洁的总结",
  "key_points": ["关键点1", "关键点2"],
  "filtered_noise": ["过滤的干扰项1", "干扰项2"]
}}
"""

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = self._clean_json_response(response.choices[0].message.content)
        result = json.loads(content)

        self.current_summary = result["summary"]
        self.all_key_points.extend(result.get("key_points", []))
        self.all_filtered_noise.extend(result.get("filtered_noise", []))

        return {
            "segment_number": self.segment_count,
            "updated_summary": self.current_summary,
            "new_key_points": result.get("key_points", []),
            "filtered_noise": result.get("filtered_noise", []),
            "summary_length": len(self.current_summary)
        }

    def _incremental_update(self, segment_text: str) -> Dict:
        """增量更新总结"""
        prompt = INCREMENTAL_SUMMARY_PROMPT.format(
            previous_summary=self.current_summary or "（这是第一段输入）",
            new_segment=segment_text
        )

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = self._clean_json_response(response.choices[0].message.content)
        result = json.loads(content)

        # 更新状态
        self.current_summary = result["updated_summary"]
        new_points = result.get("new_key_points", [])
        self.all_key_points.extend(new_points)

        filtered = result.get("filtered_noise", [])
        self.all_filtered_noise.extend(filtered)

        return {
            "segment_number": self.segment_count,
            "updated_summary": self.current_summary,
            "new_key_points": new_points,
            "filtered_noise": filtered,
            "summary_length": len(self.current_summary)
        }

    def finalize(self) -> Dict:
        """
        完成所有输入后，进行最终的结构化提取

        Returns:
            结构化的查询信息
        """
        prompt = FINAL_STRUCTURE_EXTRACTION_PROMPT.format(
            final_summary=self.current_summary,
            segment_count=self.segment_count,
            total_length=self.total_length
        )

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        content = self._clean_json_response(response.choices[0].message.content)
        structured = json.loads(content)

        return {
            "final_summary": self.current_summary,
            "structured_query": structured,
            "total_segments": self.segment_count,
            "total_length": self.total_length,
            "total_key_points": len(self.all_key_points),
            "total_filtered_noise": len(self.all_filtered_noise),
            "compression_ratio": len(self.current_summary) / self.total_length if self.total_length > 0 else 0
        }

    def _clean_json_response(self, content: str) -> str:
        """清理LLM返回的JSON内容"""
        content = content.strip()

        # 移除markdown代码块标记
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        return content.strip()


# ========== 测试代码 ==========

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # 使用通义千问API
    llm_client = OpenAI(
        api_key=os.getenv("QWEN_TOKEN"),
        base_url=os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )

    # 创建渐进式总结器
    summarizer = IncrementalSummarizer(llm_client, model_name="qwen-plus")

    # 测试数据
    test_segments = [
        "你好，嗯...我想咨询一下你们公司的产品。我们是一家金融科技公司，最近在做数字化转型。",
        "那个...我听说你们之前给中国银行做过项目。我想了解一下那个项目的情况。",
        "另外啊，我们的预算大概在50万左右。想知道能不能定制开发。",
        "对了，我们希望3个月内上线。还有，对数据安全特别重视。"
    ]

    print("=== 渐进式总结测试 ===\n")

    for i, segment in enumerate(test_segments, 1):
        print(f"【第{i}段输入】: {segment}")
        result = summarizer.add_segment(segment)
        print(f"【更新后总结】: {result['updated_summary']}")
        print(f"【新增关键点】: {result['new_key_points']}")
        print(f"【过滤噪音】: {result['filtered_noise']}")
        print(f"【总结长度】: {result['summary_length']}字\n")

    print("\n=== 最终结构化提取 ===\n")
    final = summarizer.finalize()
    print(json.dumps(final, ensure_ascii=False, indent=2))
