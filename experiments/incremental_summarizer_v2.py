"""
渐进式总结模块 V2 - 简化版
只保留最核心的summary，提升速度
"""

import json
import time
from typing import Dict, List
from openai import OpenAI


# 简化的prompt - 只要求返回summary
SIMPLE_SUMMARY_PROMPT = """你是一个语音助手，正在实时总结用户的语音输入。

【之前的总结】：
{previous_summary}

【新输入】：
{new_segment}

请更新总结，要求：
1. **保留之前总结中的所有关键信息**（不要丢失）
2. **追加新输入中的重要信息**
3. 过滤口语词、寒暄等噪音
4. 合并重复信息

只返回一行简洁的总结文本（累积式的完整总结），不要JSON格式，不要markdown："""


class SimpleSummarizer:
    """简化的渐进式总结器 - 只输出summary文本"""

    def __init__(self, llm_client: OpenAI, model_name: str = "qwen3-8b"):
        self.llm = llm_client
        self.model_name = model_name
        self.current_summary = ""
        self.segment_count = 0
        self.total_length = 0
        self.segment_times = []  # 记录每段处理时间

    def reset(self):
        """重置状态"""
        self.current_summary = ""
        self.segment_count = 0
        self.total_length = 0
        self.segment_times = []

    def add_segment(self, segment_text: str, simulate_delay: bool = True) -> Dict:
        """
        添加新片段

        Args:
            segment_text: 新输入文本
            simulate_delay: 是否模拟真实语音输入延迟

        Returns:
            更新结果
        """
        if simulate_delay:
            # 模拟用户说话时间：按每秒3个字计算
            speech_time = len(segment_text) / 3.0
            time.sleep(speech_time)

        seg_start = time.time()
        self.segment_count += 1
        self.total_length += len(segment_text)

        if self.segment_count == 1:
            # 第一段，直接作为summary
            self.current_summary = segment_text.strip()
        else:
            # 后续片段，增量更新
            prompt = SIMPLE_SUMMARY_PROMPT.format(
                previous_summary=self.current_summary or "（无）",
                new_segment=segment_text
            )

            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )

            self.current_summary = response.choices[0].message.content.strip()

        seg_time = time.time() - seg_start
        self.segment_times.append(seg_time)

        return {
            "segment_number": self.segment_count,
            "summary": self.current_summary,
            "summary_length": len(self.current_summary),
            "processing_time": seg_time
        }

    def get_final_summary(self) -> str:
        """返回最终总结"""
        return self.current_summary

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_segments": self.segment_count,
            "total_input_length": self.total_length,
            "final_summary_length": len(self.current_summary),
            "compression_ratio": len(self.current_summary) / self.total_length if self.total_length > 0 else 0,
            "total_processing_time": sum(self.segment_times),
            "avg_segment_time": sum(self.segment_times) / len(self.segment_times) if self.segment_times else 0
        }
