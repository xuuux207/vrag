"""
用户输入完整性判断Agent
判断用户是否说完话，处理停顿思考的情况
"""

import logging
from typing import List, Dict
from src.llm.qwen_service import QwenService
from src.config.settings import settings

logger = logging.getLogger(__name__)


class InputCompletionAgent:
    """用户输入完整性判断Agent"""

    def __init__(self):
        """初始化，使用sub_model（qwen3-8b）提供快速判断"""
        self.sub_llm = QwenService(
            api_base=settings.qwen.api_base,
            model=settings.qwen.sub_model,
            token=settings.qwen.token,
            temperature=0.1,  # 低温度提高确定性
        )

        self.system_prompt = """你是一个智能语音助手的输入判断器。
任务：判断用户的语音输入是否已经完整表达了意图。

判断标准（重要）：
- 有明确的疑问句或陈述句（即使带口语化开头如"那个"、"嗯"） -> 1
- 完整的请求或问题（如"你能帮我做什么吗"、"我想了解XX"） -> 1
- 只有语气词或开头（如"那个..."、"嗯..."、"我想要..."且无后续） -> 0
- 句子明显被截断（如"能不能帮我"后无任何动词或对象） -> 0

示例：
"那个，我就想问。你能帮我做什么吗？" -> 1（完整问句）
"我想问一下关于产品的信息" -> 1（完整请求）
"那个..." -> 0（只有开头）
"我想要" -> 0（明显未完成）

只输出0（未完成）或1（已完成），不要解释。"""

    def is_input_complete(
        self,
        user_input: str,
        recent_messages: List[Dict[str, str]] = None
    ) -> bool:
        """
        判断用户输入是否完整（单次判断）

        Args:
            user_input: 用户输入文本
            recent_messages: 最近的对话历史（用于上下文判断）

        Returns:
            True: 输入完整
            False: 输入不完整
        """
        # 快速检查：非常短的输入视为不完整
        if len(user_input.strip()) < 2:
            logger.debug("输入过短，视为不完整")
            return False

        # 构建上下文（如果有历史对话）
        context = ""
        if recent_messages and len(recent_messages) > 0:
            context = "最近对话:\n"
            for msg in recent_messages[-4:]:  # 只看最近2轮（4条消息）
                role = "用户" if msg["role"] == "user" else "助手"
                content = msg["content"][:60]  # 截取前60字符
                context += f"{role}: {content}...\n"
            context += "\n"

        try:
            logger.debug(f"输入完整性判断: {user_input[:50]}")

            # 构建判断消息
            user_message = f"{context}当前用户输入: {user_input}\n\n用户是否说完了? (0=未完成, 1=已完成)"

            # 调用sub_model判断
            response = self.sub_llm.chat(
                messages=[{"role": "user", "content": user_message}],
                system_prompt=self.system_prompt
            )

            # 解析结果（提取0或1）
            result = response.strip()
            logger.debug(f"输入完整性判断 - 模型返回: {result}")

            if "1" in result:
                logger.info("✓ 输入完整")
                return True
            elif "0" in result:
                logger.info("✗ 输入未完成")
                return False
            else:
                # 返回异常，默认为完整
                logger.warning(f"输入完整性判断返回异常: {result}，默认视为完整")
                return True

        except Exception as e:
            logger.error(f"输入完整性判断失败: {str(e)}，默认视为完整")
            return True  # 异常时默认为完整

