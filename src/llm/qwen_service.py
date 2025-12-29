"""
Qwen LLM 服务
支持流式调用和RAG上下文注入
"""

import logging
from typing import List, Dict, Iterator, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class QwenService:
    """Qwen LLM服务（流式调用）"""

    def __init__(
        self,
        api_base: str,
        model: str,
        token: str,
        temperature: float = 0.7,
        is_local_vllm: bool = False,
    ):
        """
        初始化Qwen服务

        Args:
            api_base: API基础URL
            model: 模型名称（如qwen-plus）
            token: API令牌
            temperature: 温度参数（0-1）
            is_local_vllm: 是否为本地vllm服务（用于兼容性处理）
        """
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.is_local_vllm = is_local_vllm

        # 初始化OpenAI客户端（兼容Qwen API）
        self.client = OpenAI(
            api_key=token if not is_local_vllm else "EMPTY",  # vllm不需要token
            base_url=api_base,
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Iterator[str]:
        """
        流式对话

        Args:
            messages: 对话历史，格式：[{"role": "user", "content": "..."}]
            system_prompt: 系统提示词（可选）

        Yields:
            生成的文本片段
        """
        # 构建完整messages
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": full_messages,
                "temperature": self.temperature,
                "stream": True,
            }

            # 仅远程API支持extra_body参数
            if not self.is_local_vllm:
                request_params["extra_body"] = {"enable_thinking": False}
            else:
                request_params["extra_body"]= {"chat_template_kwargs":{"enable_thinking": False}}

            # 使用OpenAI SDK进行流式调用
            stream = self.client.chat.completions.create(**request_params)

            # 逐个返回chunk
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # logger.info(f"[流式] yield chunk: {repr(content)}")
                    yield content

        except Exception as e:
            logger.error(f"Qwen API调用失败: {str(e)}")
            raise RuntimeError(f"LLM调用失败: {str(e)}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        非流式对话（一次性返回完整结果）

        Args:
            messages: 对话历史
            system_prompt: 系统提示词（可选）

        Returns:
            完整响应文本
        """
        # 构建完整messages
        full_messages = []

        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        full_messages.extend(messages)

        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": full_messages,
                "temperature": self.temperature,
                "stream": False,
            }

            # 仅远程API支持extra_body参数
            if not self.is_local_vllm:
                request_params["extra_body"] = {"enable_thinking": False}
            else:
                request_params["extra_body"]= {"chat_template_kwargs":{"enable_thinking": False}}
                
            # 使用OpenAI SDK进行非流式调用
            response = self.client.chat.completions.create(**request_params)

            content = response.choices[0].message.content
            return content if content else ""

        except Exception as e:
            logger.error(f"Qwen API调用失败: {str(e)}")
            raise RuntimeError(f"LLM调用失败: {str(e)}")

    def chat_with_rag(
        self,
        query: str,
        rag_context: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = True,
    ):
        """
        带RAG上下文的对话

        Args:
            query: 用户查询
            rag_context: RAG检索的上下文
            messages: 对话历史
            system_prompt: 系统提示词（可选）
            stream: 是否流式输出

        Returns:
            流式：Iterator[str]
            非流式：str
        """
        # 构建带RAG的用户消息
        enhanced_query = f"{rag_context}\n\n用户问题：{query}"

        # 添加到消息列表
        new_messages = messages + [{"role": "user", "content": enhanced_query}]

        # 调用对应方法
        if stream:
            return self.chat_stream(new_messages, system_prompt=system_prompt)
        else:
            return self.chat(new_messages, system_prompt=system_prompt)
