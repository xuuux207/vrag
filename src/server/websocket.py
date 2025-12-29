"""
WebSocket 语音交互
前端：Azure Speech SDK 语音识别/合成
后端：VoiceAssistant 处理（RAG + LLM）
"""
import logging
import asyncio
import json
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class VoiceWebSocketHandler:
    """WebSocket 语音交互处理器（文本传输）"""

    def __init__(self, websocket: WebSocket, session_manager, stt_service, tts_service):
        self.websocket = websocket
        self.session_manager = session_manager
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.session_id: Optional[str] = None
        self.session = None
        self.is_active = False

    async def connect(self):
        """建立连接"""
        await self.websocket.accept()
        self.is_active = True
        logger.info("WebSocket连接已建立")

        # 创建新会话
        self.session_id = self.session_manager.create_session()
        self.session = self.session_manager.get_session(self.session_id)
        logger.info(f"创建会话: {self.session_id}")

        # 保存事件循环引用
        self.loop = asyncio.get_event_loop()

        # 设置VoiceAssistant回调（使用run_coroutine_threadsafe从线程调度到事件循环）
        def safe_callback(coro):
            """安全地从线程调度协程到事件循环"""
            asyncio.run_coroutine_threadsafe(coro, self.loop)

        self.session.assistant.on_user_speech = lambda text: safe_callback(
            self.send_json({"type": "user_message", "text": text})
        )
        self.session.assistant.on_assistant_chunk = lambda chunk: safe_callback(
            self.send_json({"type": "assistant_chunk", "chunk": chunk})
        )
        self.session.assistant.on_assistant_response = lambda text: safe_callback(
            self.send_json({"type": "assistant_message", "text": text, "rag_used": False})
        )
        self.session.assistant.on_error = lambda error: safe_callback(
            self.send_json({"type": "error", "message": error})
        )

        # 发送会话信息
        await self.send_json({
            "type": "session_info",
            "session_id": self.session_id,
            "message": "连接成功"
        })

    async def disconnect(self):
        """断开连接"""
        self.is_active = False
        if self.session_id:
            self.session_manager.delete_session(self.session_id)
            logger.info(f"清理会话: {self.session_id}")

    async def handle_messages(self):
        """处理客户端消息"""
        try:
            while self.is_active:
                # 接收消息
                data = await self.websocket.receive_text()
                message = json.loads(data)

                msg_type = message.get("type")

                if msg_type == "text":
                    # 文本消息（STT识别后的文本）
                    await self.handle_text_message(message.get("data", ""))

                elif msg_type == "ping":
                    # 心跳
                    await self.send_json({"type": "pong"})

                else:
                    logger.warning(f"未知消息类型: {msg_type}")

        except WebSocketDisconnect:
            logger.info("WebSocket连接已断开")
            await self.disconnect()
        except Exception as e:
            logger.error(f"处理消息失败: {str(e)}")
            await self.send_json({
                "type": "error",
                "message": str(e)
            })

    async def handle_text_message(self, text: str):
        """
        处理文本消息（STT识别后）

        流程：
        1. 调用 VoiceAssistant.process_text_input()
        2. VoiceAssistant通过回调发送响应
        """
        if not text.strip():
            return

        logger.info(f"收到文本: {text[:50]}...")

        try:
            # 调用 VoiceAssistant 处理（在线程池中执行，避免阻塞）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.session.assistant.process_text_input,
                text
            )

            logger.info("文本处理完成")

        except Exception as e:
            logger.error(f"处理文本失败: {str(e)}")
            await self.send_json({
                "type": "error",
                "message": f"处理失败: {str(e)}"
            })

    async def send_json(self, data: dict):
        """发送JSON消息"""
        try:
            await self.websocket.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
