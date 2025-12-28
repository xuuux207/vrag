"""
会话管理器
管理多个用户会话，支持并发访问
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional
from threading import Lock

from src.pipeline.voice_assistant import VoiceAssistant

logger = logging.getLogger(__name__)


class Session:
    """单个会话"""

    def __init__(self, session_id: str, assistant: VoiceAssistant):
        self.session_id = session_id
        self.assistant = assistant
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.lock = Lock()  # 保护单个会话的并发访问

    def update_activity(self):
        """更新最后活动时间"""
        self.last_activity = datetime.now()


class SessionManager:
    """会话管理器（线程安全）"""

    def __init__(
        self,
        stt_service,
        tts_service,
        rag_searcher,
        llm_service,
        context_manager,
        system_prompt: Optional[str] = None,
        max_sessions: int = 100,
    ):
        """
        初始化会话管理器

        Args:
            stt_service: STT服务
            tts_service: TTS服务
            rag_searcher: RAG检索器
            llm_service: LLM服务
            context_manager: 上下文管理器
            system_prompt: 系统提示词
            max_sessions: 最大会话数
        """
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.rag_searcher = rag_searcher
        self.llm_service = llm_service
        self.context_manager = context_manager
        self.system_prompt = system_prompt
        self.max_sessions = max_sessions

        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()  # 保护_sessions字典的并发访问

    def create_session(self) -> str:
        """
        创建新会话

        Returns:
            session_id
        """
        with self._lock:
            # 检查会话数限制
            if len(self._sessions) >= self.max_sessions:
                # 清理最旧的会话
                self._cleanup_oldest_session()

            # 创建新会话
            session_id = str(uuid.uuid4())
            assistant = VoiceAssistant(
                stt_service=self.stt_service,
                tts_service=self.tts_service,
                rag_searcher=self.rag_searcher,
                llm_service=self.llm_service,
                context_manager=self.context_manager,
                system_prompt=self.system_prompt,
                enable_tts_playback=False,  # Web模式：不播放，通过回调发送
            )
            session = Session(session_id, assistant)
            self._sessions[session_id] = session

            logger.info(f"创建会话: {session_id}, 当前会话数: {len(self._sessions)}")
            return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        Args:
            session_id: 会话ID

        Returns:
            Session对象，不存在则返回None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.update_activity()
            return session

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            是否删除成功
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"删除会话: {session_id}, 当前会话数: {len(self._sessions)}")
                return True
            return False

    def get_session_count(self) -> int:
        """获取当前会话数"""
        with self._lock:
            return len(self._sessions)

    def _cleanup_oldest_session(self):
        """清理最旧的会话（需要在_lock保护下调用）"""
        if not self._sessions:
            return

        oldest_id = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid].last_activity
        )
        del self._sessions[oldest_id]
        logger.info(f"清理最旧会话: {oldest_id}")
