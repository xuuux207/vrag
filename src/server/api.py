"""
FastAPI服务器
提供REST API接口
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config.settings import settings
from src.knowledge.rag_searcher import RAGSearcher
from src.llm.qwen_service import QwenService
from src.llm.context_manager import ContextManager
from src.server.models import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    MessageHistory,
    HealthResponse,
)
from src.server.session_manager import SessionManager
from src.server.websocket import VoiceWebSocketHandler
from rag_utils import EmbeddingService, RerankingService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="TechFlow AI 客服 API",
    description="基于RAG的智能客服系统",
    version="1.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量（应用启动时初始化）
session_manager: Optional[SessionManager] = None
stt_service = None
tts_service = None
SYSTEM_PROMPT = """
你是TechFlow的智能客服，用自然口语回答问题。

核心要求：
1. 说人话，别太书面，就像朋友聊天一样
2. 根据知识库回答，不知道就直说"这个我不太清楚"
3. 回答简短点，别啰嗦，直接说重点
4. 不要用emoji
5. 不要用markdown格式（不要用**、#、*、-等符号），只输出纯文本口语
6. 少用符号，尽量用中文化的自然表达
7. 语气自然、亲切，但保持专业
""".strip()


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global session_manager, stt_service, tts_service

    logger.info("=" * 70)
    logger.info("初始化 TechFlow AI 客服 API...")
    logger.info("=" * 70)

    try:
        # 初始化语音服务（用于 WebSocket 语音交互）
        logger.info("[1/6] 初始化语音服务...")
        from src.speech.tts_service import TTSService
        from src.speech.stt_service import STTService

        tts_service = TTSService(
            key=settings.azure_speech.key,
            region=settings.azure_speech.region,
            voice_name="zh-CN-XiaoxiaoNeural",
            rate=1.0,
        )
        stt_service = STTService(
            key=settings.azure_speech.key,
            region=settings.azure_speech.region,
            language="zh-CN",
        )
        logger.info("✓ TTS 服务已初始化")
        logger.info("✓ STT 服务已初始化")

        # 初始化RAG服务
        logger.info("[2/6] 初始化RAG检索服务...")
        embedding_service = EmbeddingService()
        reranking_service = RerankingService()

        rag_searcher = RAGSearcher(
            endpoint=settings.azure_search.endpoint,
            api_key=settings.azure_search.key,
            index_name=settings.azure_search.index_name,
            embedding_service=embedding_service,
            reranking_service=reranking_service,
        )

        # 初始化LLM服务
        logger.info("[3/6] 初始化LLM服务...")
        llm_service = QwenService(
            api_base=settings.qwen.api_base,
            model=settings.qwen.model,
            token=settings.qwen.token,
            temperature=settings.qwen.temperature,
        )

        # 初始化上下文管理器
        logger.info("[4/6] 初始化上下文管理器...")
        context_manager = ContextManager(
            llm_service=llm_service,
            token_threshold=settings.context.compression_threshold,
            keep_recent_turns=settings.context.keep_recent_turns,
        )

        # 初始化会话管理器
        logger.info("[5/6] 初始化会话管理器...")
        session_manager = SessionManager(
            stt_service=stt_service,
            tts_service=tts_service,
            rag_searcher=rag_searcher,
            llm_service=llm_service,
            context_manager=context_manager,
            system_prompt=SYSTEM_PROMPT,
            max_sessions=100,
        )

        logger.info("[6/6] 初始化完成")
        logger.info("=" * 70)
        logger.info("✓ API服务初始化完成！支持文本和语音交互")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.get_session_count() if session_manager else 0,
    )


@app.get("/api/speech-config")
async def get_speech_config():
    """
    获取 Azure Speech SDK 配置

    返回前端所需的 Speech Key 和 Region
    """
    return {
        "key": settings.azure_speech.key,
        "region": settings.azure_speech.region,
        "language": settings.azure_speech.language,
        "voice_name": "zh-CN-XiaoxiaoNeural"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    聊天接口

    处理用户消息，返回AI回复
    """
    if not session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪",
        )

    try:
        # 获取或创建会话
        if request.session_id:
            session = session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"会话不存在: {request.session_id}",
                )
            session_id = request.session_id
        else:
            session_id = session_manager.create_session()
            session = session_manager.get_session(session_id)

        # 处理消息（使用会话锁保护）
        with session.lock:
            response_text, metadata = session.assistant.process_text(request.message)

        # 返回响应
        return ChatResponse(
            session_id=session_id,
            message=response_text,
            rag_used=metadata["rag_used"],
            rag_docs_count=metadata["rag_docs_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"处理聊天请求失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理失败: {str(e)}",
        )


@app.get("/api/session/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """
    获取会话历史

    返回指定会话的所有消息记录
    """
    if not session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪",
        )

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {session_id}",
        )

    with session.lock:
        messages = session.assistant.get_history()

    return HistoryResponse(
        session_id=session_id,
        messages=[
            MessageHistory(role=msg["role"], content=msg["content"])
            for msg in messages
        ],
        total_count=len(messages),
    )


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话

    清理指定会话的所有数据
    """
    if not session_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="服务未就绪",
        )

    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {session_id}",
        )

    return {"message": "会话已删除", "session_id": session_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 实时语音交互

    客户端消息格式:
    - {"type": "text", "data": "文本内容"}
    - {"type": "audio", "data": "base64编码的音频"}
    - {"type": "ping"}

    服务端消息格式:
    - {"type": "session_info", "session_id": "...", "message": "..."}
    - {"type": "user_message", "text": "..."}
    - {"type": "assistant_message", "text": "...", "rag_used": bool}
    - {"type": "audio", "data": "base64编码的音频"}
    - {"type": "error", "message": "..."}
    """
    if not session_manager:
        await websocket.close(code=1011, reason="服务未就绪")
        return

    handler = VoiceWebSocketHandler(websocket, session_manager, stt_service, tts_service)

    try:
        await handler.connect()
        await handler.handle_messages()
    except WebSocketDisconnect:
        logger.info("客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {str(e)}")
    finally:
        await handler.disconnect()


# 挂载静态文件目录（前端页面）
web_dir = Path(__file__).parent.parent.parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
    logger.info(f"已挂载静态文件目录: {web_dir}")
