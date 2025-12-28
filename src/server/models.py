"""
API数据模型
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: Optional[str] = Field(None, description="会话ID，不提供则创建新会话")
    message: str = Field(..., description="用户消息", min_length=1)


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="助手回复")
    rag_used: bool = Field(..., description="是否使用了RAG检索")
    rag_docs_count: int = Field(0, description="检索到的文档数")


class MessageHistory(BaseModel):
    """消息历史记录"""
    role: str = Field(..., description="角色: user/assistant")
    content: str = Field(..., description="消息内容")


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str = Field(..., description="会话ID")
    message_count: int = Field(..., description="消息数量")
    created_at: str = Field(..., description="创建时间")
    last_activity: str = Field(..., description="最后活动时间")


class HistoryResponse(BaseModel):
    """历史记录响应"""
    session_id: str
    messages: List[MessageHistory]
    total_count: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    active_sessions: int = 0
