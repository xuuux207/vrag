"""
配置管理模块
从.env文件加载所有配置项
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


@dataclass
class AzureSpeechConfig:
    """Azure Speech Service配置"""
    key: str
    region: str
    endpoint: str
    language: str = "zh-CN"
    segmentation_silence_timeout_ms: int = 1000  # VAD静音阈值


@dataclass
class VADConfig:
    """VAD配置"""
    type: str = "azure"  # azure 或 silero
    sample_rate: int = 16000  # Silero VAD采样率
    threshold: float = 0.5  # Silero VAD阈值
    min_speech_duration: float = 0.3  # 最小语音时长（秒）
    min_silence_duration: float = 1.0  # 最小静音时长（秒，停顿多久算结束）


@dataclass
class AzureSearchConfig:
    """Azure AI Search配置"""
    key: str
    endpoint: str
    index_name: str = "customer-service-kb"


@dataclass
class QwenConfig:
    """Qwen LLM配置"""
    api_base: str
    model: str
    token: str
    temperature: float = 0.7


@dataclass
class EmbeddingConfig:
    """Embedding服务配置"""
    model: str
    url: str
    token: str


@dataclass
class RerankingConfig:
    """Reranking服务配置"""
    model: str
    url: str
    token: str


@dataclass
class ContextConfig:
    """上下文管理配置"""
    compression_threshold: int = 4000  # token数阈值
    keep_recent_turns: int = 2  # 保留最近n轮全文
    compression_model: str = "qwen3-8b"


@dataclass
class RAGConfig:
    """RAG检索配置"""
    top_k: int = 20  # 混合检索召回数
    rerank_top_k: int = 5  # Rerank后返回数
    qa_weight_boost: float = 1.5  # QA类型加权
    qa_direct_threshold: float = 0.85  # 直接返回阈值


class Settings:
    """全局配置管理"""

    def __init__(self):
        # Azure Speech（支持旧键名SPEECH_*）
        self.azure_speech = AzureSpeechConfig(
            key=self._get_env_compat("AZURE_SPEECH_KEY", "SPEECH_KEY"),
            region=self._get_env_compat("AZURE_SPEECH_REGION", "SPEECH_REGION"),
            endpoint=self._get_env_compat("AZURE_SPEECH_ENDPOINT", "SPEECH_ENDPOINT"),
        )

        # VAD配置
        self.vad = VADConfig(
            type=os.getenv("VAD_TYPE", "azure").lower(),
            threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
            min_speech_duration=float(os.getenv("VAD_MIN_SPEECH_DURATION", "0.3")),
            min_silence_duration=float(os.getenv("VAD_MIN_SILENCE_DURATION", "1.0")),
        )

        # Azure AI Search（支持旧键名AI_SEARCH_*）
        self.azure_search = AzureSearchConfig(
            key=self._get_env_compat("AZURE_SEARCH_KEY", "AI_SEARCH_KEY"),
            endpoint=self._get_env_compat("AZURE_SEARCH_ENDPOINT", "AI_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "customer-service-kb"),
        )

        # Qwen LLM
        self.qwen = QwenConfig(
            api_base=self._get_env("QWEN_API_BASE"),
            model=self._get_env("QWEN_MODEL"),
            token=self._get_env("QWEN_TOKEN"),
        )

        # Embedding
        self.embedding = EmbeddingConfig(
            model=self._get_env("EMBEDDING_MODEL"),
            url=self._get_env("EMBEDDING_URL"),
            token=self._get_env("EMBEDDING_TOKEN"),
        )

        # Reranking
        self.reranking = RerankingConfig(
            model=self._get_env("RERANKING_MODEL"),
            url=self._get_env("RERANKING_URL"),
            token=self._get_env("RERANKING_TOKEN"),
        )

        # 上下文管理
        self.context = ContextConfig()

        # RAG配置
        self.rag = RAGConfig()

    def _get_env(self, key: str, default: Optional[str] = None) -> str:
        """获取环境变量，若不存在则抛出异常"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"环境变量 {key} 未设置，请检查 .env 文件")
        return value

    def _get_env_compat(self, new_key: str, old_key: str, default: Optional[str] = None) -> str:
        """获取环境变量（向后兼容旧键名）"""
        value = os.getenv(new_key) or os.getenv(old_key, default)
        if value is None:
            raise ValueError(f"环境变量 {new_key} 或 {old_key} 未设置，请检查 .env 文件")
        return value


# 全局配置实例
settings = Settings()
