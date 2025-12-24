"""
渐进式总结模块 V3 - 增量RAG版本
核心改进：
1. 总结输入 = 之前总结 + 当前段全文（保证信息不丢失）
2. 每次输入都进行RAG检索（增量检索）
3. 过滤低相关度文档（只保留有效结果）
"""

import json
import time
import numpy as np
from typing import Dict, List, Set
from openai import OpenAI


# 总结prompt - 输入包含总结+全文
INCREMENTAL_SUMMARY_PROMPT = """你是一个语音助手，正在实时总结用户的语音输入。

【之前的总结】：
{previous_summary}

【当前段落（完整内容）】：
{current_segment}

请生成一个累积式的总结，要求：
1. **保留之前总结中的所有关键信息**（不要丢失）
2. **整合当前段落的重要信息**
3. **过滤口语词、寒暄等噪音**
4. **合并重复信息**
5. **保持简洁但完整**

只返回一行简洁的总结文本，不要JSON格式，不要markdown："""


class IncrementalRAGSummarizer:
    """增量RAG总结器 - 边总结边检索"""

    def __init__(
        self,
        llm_client: OpenAI,
        embedding_service,
        vector_index,
        model_name: str = "qwen3-8b",
        relevance_threshold: float = 0.6
    ):
        self.llm = llm_client
        self.embedding = embedding_service
        self.vector_index = vector_index
        self.model_name = model_name
        self.relevance_threshold = relevance_threshold

        # 状态
        self.current_summary = ""
        self.segment_count = 0
        self.total_length = 0

        # RAG相关
        self.retrieved_doc_ids: Set[str] = set()  # 已检索的文档ID（去重）
        self.relevant_docs: List[Dict] = []       # 相关文档列表

        # 性能统计
        self.segment_times = []
        self.rag_times = []

    def reset(self):
        """重置状态"""
        self.current_summary = ""
        self.segment_count = 0
        self.total_length = 0
        self.retrieved_doc_ids.clear()
        self.relevant_docs.clear()
        self.segment_times = []
        self.rag_times = []

    def _calculate_relevance(
        self,
        doc_embedding: np.ndarray,
        query_embedding: np.ndarray
    ) -> float:
        """计算文档与查询的相关度（余弦相似度）"""
        dot_product = np.dot(doc_embedding, query_embedding)
        norm_doc = np.linalg.norm(doc_embedding)
        norm_query = np.linalg.norm(query_embedding)

        if norm_doc == 0 or norm_query == 0:
            return 0.0

        return dot_product / (norm_doc * norm_query)

    def _rag_search(self, query_text: str, segment_text: str, top_k: int = 5) -> List[Dict]:
        """
        RAG检索并过滤

        Args:
            query_text: 查询文本（总结）
            segment_text: 当前段落文本（用于相关度判断）
            top_k: 检索数量

        Returns:
            过滤后的相关文档列表
        """
        rag_start = time.time()

        # 1. 向量检索
        query_vector = self.embedding.embed_single(query_text)
        results = self.vector_index.search(query_vector, top_k=top_k)

        # 2. 获取当前段落的embedding（用于相关度计算）
        segment_vector = self.embedding.embed_single(segment_text)

        # 3. 过滤：去重 + 相关度判断
        new_relevant_docs = []
        for doc in results:
            doc_id = doc.get("doc_id")

            # 去重
            if doc_id in self.retrieved_doc_ids:
                continue

            # 计算与当前段落的相关度
            doc_content = doc.get("content", "")
            doc_vector = self.embedding.embed_single(doc_content[:500])
            relevance = self._calculate_relevance(doc_vector, segment_vector)

            # 相关度过滤
            if relevance >= self.relevance_threshold:
                self.retrieved_doc_ids.add(doc_id)
                new_relevant_docs.append({
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "content": doc_content,
                    "relevance": float(relevance),
                    "segment": self.segment_count
                })

        rag_time = time.time() - rag_start
        self.rag_times.append(rag_time)

        return new_relevant_docs

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

        # 1. 总结：输入 = 之前的总结 + 当前段全文
        if self.segment_count == 1:
            # 第一段，直接作为总结（去除口语词）
            self.current_summary = segment_text.strip()
        else:
            # 后续片段：总结 + 全文
            prompt = INCREMENTAL_SUMMARY_PROMPT.format(
                previous_summary=self.current_summary or "（无）",
                current_segment=segment_text
            )

            response = self.llm.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=False,
                extra_body={"enable_thinking": False}
            )

            self.current_summary = response.choices[0].message.content.strip()

        # 2. RAG检索：用总结作为query
        new_docs = self._rag_search(
            query_text=self.current_summary,
            segment_text=segment_text,
            top_k=5
        )

        # 3. 累积相关文档
        self.relevant_docs.extend(new_docs)

        seg_time = time.time() - seg_start
        self.segment_times.append(seg_time)

        return {
            "segment_number": self.segment_count,
            "summary": self.current_summary,
            "summary_length": len(self.current_summary),
            "new_docs_count": len(new_docs),
            "total_docs_count": len(self.relevant_docs),
            "processing_time": seg_time,
            "rag_time": self.rag_times[-1]
        }

    def get_final_summary(self) -> str:
        """返回最终总结"""
        return self.current_summary

    def get_relevant_docs(self) -> List[Dict]:
        """返回所有相关文档（按相关度排序）"""
        # 按相关度降序排序
        sorted_docs = sorted(
            self.relevant_docs,
            key=lambda x: x["relevance"],
            reverse=True
        )
        return sorted_docs

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "total_segments": self.segment_count,
            "total_input_length": self.total_length,
            "final_summary_length": len(self.current_summary),
            "compression_ratio": len(self.current_summary) / self.total_length if self.total_length > 0 else 0,
            "total_processing_time": sum(self.segment_times),
            "total_rag_time": sum(self.rag_times),
            "avg_segment_time": sum(self.segment_times) / len(self.segment_times) if self.segment_times else 0,
            "avg_rag_time": sum(self.rag_times) / len(self.rag_times) if self.rag_times else 0,
            "total_retrieved_docs": len(self.retrieved_doc_ids),
            "total_relevant_docs": len(self.relevant_docs)
        }
