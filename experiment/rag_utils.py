"""
RAG 实用工具库

使用在线 Embedding（BAAI/bge-m3）和 Reranking（BAAI/bge-reranker-v2-m3）
完整的 Embedding → Indexing → Retrieval → Reranking 流程
"""

import os
import re
import time
import json
import pickle
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============================================================================
# 第 1 部分：Embedding 服务
# ============================================================================

class EmbeddingService:
    """在线 Embedding 服务（使用 BAAI/bge-m3）"""
    
    def __init__(self):
        self.model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.url = os.getenv("EMBEDDING_URL", "https://api.siliconflow.cn/v1/embeddings")
        self.token = os.getenv("EMBEDDING_TOKEN")
        
        if not self.token:
            raise ValueError("❌ 缺少 EMBEDDING_TOKEN，请在 .env 中配置")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.dimension = 768  # BGE-M3 输出维度
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文本
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表，每个向量是 768 维
        
        示例：
            >>> service = EmbeddingService()
            >>> embeddings = service.embed_texts(["文本1", "文本2"])
            >>> len(embeddings)  # 2
            >>> len(embeddings[0])  # 768
        """
        if not texts:
            return []
        
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 提取向量
            embeddings = []
            for item in result.get("data", []):
                embeddings.append(item["embedding"])
            
            if len(embeddings) != len(texts):
                raise ValueError(f"返回的向量数 ({len(embeddings)}) != 输入文本数 ({len(texts)})")
            
            return embeddings
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"❌ Embedding API 调用失败: {e}")
    
    def embed_single(self, text: str) -> List[float]:
        """嵌入单个文本"""
        return self.embed_texts([text])[0]


# ============================================================================
# 第 2 部分：向量索引
# ============================================================================

class VectorIndex:
    """
    向量索引（支持本地持久化）
    
    存储方式：
    - 内存：Python dict（快速访问）
    - 持久化：pickle 文件（保存/加载）
    
    功能：
    - 文档向量存储
    - 文档元信息管理
    - 保存到本地文件
    - 从本地文件加载
    - 增量更新（避免重复嵌入）
    """
    
    def __init__(self, embedding_service: EmbeddingService, cache_dir: str = "./vector_cache"):
        self.embedding_service = embedding_service
        self.documents = {}  # doc_id -> {title, content}
        self.vectors = {}    # doc_id -> np.array
        self.dimension = embedding_service.dimension
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def add_documents(self, documents: List[Dict]) -> None:
        """
        添加文档并生成向量
        
        Args:
            documents: [
                {
                    "id": "doc_1",
                    "title": "标题",
                    "content": "内容"
                },
                ...
            ]
        
        返回：None（修改内部状态）
        """
        if not documents:
            print("⚠️  没有文档要添加")
            return
        
        # 提取内容用于嵌入
        doc_contents = []
        doc_ids = []
        
        for doc in documents:
            doc_ids.append(doc["id"])
            # 组合标题和内容作为嵌入对象
            title = doc.get("title", "")
            content = doc.get("content", "")
            combined = f"{title} {content}".strip()
            doc_contents.append(combined)
        
        print(f"📊 正在嵌入 {len(doc_contents)} 个文档...")
        start_time = time.time()
        
        # 批量调用 API
        embeddings = self.embedding_service.embed_texts(doc_contents)
        
        elapsed = time.time() - start_time
        print(f"✓ Embedding 完成 (耗时 {elapsed:.2f}s)")
        
        # 存储
        for doc_id, doc, embedding in zip(doc_ids, documents, embeddings):
            self.documents[doc_id] = {
                "title": doc.get("title", ""),
                "content": doc.get("content", "")
            }
            self.vectors[doc_id] = np.array(embedding)
        
        print(f"✓ 成功索引 {len(self.vectors)} 个文档")
    
    def add_document_incremental(self, doc_id: str, title: str, content: str) -> bool:
        """
        增量添加单个文档（如果已存在则跳过）
        
        Args:
            doc_id: 文档 ID
            title: 文档标题
            content: 文档内容
        
        Returns:
            True: 新增成功，False: 已存在跳过
        """
        if doc_id in self.documents:
            print(f"⏭️  文档 {doc_id} 已存在，跳过")
            return False
        
        # 嵌入单个文档
        combined = f"{title} {content}".strip()
        embedding = self.embedding_service.embed_single(combined)
        
        # 存储
        self.documents[doc_id] = {"title": title, "content": content}
        self.vectors[doc_id] = np.array(embedding)
        
        print(f"✓ 新增文档 {doc_id}")
        return True
    
    def save(self, filename: str = "vector_index.pkl") -> None:
        """
        保存索引到本地文件
        
        Args:
            filename: 保存的文件名
        """
        filepath = self.cache_dir / filename
        
        data = {
            "documents": self.documents,
            "vectors": {doc_id: vec.tolist() for doc_id, vec in self.vectors.items()},
            "dimension": self.dimension
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        print(f"💾 索引已保存到 {filepath} ({len(self.documents)} 个文档)")
    
    def load(self, filename: str = "vector_index.pkl") -> bool:
        """
        从本地文件加载索引
        
        Args:
            filename: 加载的文件名
        
        Returns:
            True: 加载成功，False: 文件不存在
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  索引文件 {filepath} 不存在")
            return False
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.documents = data["documents"]
        self.vectors = {doc_id: np.array(vec) for doc_id, vec in data["vectors"].items()}
        self.dimension = data["dimension"]
        
        print(f"📂 索引已加载 {filepath} ({len(self.documents)} 个文档)")
        return True
    
    def clear(self) -> None:
        """清空索引"""
        self.documents.clear()
        self.vectors.clear()
        print("🗑️  索引已清空")
    
    def get_vector(self, doc_id: str) -> Optional[np.ndarray]:
        """获取文档向量"""
        return self.vectors.get(doc_id)
    
    def get_all_vectors(self) -> Tuple[List[str], np.ndarray]:
        """
        获取所有向量
        
        Returns:
            (doc_ids, vectors_matrix)
            vectors_matrix 的形状为 (num_docs, 768)
        """
        doc_ids = list(self.vectors.keys())
        vectors = np.array([self.vectors[doc_id] for doc_id in doc_ids])
        return doc_ids, vectors
    
    def size(self) -> int:
        """索引中的文档数"""
        return len(self.vectors)


# ============================================================================
# 第 3 部分：检索函数
# ============================================================================

def retrieve_by_similarity(query: str,
                          embedding_service: EmbeddingService,
                          index: VectorIndex,
                          top_k: int = 10) -> List[Dict]:
    """
    相似度检索
    
    Args:
        query: 查询文本
        embedding_service: EmbeddingService 实例
        index: VectorIndex 实例
        top_k: 返回前 k 个结果
    
    Returns:
        [
            {
                "doc_id": "...",
                "similarity": 0.856,
                "title": "...",
                "content": "..."
            },
            ...
        ]
    """
    # 嵌入查询
    query_embedding = embedding_service.embed_single(query)
    query_vector = np.array(query_embedding)
    
    # 获取所有向量
    doc_ids, all_vectors = index.get_all_vectors()
    
    if len(doc_ids) == 0:
        print("⚠️  索引中没有文档")
        return []
    
    # 计算相似度（余弦相似度）
    similarities = []
    
    for doc_id, doc_vector in zip(doc_ids, all_vectors):
        # 余弦相似度 = dot(a, b) / (norm(a) * norm(b))
        # 因为向量已通常被 embedding 模型归一化，所以可以直接使用 dot 作为相似度
        similarity = np.dot(query_vector, doc_vector)
        
        similarities.append({
            "doc_id": doc_id,
            "similarity": float(similarity),
            "title": index.documents[doc_id]["title"],
            "content": index.documents[doc_id]["content"]
        })
    
    # 排序并返回 Top-K
    similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# ============================================================================
# 第 4 部分：Reranking 服务
# ============================================================================

class RerankingService:
    """在线 Reranking 服务（使用 BAAI/bge-reranker-v2-m3）"""
    
    def __init__(self):
        self.model = os.getenv("RERANKING_MODEL", "BAAI/bge-reranker-v2-m3")
        self.url = os.getenv("RERANKING_URL", "https://api.siliconflow.cn/v1/rerankings")
        self.token = os.getenv("RERANKING_TOKEN")
        
        if not self.token:
            raise ValueError("❌ 缺少 RERANKING_TOKEN，请在 .env 中配置")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def rerank(self, query: str, passages: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Reranking 精排
        
        Args:
            query: 查询文本
            passages: [{"doc_id": "...", "title": "...", "content": "...", "similarity": 0.8}, ...]
            top_k: 返回前 k 个结果
        
        Returns:
            重排后的 passages（同样的结构，但加上 rerank_score）
        """
        if not passages:
            return []
        
        # 提取内容列表
        contents = [p["content"] for p in passages]
        
        payload = {
            "model": self.model,
            "query": query,
            "passages": contents,
            "top_n": min(top_k, len(passages))
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # 提取重排后的结果
            reranked = []
            for item in result.get("results", []):
                idx = item["index"]
                score = item["score"]
                
                reranked.append({
                    **passages[idx],
                    "rerank_score": float(score)
                })
            
            return reranked[:top_k]
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Reranking 失败: {e}，使用原始顺序")
            return passages[:top_k]


# ============================================================================
# 第 5 部分：完整 RAG 流程
# ============================================================================

def rag_retrieve_and_rerank(query: str,
                           embedding_service: EmbeddingService,
                           reranking_service: RerankingService,
                           index: VectorIndex,
                           retrieval_top_k: int = 10,
                           rerank_top_k: int = 3,
                           verbose: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    完整 RAG 流程：Embedding → Retrieval → Reranking
    
    Args:
        query: 查询文本
        embedding_service: EmbeddingService 实例
        reranking_service: RerankingService 实例
        index: VectorIndex 实例
        retrieval_top_k: 检索阶段返回多少个候选
        rerank_top_k: Reranking 后返回多少个结果
        verbose: 是否打印详细信息
    
    Returns:
        (final_results, retrieval_results)
        - final_results: 最终的 reranked 结果
        - retrieval_results: 原始检索结果（用于对比）
    """
    if verbose:
        print("\n" + "="*70)
        print(f"【RAG 流程】查询: {query[:60]}")
        print("="*70)
    
    start_time = time.time()
    
    # 第 1 步：相似度检索
    if verbose:
        print("\n[步骤 1] 相似度检索...")
    
    retrieval_results = retrieve_by_similarity(
        query, embedding_service, index, top_k=retrieval_top_k
    )
    
    if verbose:
        print(f"✓ 检索到 {len(retrieval_results)} 个候选文档")
        for i, r in enumerate(retrieval_results[:3], 1):
            print(f"  {i}. {r['title']} (相似度: {r['similarity']:.3f})")
    
    # 第 2 步：Reranking 精排
    if verbose:
        print("\n[步骤 2] Reranking 精排序...")
    
    final_results = reranking_service.rerank(
        query, retrieval_results, top_k=rerank_top_k
    )
    
    if verbose:
        print(f"✓ 精排后 Top {len(final_results)} 个结果：")
        for i, r in enumerate(final_results, 1):
            score = r.get("rerank_score", r.get("similarity"))
            print(f"  {i}. {r['title']} (分数: {score:.3f})")
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\n⏱️  总耗时: {elapsed:.2f}s")
    
    return final_results, retrieval_results


def build_rag_context(rag_results: List[Dict]) -> str:
    """
    从 RAG 结果组织背景知识上下文
    
    Args:
        rag_results: RAG 返回的结果列表
    
    Returns:
        格式化的上下文字符串，可直接用于 LLM 提示词
    """
    if not rag_results:
        return ""
    
    context = "【检索到的背景知识】\n\n"
    
    for i, result in enumerate(rag_results, 1):
        context += f"{i}. {result['title']}\n"
        context += f"   {result['content'][:300]}...\n\n"
    
    return context


# ============================================================================
# 工具函数
# ============================================================================

def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    简单关键词提取
    
    Args:
        text: 输入文本
        num_keywords: 返回关键词数量
    
    Returns:
        关键词列表
    """
    # 简单实现：按长度过滤
    words = re.findall(r'[\w\u4e00-\u9fff]+', text)
    stopwords = {
        '的', '是', '在', '了', '和', '与', '或', '等', '我', '们', '您',
        '有', '能', '可以', '进行', '实现', '为', '被', '来', '到', '从'
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    return keywords[:num_keywords]


if __name__ == "__main__":
    print("✓ RAG 工具库加载成功")
    print("  - EmbeddingService")
    print("  - VectorIndex")
    print("  - RerankingService")
    print("  - 检索和 Reranking 函数")
