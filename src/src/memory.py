# src/memory.py
"""
长期记忆（RAG） + 短期会话记忆

- VectorMemory：基于 Chroma 的向量库
    - add_memories(texts, metadatas=None) -> List[str]
    - query(query_text, k=5) -> List[Dict]
    - count() -> int
    - reset() -> None

- ChatMemory：简易会话缓冲（只存最近 N 轮）
    - add(role, content) -> None
    - get() -> List[Dict]
"""

import time
import uuid
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings

from .embeddings import Embedder
from .config import VECTOR_DB_PATH


class VectorMemory:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        persist_dir: str = VECTOR_DB_PATH,
        collection: str = "long_term",
    ):
        # 关闭匿名遥测，指定持久化路径
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},  # 余弦距离（越小越相似）
        )
        self.embedder = embedder or Embedder()

    # ---------- 写入 ----------
    def add_memories(
        self, texts: List[str], metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """写入多条文本到向量库；返回生成的 ids"""
        texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
        if not texts:
            return []

        embs = self.embedder.embed(texts)
        ids = [
            f"m-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}-{i}"
            for i in range(len(texts))
        ]
        metas = metadatas or [{} for _ in texts]

        # Chroma 要求 metadatas/documents/embeddings/ids 等长
        if len(metas) != len(texts):
            if len(metas) < len(texts):
                metas = metas + [{} for _ in range(len(texts) - len(metas))]
            else:
                metas = metas[: len(texts)]

        self.col.add(
            ids=ids,
            documents=texts,
            embeddings=embs,
            metadatas=metas,
        )
        return ids

    # ---------- 检索 ----------
    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        语义检索：返回按相似度降序排列的结果
        字段：
          - id: 文档ID
          - text: 文档内容
          - meta: 元信息
          - score: 相似度（0~1，越大越相似）
        """
        if not query_text or self.count() == 0:
            return []

        q_emb = self.embedder.embed_one(query_text)
        res = self.col.query(query_embeddings=[q_emb], n_results=max(1, k))

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0] 

        out = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(
                {
                    "id": _id,
                    "text": doc,
                    "meta": meta,
                    "score": float(1.0 - dist),  
                }
            )

        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    # ---------- 维护 ----------
    def count(self) -> int:
        peek = self.col.peek() or {}
        return len(peek.get("ids") or [])

    def reset(self):
        """清空集合（不可逆）"""
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )


class ChatMemory:
    """
    简单短期记忆：保存最近 N 轮对话（user/assistant 各算一条）
    用于 main.py 里的 memory.add(...) / memory.get()
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.turns: List[Dict] = []

    def add(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})
        
        if len(self.turns) > self.max_turns * 2:
            self.turns = self.turns[-self.max_turns * 2:]

    # 兼容别名
    append = add

    def get(self) -> List[Dict]:
        return self.turns[:]