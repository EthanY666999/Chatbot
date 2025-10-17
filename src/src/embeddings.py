"""
Embedding 封装：
- 统一用 OpenAI 的 embedding 接口
- 提供批量与单条向量化
"""

from typing import List, Sequence
from openai import OpenAI
from .config import OPENAI_API_KEY, EMBED_MODEL

# 初始化全局客户端（避免重复创建）
_client = OpenAI(api_key=OPENAI_API_KEY)


class Embedder:
    """
    用法：
        embedder = Embedder()
        vecs = embedder.embed(["你好", "这是第二段文本"])
        vec  = embedder.embed_one("只向量化一条文本")
    """

    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        self.client = _client

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """
        批量向量化：返回与输入顺序一致的向量列表（List[List[float]]）
        """
        # OpenAI 新版 SDK: embeddings.create
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in resp.data]

    def embed_one(self, text: str) -> List[float]:
        """
        单条向量化：返回一个向量（List[float]）
        """
        return self.embed([text])[0]
    
