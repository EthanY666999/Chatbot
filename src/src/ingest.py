"""
将本地文件（data/）切块 -> 调用 OpenAI Embeddings -> 写入 Chroma 持久化向量库。
运行示例：
    python src/ingest.py --source data --persist .chroma --collection docs
"""

import os
import glob
import json
import argparse
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

# ---- OpenAI Embeddings（直接用官方 SDK，不依赖 langchain）----
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("未找到 OPENAI_API_KEY。请在项目根目录 .env 写入：OPENAI_API_KEY=sk-xxxx")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ---- Chroma 持久化客户端 ----
import chromadb
from chromadb.config import Settings


# ========== I/O ==========

SUPPORTED_EXTS = {".txt", ".md", ".mdx", ".csv", ".json"}

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_file(path: str) -> str:
    """可选：安装 pypdf 后启用 PDF 读取。"""
    try:
        from pypdf import PdfReader  # pip install pypdf
    except Exception as e:
        raise RuntimeError("读取 PDF 需要安装 pypdf：pip install pypdf") from e
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)

def load_raw_documents(source_dir: str, pattern: str) -> List[Tuple[str, str]]:
    """
    返回 [(path, text), ...]
    """
    files = glob.glob(os.path.join(source_dir, pattern), recursive=True)
    result = []
    for fp in files:
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext == ".pdf":
                text = read_pdf_file(fp)
            elif ext in SUPPORTED_EXTS:
                text = read_text_file(fp)
            else:
                # 不支持的后缀直接跳过
                continue
            text = text.strip()
            if text:
                result.append((fp, text))
        except Exception as e:
            print(f"⚠️ 读取失败：{fp} -> {e}")
    return result


# ========== 文本切块 ==========

def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    简单按字符切块，带重叠。生产环境可替换为更聪明的分句切块。
    """
    text = " ".join(text.split())  # 压缩多空白
    if chunk_size <= 0:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return [c for c in chunks if c.strip()]


# ========== Embeddings ==========

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    调用 OpenAI embeddings。一次最多输入 2048 条左右（官方限制会变化；这里做分批更稳）。
    """
    out: List[List[float]] = []
    BATCH = 256
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


# ========== 写入 Chroma ==========

def get_chroma_collection(persist_dir: str, collection_name: str):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(allow_reset=False)
    )
    coll = client.get_or_create_collection(collection_name)
    return coll

def upsert_documents(
    coll,
    docs: List[Tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int
):
    """
    将文档切块后写入/更新到 Chroma。
    id 规则： {path}::{idx}
    metadata: {"source": path, "chunk": idx}
    """
    all_ids: List[str] = []
    all_docs: List[str] = []
    all_metas: List[Dict] = []

    for path, text in docs:
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, ck in enumerate(chunks):
            all_ids.append(f"{path}::{idx}")
            all_docs.append(ck)
            all_metas.append({"source": path, "chunk": idx})

    if not all_docs:
        print("ℹ️ 没有可写入的文档块。")
        return 0

    print(f"🧩 共 {len(all_docs)} 个文本块，开始生成向量（模型：{EMBED_MODEL}）...")
    vectors = embed_batch(all_docs)
    print("✅ 向量生成完成，写入 Chroma ...")

    # Chroma 会根据 id 去重/更新
    coll.upsert(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas,
        embeddings=vectors
    )
    print("🎉 写入完成。")
    return len(all_docs)


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(description="将本地文件向量化并写入 Chroma")
    parser.add_argument("--source", default="data", help="原始文档目录（默认 data）")
    parser.add_argument(
        "--pattern",
        default="**/*.*",
        help="文件匹配模式（glob），默认 '**/*.*'；支持 .txt .md .mdx .csv .json .pdf"
    )
    parser.add_argument("--persist", default=".chroma", help="Chroma 持久化目录（默认 .chroma）")
    parser.add_argument("--collection", default="docs", help="集合名称（默认 docs）")
    parser.add_argument("--chunk-size", type=int, default=800, help="切块大小（默认 800）")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="切块重叠（默认 100）")
    args = parser.parse_args()

    print(f"📂 扫描目录：{args.source}（模式：{args.pattern}）")
    raw_docs = load_raw_documents(args.source, args.pattern)
    print(f"📝 读取到 {len(raw_docs)} 个文件。")

    coll = get_chroma_collection(args.persist, args.collection)
    n = upsert_documents(coll, raw_docs, args.chunk_size, args.chunk_overlap)

    # 记录一次 ingest 信息
    meta = {
        "source": os.path.abspath(args.source),
        "pattern": args.pattern,
        "collection": args.collection,
        "persist": os.path.abspath(args.persist),
        "chunks_written": n,
        "embed_model": EMBED_MODEL,
    }
    os.makedirs(args.persist, exist_ok=True)
    with open(os.path.join(args.persist, "ingest_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"🧾 元数据已写入：{args.persist}/ingest_meta.json")


if __name__ == "__main__":
    main()