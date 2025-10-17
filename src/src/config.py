# src/config.py
"""
项目配置文件：负责加载环境变量。
"""

import os
from dotenv import load_dotenv

# 从项目根目录的 .env 文件加载环境变量
load_dotenv()

# 读取 OpenAI 的 API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 检查是否设置了密钥
if not OPENAI_API_KEY:
    raise ValueError("❌ 未检测到 OPENAI_API_KEY，请在项目根目录创建 .env 文件并设置。")

# 你也可以在这里集中定义其他常量：
EMBED_MODEL = "text-embedding-3-small"   # 向量化模型
CHAT_MODEL = "gpt-4o-mini"               # 聊天模型
VECTOR_DB_PATH = ".chroma"               # 向量库存储路径

# 方便打印确认（调试时用）
if __name__ == "__main__":
    print("✅ Config 加载成功")
    print("OPENAI_API_KEY 前5位:", OPENAI_API_KEY[:5], "...")
    print("向量库目录:", VECTOR_DB_PATH)

