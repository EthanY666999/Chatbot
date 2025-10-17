# 🤖 Chatbot — A CLI AI Agent with Memory, RAG, and Chain-of-Thought

A lightweight command-line **AI Agent Chatbot** built from scratch using Python and OpenAI API.  
This project demonstrates how to integrate **short-term memory**, **long-term vector memory (RAG)**, and **Chain-of-Thought (CoT)** reasoning into an interactive assistant.

---

## 🧠 Features

### 1. 🔁 Conversational Memory
- **Short-term memory** (`ChatMemory`) keeps track of the recent N dialogue turns.  
- Messages are stored and reloaded into each model call.

### 2. 📚 Long-term Memory (RAG)
- **VectorMemory** uses [ChromaDB](https://docs.trychroma.com/) as a persistent vector store.  
- Supports:
  - Adding semantic memories (`add_memories`)
  - Querying similar past contexts (`query`)
- Embeddings are generated with `text-embedding-3-small`.

### 3. 🧩 Chain-of-Thought Reasoning (CoT)
- The chatbot performs **step-by-step reasoning** before producing answers.  
- CoT is implemented at the prompt level with an optional `cot` command.

### 4. ⚙️ Modular Design
- Easy to extend with more tools (search, file agents, etc.)
- Separated modules:
- src/
-├── config.py # environment configs and constants
-├── embeddings.py # embedding model wrapper
-├── memory.py # ChatMemory + VectorMemory (RAG)
-├── prompt.py # system prompts and reasoning templates
-├── main.py # CLI entry point
-└── ingest.py # data ingestion for RAG

Run the chatbot:
python -m src.src.main

You’ll see:
-🤖 Chatbot 已启动，输入 'exit' 退出。
-你:

License

MIT License © 2025 EthanY666999
Feel free to fork, modify, and build your own AI Agent!
