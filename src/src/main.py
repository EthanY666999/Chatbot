# src/main.py
from .config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DB_PATH
from .prompt import SYSTEM_PROMPT, DEVELOPER_HINT
from .memory import VectorMemory, ChatMemory
import re
from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DB_PATH
from .prompt import SYSTEM_PROMPT, DEVELOPER_HINT
from .memory import VectorMemory, ChatMemory

# 各集合
DOCS_COLLECTION   = "docs"        # 外部文档（ingest写入）
FACTS_COLLECTION  = "long_term"   # 长期事实（可选）
NOTES_COLLECTION  = "user_notes"  # ✨ 新增：用户命名记忆
PERSIST_DIR       = VECTOR_DB_PATH

# ---------- 工具函数 ----------
def build_recalled_context(recalls, min_score=0.2, max_items=5, label=""):
    seen, kept = set(), []
    for r in sorted(recalls, key=lambda x: x["score"], reverse=True):
        if r["score"] < min_score: continue
        key = r["text"].strip()[:100]
        if key in seen: continue
        seen.add(key); kept.append(r)
        if len(kept) >= max_items: break
    lines = []
    for i, r in enumerate(kept, 1):
        src = r.get("meta", {}).get("source") or r.get("meta", {}).get("name") or r.get("meta", {}).get("path") or ""
        src_info = f"  ({src})" if src else ""
        tag = f"[{label}]" if label else ""
        lines.append(f"[R{i}{tag}] {r['text']}{src_info}")
    return kept, "\n".join(lines)

def query_all(v_docs, v_facts, v_notes, q: str, k_each=6, min_score=0.2):
    docs_hits  = v_docs.query(q,  k=k_each) if v_docs  else []
    facts_hits = v_facts.query(q, k=k_each) if v_facts else []
    notes_hits = v_notes.query(q, k=k_each) if v_notes else []

    kd, bd = build_recalled_context(docs_hits,  min_score, 5, "docs")
    kf, bf = build_recalled_context(facts_hits, min_score, 3, "facts")
    kn, bn = build_recalled_context(notes_hits, min_score, 5, "note")

    blocks = [b for b in (bd, bf, bn) if b]
    return kd, kf, kn, "\n".join(blocks)

def extract_saveas(cmd: str):
    # 形如：saveas 记忆名: 内容
    m = re.match(r"^saveas\s+([^\:]+?)\s*:\s*(.+)$", cmd, flags=re.I)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)

# ---------- 主程序 ----------
def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    memory     = ChatMemory(max_turns=10)
    vmem_docs  = VectorMemory(persist_dir=PERSIST_DIR, collection=DOCS_COLLECTION)
    vmem_facts = VectorMemory(persist_dir=PERSIST_DIR, collection=FACTS_COLLECTION)
    vmem_notes = VectorMemory(persist_dir=PERSIST_DIR, collection=NOTES_COLLECTION)  # 新增集合

    print("🤖 Chatbot 已启动，输入 'exit' 退出。")
    print("命令：\n"
          "  rag? 关键词        查看 RAG 命中\n"
          "  save 名称          保存最近对话摘要为命名记忆\n"
          "  saveas 名称: 内容  直接把指定内容保存为记忆\n"
          "  recall 名称/关键词  召回记忆并注入上下文\n"
          "  list memories      列出所有命名记忆（Top 20）\n"
          "  delete 名称        删除最相关的一条命名记忆\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 再见！"); break

        # ---------- 调试/管理命令 ----------
        # 1) 查看 RAG 命中
        if user_input.lower().startswith("rag?"):
            q = user_input.split("?", 1)[1].strip() or " "
            _, _, _, block = query_all(vmem_docs, vmem_facts, vmem_notes, q, k_each=8, min_score=0.0)
            print("\n🔎 RAG 命中：\n" + (block or "(无检索结果)")); continue

        # 2) saveas 名称: 内容
        name, content = extract_saveas(user_input)
        if name and content:
            vmem_notes.add_memories([content], [{"type":"user_note", "name": name}])
            print(f"✅ 已保存记忆：{name}"); continue

        if re.match(r"^save\d*\s", user_input.lower()):
        # 匹配 save / save2 / save3 / save10 ...
            m = re.match(r"^save(\d*)\s+(.+)$", user_input.strip(), flags=re.I)
            count_str, name = m.groups()
            max_msgs = int(count_str) * 2 if count_str else 8  # 每轮2条消息(user+assistant)
            text = memory.to_text(max_msgs=max_msgs) or "(空)"
            vmem_notes.add_memories([text], [{"type": "user_note", "name": name}])
            print(f"✅ 已保存最近 {max_msgs//2} 轮对话为记忆：{name}") 
            continue

        # 4) list memories
        if user_input.lower().startswith("list memories"):
            hits = vmem_notes.query("", k=20)  # 空查询时 peek，VectorMemory.query 已做保护
            if not hits:
                print("（暂无命名记忆）")
            else:
                print("\n🗂 已保存记忆（Top 20）：")
                for i, h in enumerate(hits, 1):
                    nm = h.get("meta", {}).get("name") or "(未命名)"
                    print(f"{i}. {nm}  score={h['score']:.2f}")
            continue

        # 5) delete 名称
        if user_input.lower().startswith("delete "):
            name = user_input.split(" ", 1)[1].strip()
            cand = vmem_notes.query(name, k=1)
            if not cand:
                print("❌ 未找到可删除的记忆。"); continue
            _id = cand[0]["id"]
            # Chroma Python 客户端支持 delete(ids=[...])（你的 VectorMemory 没封装，直接用底层）
            vmem_notes.col.delete(ids=[_id])
            print(f"🗑️ 已删除：{cand[0].get('meta',{}).get('name','(未命名)')}"); continue

        # 6) recall 名称/关键词 —— 召回并注入上下文
        if user_input.lower().startswith("recall "):
            key = user_input.split(" ", 1)[1].strip()
            hits = vmem_notes.query(key, k=5)
            if not hits:
                print("❌ 未找到相关记忆。"); continue
            _, block = build_recalled_context(hits, min_score=0.0, max_items=5, label="note")
            # 直接把召回的记忆作为 system 上下文注入（一次性有效）
            memory.add("system", "【召回记忆】\n" + block)
            print("🔁 已将记忆注入上下文，本轮回答会参考以上内容。"); 
            # 不继续，因为还要让用户下一条问问题
            continue

        # ---------- 常规对话：先做 RAG 召回，再问答 ----------
        kd, kf, kn, recalled_block = query_all(
            vmem_docs, vmem_facts, vmem_notes, user_input, k_each=8, min_score=0.2
        )
        context_msg = ""
        if recalled_block:
            context_msg = ("【检索到的相关资料（请优先依据这些片段回答，并在句末标注 [R#] 引用；"
                           "docs=外部文档，facts=长期事实，note=用户记忆）】\n" + recalled_block)

        # 短期记忆：写入用户消息
        memory.add("user", user_input)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "developer", "content": DEVELOPER_HINT},
        ]
        if context_msg:
            messages.append({"role": "system", "content": context_msg})
        messages += memory.get()

        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL, messages=messages, temperature=0.7
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ 出错了: {e}"); continue

        print("Chatbot：", reply, "\n")
        memory.add("assistant", reply)

if __name__ == "__main__":
    main()