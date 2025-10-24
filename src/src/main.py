# src/main.py
from .config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DB_PATH
from .prompt import SYSTEM_PROMPT, DEVELOPER_HINT
from .memory import VectorMemory, ChatMemory
import re
from openai import OpenAI

import os, traceback
DEBUG = os.getenv("DEBUG", "0") == "1" 

# --- Debug/Logging helpers ---
import os, time, traceback

DEBUG = os.getenv("DEBUG", "0") == "1"

def log(*args):
    if DEBUG:
        print("[DEBUG]", *args)

def warn(*args):
    print("⚠️", *args)

def oops(title: str, e: Exception):
    """统一的人类可读错误 & 调试详情"""
    msg = str(e) or e.__class__.__name__
    print(f"❌ {title}: {msg}")
    if DEBUG:
        traceback.print_exc()

# --- OpenAI 调用重试与错误分类 ---
RETRYABLE = ("rate_limit", "timeout", "temporarily_unavailable")

def classify_openai_error(text: str) -> str:
    t = (text or "").lower()
    if "insufficient_quota" in t or "quota" in t:
        return "quota"
    if "invalid_api_key" in t or "authentication" in t or "unauthorized" in t:
        return "auth"
    if "model_not_found" in t or "not found" in t:
        return "model"
    if "connection" in t or "dns" in t or "resolve host" in t:
        return "network"
    if "rate" in t or "429" in t or "retry" in t:
        return "rate"
    if "timeout" in t:
        return "timeout"
    return "unknown"

def call_openai_with_retry(client, model, messages, temperature=0.7, max_tries=3):
    """对 429/网络问题做指数退避重试；其它错误给出具体提示"""
    delay = 1.0
    for attempt in range(1, max_tries + 1):
        try:
            log(f"OpenAI call attempt {attempt}, model={model}, msgs={len(messages)}")
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as e:
            kind = classify_openai_error(str(e))
            if kind in ("rate", "timeout", "network") and attempt < max_tries:
                warn(f"请求暂时失败（{kind}），{delay:.1f}s 后重试…")
                if DEBUG:
                    traceback.print_exc()
                time.sleep(delay)
                delay *= 2
                continue
            # 不可重试或已用尽次数
            if kind == "quota":
                oops("额度不足/已用尽（insufficient_quota）", e)
                print("💡 解决：检查账单/充值或换用可用的 API Key。")
            elif kind == "auth":
                oops("鉴权失败（API Key 无效或未配置）", e)
                print("💡 解决：检查 .env 中的 OPENAI_API_KEY 是否正确；或环境变量是否生效。")
            elif kind == "model":
                oops("模型不可用/不存在", e)
                print("💡 解决：确认 config.CHAT_MODEL 名称无误、账号有权限。")
            elif kind == "network":
                oops("网络错误（DNS/连接）", e)
                print("💡 解决：切换网络/DNS(1.1.1.1/8.8.8.8)，或清理代理。")
            elif kind == "rate":
                oops("请求被限速（429）", e)
                print("💡 解决：降低并发/频率，或提高额度/速率限制。")
            else:
                oops("未知错误", e)
            return None
        


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
        
        # ---- testerr <kind> ：模拟各种错误，验证报错分支 ----
        if user_input.lower().startswith("testerr "):
            kind = user_input.split(" ", 1)[1].strip().lower()
            def show(title): print(f"🔬 模拟：{title}")
            if kind == "quota":
                show("额度用尽"); oops("额度不足/已用尽（insufficient_quota）", Exception("insufficient_quota"))
            elif kind == "auth":
                show("鉴权失败"); oops("鉴权失败（API Key 无效或未配置）", Exception("invalid_api_key"))
            elif kind == "model":
                show("模型不存在"); oops("模型不可用/不存在", Exception("model_not_found"))
            elif kind == "network":
                show("网络/DNS"); oops("网络错误（DNS/连接）", Exception("Could not resolve host"))
            elif kind == "rate":
                show("限速429"); oops("请求被限速（429）", Exception("Rate limit exceeded"))
            elif kind == "timeout":
                show("超时"); oops("未知错误", Exception("timeout"))
            elif kind == "parse":
                show("解析返回失败"); oops("解析 OpenAI 返回内容失败", Exception("list index out of range"))
            elif kind == "vdb":
                show("向量库写入失败"); oops("保存记忆失败（向量库写入）", Exception("chroma write failed"))
            else:
                print("可选：quota/auth/model/network/rate/timeout/parse/vdb")
            continue

        # ---------- 调试/管理命令 ----------
        # 1) 查看 RAG 命中
        if user_input.lower().startswith("rag?"):
            q = user_input.split("?", 1)[1].strip() or " "
            _, _, _, block = query_all(vmem_docs, vmem_facts, vmem_notes, q, k_each=8, min_score=0.0)
            print("\n🔎 RAG 命中：\n" + (block or "(无检索结果)")); continue

                # 2) saveas 名称: 内容  （支持中文冒号：）
        name, content = extract_saveas(user_input)
        if name and content:
            try:
                vmem_notes.add_memories([content], [{"type": "user_note", "name": name.strip()}])
                print(f"✅ 已保存记忆：{name.strip()}")
            except Exception as e:
                print(f"⚠️ 保存记忆失败（向量库写入）: {e}")
                if DEBUG:
                    traceback.print_exc()
            continue
        elif user_input.lower().startswith("saveas"):
            print("⚠️ 用法：saveas 名称: 内容   （支持中文冒号：）")
            continue


        # 3) save / save2 / save3 / save10 ...
        if re.match(r"^save\d*\s", user_input.lower()):
            m = re.match(r"^save(\d*)\s+(.+)$", user_input.strip(), flags=re.I)
            if not m:
                print("⚠️ 用法：save[轮数] 名称   例如：save2 项目计划")
                continue

            count_str, name = m.groups()
            if not name.strip():
                print("⚠️ 请提供记忆名称，例如：save3 项目计划")
                continue

            try:
                # 每轮=2条消息（user+assistant）
                max_msgs = int(count_str) * 2 if count_str else 8
            except ValueError:
                max_msgs = 8

            try:
                text = memory.to_text(max_msgs=max_msgs) or "(空)"
                vmem_notes.add_memories([text], [{"type": "user_note", "name": name.strip()}])
                print(f"✅ 已保存最近 {max_msgs//2} 轮对话为记忆：{name.strip()}")
            except Exception as e:
                print(f"⚠️ 保存记忆失败（向量库写入）: {e}")
                if DEBUG:
                    traceback.print_exc()
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
                print("❌ 未找到可删除的记忆。"); 
                continue

            _id = cand[0]["id"]

            try:
                # 尝试删除
                vmem_notes.col.delete(ids=[_id])
                print(f"🗑️ 已删除：{cand[0].get('meta', {}).get('name', '(未命名)')}")
            except Exception as e:
                print(f"⚠️ 删除记忆失败（向量库）: {e}")
                # 如果开了调试模式，输出完整堆栈
                if DEBUG:
                    import traceback
                    traceback.print_exc()
            continue

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

            # ---- 细化分类 ----
        if "insufficient_quota" in err or "quota" in err:
            print("💡 原因：API 额度已用完。")
            print("👉 解决：登录 https://platform.openai.com/account/billing 查看账单或充值。")

        elif "invalid_api_key" in err or "authentication" in err:
            print("💡 原因：API Key 无效或环境变量未加载。")
            print("👉 解决：检查 .env 文件或环境变量 OPENAI_API_KEY。")

        elif "model_not_found" in err or "does not exist" in err:
            print("💡 原因：模型名称错误或你账号无访问权限。")
            print("👉 解决：检查 config.py 中的 CHAT_MODEL 是否拼写正确。")

        elif "rate limit" in err or "429" in err:
            print("💡 原因：请求被限速（429）。")
            print("👉 解决：稍等几秒后再试，或降低调用频率。")

        elif "connection" in err or "resolve host" in err or "timeout" in err:
            print("💡 原因：网络或 DNS 问题。")
            print("👉 解决：检查网络连接、代理设置，或更换 DNS（如 8.8.8.8）。")

        else:
            print("💡 原因未知，请查看完整错误日志以排查。")

        continue  # 跳过本轮，继续下一轮输入


if __name__ == "__main__":
    main()