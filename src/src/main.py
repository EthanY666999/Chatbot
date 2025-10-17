from openai import OpenAI
from .config import OPENAI_API_KEY, CHAT_MODEL  # 或 MODEL（看你选择的命名）
from .prompt import SYSTEM_PROMPT
from .memory import VectorMemory, ChatMemory

def main():
    """
    一个最小可运行的命令行 Chatbot。
    支持：
      - 连续对话记忆
      - 退出命令（exit / quit）
      - 自动读取 .env 里的 OPENAI_API_KEY
    """
    # 初始化客户端与记忆
    client = OpenAI(api_key=OPENAI_API_KEY)
    memory = ChatMemory(max_turns=10)  # 短期会话
    vmem = VectorMemory()   

    print("🤖 Chatbot 已启动，输入 'exit' 退出。\n")

    # 循环对话
    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 再见！")
            break

        # 存入用户输入
        memory.add("user", user_input)

        # 构造上下文
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory.get()

        try:
            # 调用 OpenAI 接口
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
            )
            reply = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ 出错了: {e}")
            continue

        # 输出并存入记忆
        print("Chatbot：", reply, "\n")
        memory.add("assistant", reply)


if __name__ == "__main__":
    main()