from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from config import API_KEY

llm = ChatOpenAI(
    model="qwen/qwen2.5-vl-32b-instruct:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4
)

history = InMemoryChatMessageHistory()
chat = RunnableWithMessageHistory(
    llm,
    lambda _: history
)

config = {"configurable": {"session_id": "user1"}}

name = input("Введите своё имя: ").strip()
resp = chat.invoke(f"Меня зовут {name}. Запомни, пожалуйста, что моё имя — {name}.", config=config)
print("Бот:", resp.content)

print("\nТеперь можешь общаться с ботом. Напиши 'как меня зовут?' чтобы проверить память.")
print("Команды: /reset — очистить память, /exit — выйти\n")

while True:
    user_msg = input("Ты: ").strip()
    if not user_msg:
        continue
    if user_msg.lower() in ("/exit", "exit", "quit"):
        print("Выход...")
        break
    if user_msg.lower() == "/reset":
        history.clear()
        print("Память очищена.")
        continue

    answer = chat.invoke(user_msg, config=config)
    print("Бот:", answer.content)