from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import API_KEY

llm = ChatOpenAI(
    model="qwen/qwen2.5-vl-32b-instruct:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

print("Теперь можешь общаться с ботом.\nКоманды: /exit — выйти\n")

while True:
    user_msg = input("Ты: ").strip()
    if not user_msg:
        continue
    if user_msg.lower() in ("/exit", "exit", "quit"):
        print("Выход...")
        break

    llm.invoke(user_msg)
