from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import numexpr

from config import API_KEY

llm = ChatOpenAI(
    model="qwen/qwen2.5-vl-32b-instruct:free",
    openai_api_key=API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.4,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)


@tool("Calculator", description="Вычисляет математические выражения", return_direct=True)
def calculator(expression: str) -> str:
    try:
        return str(numexpr.evaluate(expression))
    except Exception as e:
        return f"Ошибка: {e}"


tools = [calculator]

template = """You have access to the following tools:
{tools}

Use the format:

Question: <the question>
Thought: <your thoughts>
What used: <one of [{tool_names}]>
Observation: <result of action>

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=False
)

while True:
    user_msg = input("Ты: ").strip()
    if user_msg.lower() in ("/exit", "exit", "quit"):
        break
    response = agent_executor.invoke({"input": user_msg})
    print(response.get('output'))
