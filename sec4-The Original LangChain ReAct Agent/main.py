from dotenv import load_dotenv

load_dotenv()

from langchainhub import pull
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from tavily import TavilySearch
from langchain.tools import tool

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = pull("hwchase17/react")
agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = react_prompt
)

agent_exector = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True
)
 



def main():
    print("Hello from langchain-course!")

if __name__ == "__main__":
    main()