from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchainhub import Client

from prompts import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse


tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(AgentResponse)

client = Client()

react_prompt = client.pull("hwchase17/react")

react_prompt_with_format_instructions = PromptTemplate(
    template = REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables = ["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions="")

agent = create_agent(model=llm, tools=tools, system_prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"]) # -> Extract the "output" from the output dictionary of the agent response
chain = agent_executor | extract_output | structured_llm


def main():
    print("Hello from langchain-course!")
    result = agent_executor.invoke(
        {
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
