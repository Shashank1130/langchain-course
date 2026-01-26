from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchainhub import Client

from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")

# client = Client()
# react_prompt = client.pull("hwchase17/react")


# New createAgent interface with built-in structured output
agent = create_agent(
    model = llm,
    tools = tools,
    response_format=AgentResponse
)


def main():
    # ✅ New invocation format: use "messages" instead of "input"
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
            }
        ]
    })

    # ✅ Access structured response directly
    print(f"Structured Response: {result.get('structured_response')}")
    # print("Full Result:", result)


if __name__ == "__main__":
    main()