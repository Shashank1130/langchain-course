import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


# Custom Tool from Tavily
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def search_web(query: str) -> str:
    """
    Tool helper to search the web for information about the query
    Args:
        query: The query to search the web for
    Returns:
        The information found on the web about the query
    """
    print(f"Searching the web for {query}")
    return tavily.search(query=query)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [search_web]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content= "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()