import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


# Custom Tool from Tavily
# tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# We can use this fucntion when we are using tavily from Tavily SDK
# @tool
# def search_web(query: str) -> str:
#     """
#     Tool helper to search the web for information about the query
#     Args:
#         query: The query to search the web for
#     Returns:
#         The information found on the web about the query
#     """
#     print(f"Searching the web for {query}")
#     return tavily.search(query=query)


## STRUCTURED RESPONSE FROM THE LLM
class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer ans sources"""
    answer: str = Field(description="The agent's answer to the user's question")
    sources: List[Source] = Field(description="List of sources used to generate the answer")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# tools = [search_web]
tools = [TavilySearch()]
agent = create_agent(
    model=llm, 
    tools=tools,
    response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content= "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
            )
        }
    )
    print(result)

if __name__ == "__main__":
    main()