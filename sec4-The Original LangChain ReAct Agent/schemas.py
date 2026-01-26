from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source by the agent"""

    url: str = Field(description="The URL of the cource")


class AgentResponse(BaseModel):
    """Schema for agent response with answer ans sources"""

    answer: str = Field(description="The agent's answer to the user's question")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
