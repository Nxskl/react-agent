# schemas.py

import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langgraph.graph import MessagesState




class requirement(BaseModel):

    description: str = Field(
        description="Bullet point description of the functional requirement.",
    )
    suggestions: str = Field(
        description="Bullet point of suggestions related to the functional requirement.",
    )
    presumptions: str = Field(
        description="List of presumptions related to the functional requirement.",
    )
    questions: List[str] = Field(
        description="List of questions related to the functional requirement.",
    )
    
    @property
    def summary(self) -> str:
        return (
            f"Description: {self.description}\n"
            f"Suggestions: {self.suggestions}\n"
            f"Presumptions: {self.presumptions}\n"
            f"Questions: {', '.join(self.questions)}\n"
        )
        
        
class functional_requirement(BaseModel):
    requirements: str = Field(
        description="Comprehensive list of functional requirements with suggestions, presumptions, and questions.",
    )


class developer(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the developer.",
    )
    name: str = Field(
        description="Name of the developer."
    )
    role: str = Field(
        description="Role of the developer in the context of the topic.",
    )
    description: str = Field(
        description="Description of the developer focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    developers: List[developer] = Field(
        description="Comprehensive list of developers with their roles and affiliations.",
    )

class GenerateDeveloperState(TypedDict):
    topic: str  # Research topic
    max_developer: int  # Number of developers
    human_developer_feedback: str  # Human feedback
    developers: List[developer]  # developer asking questions
    requirements: str  # Functional requirements

class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    developer: developer  # developer asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_developer: int  # Number of developers
    human_developer_feedback: str  # Human feedback
    developers: List[developer]  # developer asking questions
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report



'''

class GenerateDeveloperState(TypedDict):
    topic: str  # Research topic


class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    
'''

    