import operator
from typing import List, Annotated
from typing_extensions import TypedDict

from langgraph.graph import MessagesState
from schemas import developer

class GeneratedevelopersState(TypedDict):
    topic: str  # Research topic
    max_developer: int  # Number of developers
    human_developer_feedback: str  # Human feedback
    developers: List[developer]  # List of developers

class InterviewState(MessagesState):
    max_num_turns: int  # Number of conversation turns
    context: Annotated[list, operator.add]  # Source documents
    developer: developer  # developer asking questions
    interview: str  # Interview transcript
    sections: list  # Key duplicated in outer state for Send() API

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_developer: int  # Number of developers
    human_developer_feedback: str  # Human feedback
    developers: List[developer]  # List of developers
    sections: Annotated[list, operator.add]  # Send() API key
    introduction: str  # Introduction for the final report
    content: str  # Content for the final report
    conclusion: str  # Conclusion for the final report
    final_report: str  # Final report