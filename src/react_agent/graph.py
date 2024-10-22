import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompts import developer_instructions,question_instructions,search_instructions,answer_instructions,section_writer_instructions,report_writer_instructions,intro_conclusion_instructions # or other imports
from langgraph.constants import Send
from schemas import developer, Perspectives, SearchQuery, GenerateDeveloperState, InterviewState, ResearchGraphState
    
from langgraph.graph import END, MessagesState, START, StateGraph
from schemas import InterviewState, ResearchGraphState
from node import (
    generate_question,
    search_web,
    search_wikipedia,
    generate_answer,
    save_interview,
    route_messages,
    write_section,
    create_developers,
    human_feedback,
    initiate_all_interviews,
    dependencies,
    backend_end,
    front_end,
    finalize_report,
    process_requirements,
    human_feedback_for_requirements,
)

### Build Interview Graph











interview_builder = StateGraph(InterviewState)

interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges(
    "answer_question", 
    route_messages, 
    ['ask_question', 'save_interview']
)
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

compiled_interview_graph = interview_builder.compile()

### Build Main Graph

builder = StateGraph(ResearchGraphState)
builder.add_node("create_developer", create_developers)

builder.add_node("process_requirements", process_requirements)

builder.add_node("human_feedback_for_requirements", human_feedback_for_requirements)

builder.add_node("human_feedback", human_feedback)
builder.add_node("conduct_interview", compiled_interview_graph)
builder.add_node("dependencies", dependencies)
builder.add_node("backend_end", backend_end)
builder.add_node("front_end", front_end)
builder.add_node("finalize_report", finalize_report)

# Logic
builder.add_edge(START,"process_requirements")
builder.add_edge("process_requirements","human_feedback_for_requirements")
builder.add_conditional_edges(
    "human_feedback_for_requirements", 
    initiate_all_interviews, 
    ["process_requirements", "create_developer"]
)

builder.add_edge("create_developer", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", 
    initiate_all_interviews, 
    ["create_developer", "conduct_interview"]
)
builder.add_edge("conduct_interview", "dependencies")
builder.add_edge("conduct_interview", "backend_end")
builder.add_edge("conduct_interview", "front_end")
builder.add_edge(
    ["front_end", "dependencies", "backend_end"], 
    "finalize_report"
)
builder.add_edge("finalize_report", END)

# Compile the main graph
graph = builder.compile(interrupt_before=['human_feedback','create_developer','human_feedback_for_requirements' ])









'''


### Build Main Graph

builder = StateGraph(ResearchGraphState)


builder.add_node("create_developers", create_developers)
builder.add_node("human_feedback", human_feedback)



# Logic
builder.add_edge(START,"create_developers")
builder.add_edge("create_developers","human_feedback")

builder.add_edge("human_feedback",END)


# Compile the main graph
graph = builder.compile()

'''