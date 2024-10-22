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
from prompts import process_instructions,developer_instructions,question_instructions,search_instructions,answer_instructions,section_writer_instructions,report_writer_instructions,intro_conclusion_instructions # or other imports
from langgraph.constants import Send
from schemas import functional_requirement,developer, Perspectives, SearchQuery, GenerateDeveloperState, InterviewState, ResearchGraphState
    
from langgraph.graph import END, MessagesState, START, StateGraph


llm = ChatOpenAI(model="gpt-4o", temperature=0)

### Function Definitions



def process_requirements(state: GenerateDeveloperState):
    
    
    
    """Process requirements"""
    topic = state['topic']
    max_developer = state['max_developer']

    human_developer_feedback = state.get('human_developer_feedback', '')

    # Enforce structured output
    structured_llm = llm.with_structured_output(functional_requirement)

    # System message
    system_message = process_instructions.format(
        topic=topic,
        human_developer_feedback=human_developer_feedback,
        max_developer=max_developer

    )

    # Generate developers
    requirements = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Process the requirements.")
    ])

    # Write the list of developers to state
    return {"requirements": requirements.requirements}

    
    
    
    

def create_developers(state: GenerateDeveloperState):
    """Create developers"""
    topic = state['topic']
    max_developer = state['max_developer']
    human_developer_feedback = state.get('human_developer_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = developer_instructions.format(
        topic=topic,
        human_developer_feedback=human_developer_feedback, 
        max_developer=max_developer
    )

    # Generate developers
    developers = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of developers.")
    ])
    
    # Write the list of developers to state
    return {"developers": developers.developers}

def human_feedback_for_requirements(state: GenerateDeveloperState):
    """No-op node that should be interrupted on"""
    pass

def human_feedback(state: GenerateDeveloperState):
    """No-op node that should be interrupted on"""
    pass

def generate_question(state: InterviewState):
    """Node to generate a question"""
    developer = state["developer"]
    messages = state["messages"]

    # Generate question 
    system_message = question_instructions.format(goals=developer.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
        
    # Write messages to state
    return {"messages": [question]}

def search_web(state: InterviewState):
    """Retrieve documents from web search"""
    tavily_search = TavilySearchResults(max_results=3)

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [SystemMessage(content=search_instructions)] + state['messages']
    )
    
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

    # Format
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted_search_docs]} 

def search_wikipedia(state: InterviewState):
    """Retrieve documents from Wikipedia"""
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke(
        [SystemMessage(content=search_instructions)] + state['messages']
    )
    
    # Search
    search_docs = WikipediaLoader(
        query=search_query.search_query, 
        load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    return {"context": [formatted_search_docs]} 

def generate_answer(state: InterviewState):
    """Node to answer a question"""
    developer = state["developer"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(
        goals=developer.persona, 
        context=context
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """Save interviews"""
    messages = state["messages"]
    interview = get_buffer_string(messages)
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers 
    num_responses = len([
        m for m in messages if isinstance(m, AIMessage) and m.name == name
    ])

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # Get the last question to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

def write_section(state: InterviewState):
    """Node to write a section"""
    interview = state["interview"]
    context = state["context"]
    developer = state["developer"]
   
    system_message = section_writer_instructions.format(focus=developer.description)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source to write your section: {context}")
    ]) 
                
    # Append it to state
    return {"sections": [section.content]}

def initiate_all_interviews(state: ResearchGraphState):
    """Conditional edge to initiate all interviews via Send() API or return to create_developer"""    
    human_developer_feedback = state.get('human_developer_feedback', 'approve')
    if human_developer_feedback.lower() != 'approve':
        return "create_developer"
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {
            "developer": developer,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        }) for developer in state["developers"]]

def dependencies(state: ResearchGraphState):
    """Node to write the final report body"""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([section for section in sections])
    
    system_message = report_writer_instructions.format(
        topic=topic, 
        context=formatted_str_sections
    )    
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos.")
    ]) 
    return {"content": report.content}

def backend_end(state: ResearchGraphState):
    """Node to write the introduction"""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([section for section in sections])
    
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report introduction")
    ]) 
    return {"introduction": intro.content}

def front_end(state: ResearchGraphState):
    """Node to write the conclusion"""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([section for section in sections])
    
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report conclusion")
    ]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """Gather all sections and write the final report"""
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except ValueError:
            sources = None
    else:
        sources = None

    final_report = (
        state["introduction"] + "\n\n---\n\n" + 
        content + "\n\n---\n\n" + state["conclusion"]
    )
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}