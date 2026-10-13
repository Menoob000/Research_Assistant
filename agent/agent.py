import os
from typing import Annotated, List, TypedDict, Literal

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field, BaseModel
from config import MODEL_NAME
from rag.vector_store import VectorStore
from tools.search import search_tool
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

llm = ChatOpenRouter(model=MODEL_NAME)

tools = [search_tool]

llm_with_tools = llm.bind_tools(tools)

_vector_store = None

conn = sqlite3.connect("prototype.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)


def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    reflection: str
    refined_query: str
    iterations: int
    context: str
    reflection_log: List[str]
    route : str

class RouteDecision(BaseModel):
    decision : Literal["retrieve", "general"] = Field(
        description="Use retrieve for questions that may need RAG; general for simple conversations like greetings and simple questions"
    )

router_llm = llm.with_structured_output(RouteDecision, method="json_mode")
def route_decision(state: AgentState):
    """A node to route to a general agent, or a rag based agent"""

    prompt = f"""
    You are a router agent that's going to evaluate whether the user's question is a general and simple question, that can be answered without retrieving using RAG

    Route to 'general' if : 
    The user asked a simple question like : 
        - Hi, who are you
        - What's the capital of France

    Route to 'retrieve' if the user asked a question that may need a particular document to solve like : 
        - Who are the authors of this research paper
        - How many contributors are there in this research paper
        - What's the methodology of this research paper
    
    The user's question : 
    {state['messages'][-1].content}
Return your response as valid JSON.
Example:
{{"route": "general"}}
"""
    response = router_llm.invoke(prompt)
    print("[Router]", response.decision)
    return {
        'route' : response.decision
    }

def route_after_router(state: AgentState):
    if state["route"] == "general":
        return "general"
    return "retrieve"

def extract_text(response) -> str:
    """
    Safely extract a plain string from an LLM response.
    """
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(parts).strip()
    return str(content).strip()


def retriever(state: AgentState):
    """
    Retrieve documents from the vector store using the query or refined_query.
    """
    query = state.get("refined_query") or state.get("query")
    if not query and state.get("messages"):
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                query = msg.content
                break
            elif getattr(msg, "content", None):
                query = msg.content
                break

    if not query:
        query = ""

    store = get_vector_store()
    retriever_instance = store.as_retriever()
    try:
        retrieved_docs = retriever_instance.invoke(query)
    except Exception:
        retrieved_docs = []

    context = "\n\n".join(
        [f"[Chunk {i + 1}]:\n{d.page_content}" for i, d in enumerate(retrieved_docs)]
    )

    update = {"context": context, "query": query}
    if not state.get("messages") and query:
        update["messages"] = [HumanMessage(content=query)]

    return update


def rewrite_query(state: AgentState):
    """
    Extract the REFINED_QUERY from the grader's output.
    Falls back to the original question if parsing fails.
    """
    reflection = state.get("reflection", "")
    fallback = state.get("query") or (
        state["messages"][-1].content if state.get("messages") else ""
    )
    refined = fallback

    for line in reflection.splitlines():
        line = line.strip()
        if line.upper().startswith("REFINED_QUERY:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate and candidate.upper() != "NONE":
                refined = candidate
                break

    return {"refined_query": refined}


def grade_retrieval(state: AgentState) -> dict:
    """
    LLM judges whether the retrieved context is relevant and sufficient.
    Outputs a structured verdict and, if needed, a refined search query.
    """
    question = state.get("query")
    if not question and state.get("messages"):
        question = state["messages"][-1].content
    if not question:
        question = ""

    prompt = f"""You are a strict retrieval quality judge for a RAG system.

Your job: decide if the retrieved context is good enough to answer the question accurately.

Question: {question}

Retrieved Context:
{state.get('context', '')}

Evaluate the context on three criteria:
1. RELEVANCE — Does it directly address the question?
2. SUFFICIENCY — Does it contain enough detail for a complete answer?
3. CONSISTENCY — Are there contradictions between chunks?

Reply in this EXACT format (no extra lines):
VERDICT: YES
REASON: <one sentence>
REFINED_QUERY: NONE

OR if context is not good enough:
VERDICT: NO
REASON: <one sentence explaining what is missing or wrong>
REFINED_QUERY: <a better, more specific search query to find the missing information>

"""

    response = llm.invoke(prompt)
    content = extract_text(response)

    current_iteration = state.get("iterations", 0) + 1
    log_entry = f"Iteration {current_iteration}\n{content}"
    reflection_log = list(state.get("reflection_log", [])) + [log_entry]

    return {
        "reflection": content,
        "reflection_log": reflection_log,
        "iterations": current_iteration,
    }


def route_to_general(state: AgentState):
    """
    Route to 'general' assistant if there are no retrieved documents,
    otherwise route to 'grade' to assess retrieval quality.
    """
    context = state.get("context", "")
    if not context or not context.strip():
        return "general"
    return "grade"


def general_assistant(state: AgentState):
    """
    General assistant to answer questions using tools (e.g. web search)
    when vector retrieval is empty or unavailable.
    """
    sys_prompt = (
        "You are a helpful assistant with access to tools. "
        "Answer the user's question with as much detail and context as possible using the available tools."
        "Tools : {tools}"
        "Always print out the name of the tools you are using"
    )
    history = list(state.get("messages", []))
    if not history and state.get("query"):
        history = [HumanMessage(content=state["query"])]

    messages = [SystemMessage(content=sys_prompt.format(tools=tools))] + history
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Route to 'generate' if context passed grading or max iterations reached.
    Route to 'rewrite' otherwise to refine the query and re-retrieve.
    Route to 'general' if the user's question is of a general kind that can be answered without retrieving documents from the vector db.
    """
    MAX_ITERATIONS = 3
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        return "generate"  # forced exit — generate with best context so far

    reflection = state.get("reflection", "")
    for line in reflection.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            if "YES" in line.upper():
                return "generate"
            

    return "rewrite"


workflow = StateGraph(AgentState)

tool_node = ToolNode(tools)
workflow.add_node("retrieve", retriever)
workflow.add_node("grade_retrieval", grade_retrieval)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node('initial_decision', route_decision)

workflow.add_node("general", general_assistant)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("initial_decision")
workflow.add_conditional_edges("initial_decision", route_after_router, {
    "general" : "general",
    "retrieve" : "retrieve"
})
workflow.add_edge("retrieve", "grade_retrieval")

workflow.add_conditional_edges(
    "grade_retrieval",
    should_continue,
    {
        "generate": "general",
        "rewrite": "rewrite_query",
    
    },
)
workflow.add_conditional_edges("general", tools_condition)
workflow.add_edge("tools", "general")
workflow.add_edge("rewrite_query", "retrieve")


app = workflow.compile(checkpointer=checkpoint)

try:
    graph = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as file:
        file.write(graph)
except Exception as e:
    print(f"Could not save graph.png: {e}")
