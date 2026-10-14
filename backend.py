import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agent.agent import app, get_all_threads


def generate_thread_id() -> str:
    return str(uuid.uuid4())


backend_app = FastAPI(
    title="Research Assistant API",
    description="Backend service for Agentic Research Assistant LangGraph workflow",
    version="1.0.0",
)

# Enable CORS for frontend requests
backend_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentInput(BaseModel):
    query: str
    thread_id: str


class MessageItem(BaseModel):
    role: str
    content: str


class ConversationHistoryResponse(BaseModel):
    thread_id: str
    messages: List[MessageItem]


class ThreadListResponse(BaseModel):
    threads: List[str]


class NewThreadResponse(BaseModel):
    thread_id: str


@backend_app.get("/")
def health_check():
    return {"status": "ok", "service": "Research Assistant Backend"}


@backend_app.get("/api/threads", response_model=ThreadListResponse)
def list_threads():
    """Retrieve all conversation thread IDs saved in SQLite checkpointer."""
    try:
        threads = get_all_threads()
        return {"threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@backend_app.post("/api/threads/new", response_model=NewThreadResponse)
def create_new_thread():
    """Generate and return a new unique thread ID."""
    return {"thread_id": generate_thread_id()}


@backend_app.get("/api/threads/{thread_id}/messages", response_model=ConversationHistoryResponse)
def get_thread_messages(thread_id: str):
    """Retrieve formatted message history for a specific thread."""
    try:
        state = app.get_state(config={"configurable": {"thread_id": thread_id}})
        raw_messages = state.values.get("messages", []) if state else []

        formatted_messages = []
        for message in raw_messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})

        return {
            "thread_id": thread_id,
            "messages": formatted_messages,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@backend_app.post("/api/chat/stream")
def chat_stream(input: AgentInput):
    """
    Stream assistant responses token-by-token for Streamlit st.write_stream consumption.
    """
    config = {
        "configurable": {"thread_id": input.thread_id},
        "metadata": {"thread_id": input.thread_id},
        "run_name": "chat_trace",
    }

    def event_stream():
        for message_chunk, metadata in app.stream(
            {"messages": [HumanMessage(content=input.query)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessage) and message_chunk.content:
                yield message_chunk.content

    return StreamingResponse(event_stream(), media_type="text/plain")


@backend_app.post("/api/chat")
def chat(input: AgentInput):
    """
    Non-streaming endpoint to get the full final response.
    """
    config = {
        "configurable": {"thread_id": input.thread_id},
        "metadata": {"thread_id": input.thread_id},
        "run_name": "chat_trace",
    }

    try:
        final_state = app.invoke(
            {"messages": [HumanMessage(content=input.query)]},
            config=config,
        )
        messages = final_state.get("messages", [])
        last_ai_message = next(
            (msg.content for msg in reversed(messages) if isinstance(msg, AIMessage)),
            "",
        )
        return {
            "thread_id": input.thread_id,
            "response": last_ai_message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:backend_app", host="0.0.0.0", port=8000, reload=True)