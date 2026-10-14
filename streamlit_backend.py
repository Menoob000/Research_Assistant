import uuid
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic Research Assistant",
    page_icon="🤖",
    layout="wide",
)


# ========================= Helper Functions =========================

def get_backend_threads():
    """Fetch all saved thread IDs from the backend checkpointer."""
    try:
        res = requests.get(f"{BACKEND_URL}/api/threads", timeout=5)
        if res.status_code == 200:
            return res.json().get("threads", [])
    except Exception:
        pass
    return []


def create_new_thread():
    """Request a new thread ID from the backend or generate one locally."""
    try:
        res = requests.post(f"{BACKEND_URL}/api/threads/new", timeout=5)
        if res.status_code == 200:
            return res.json().get("thread_id")
    except Exception:
        pass
    return str(uuid.uuid4())


def load_conversation(thread_id: str):
    """Fetch full message history for a thread from the backend."""
    try:
        res = requests.get(f"{BACKEND_URL}/api/threads/{thread_id}/messages", timeout=10)
        if res.status_code == 200:
            return res.json().get("messages", [])
    except Exception as e:
        st.error(f"Failed to load conversation: {e}")
    return []


def stream_agent_response(query: str, thread_id: str):
    """Stream response token-by-token from the backend."""
    try:
        with requests.post(
            f"{BACKEND_URL}/api/chat/stream",
            json={"query": query, "thread_id": thread_id},
            stream=True,
            timeout=120,
        ) as response:
            if response.status_code != 200:
                yield f"⚠️ Backend error (Status {response.status_code}): {response.text}"
                return
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except requests.exceptions.ConnectionError:
        yield "⚠️ Could not connect to the backend server. Please make sure the FastAPI backend is running on `http://localhost:8000`."
    except Exception as e:
        yield f"⚠️ An error occurred while streaming response: {str(e)}"


# ========================= State Initialization =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = create_new_thread()

if "chat_threads" not in st.session_state:
    existing_threads = get_backend_threads()
    if st.session_state["thread_id"] not in existing_threads:
        existing_threads.insert(0, st.session_state["thread_id"])
    st.session_state["chat_threads"] = existing_threads


# ========================= Sidebar =========================

with st.sidebar:
    st.title("💬 Chat Threads")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_thread = create_new_thread()
        st.session_state["thread_id"] = new_thread
        st.session_state["message_history"] = []
        if new_thread not in st.session_state["chat_threads"]:
            st.session_state["chat_threads"].insert(0, new_thread)
        st.rerun()

    st.markdown("---")
    st.caption("Saved Conversations")

    for thread in st.session_state["chat_threads"]:
        is_active = thread == st.session_state["thread_id"]
        button_label = f"📌 {thread[:8]}..." if len(thread) > 8 else f"📌 {thread}"
        if is_active:
            button_label = f"👉 {button_label} (active)"

        if st.button(button_label, key=f"thread_{thread}", use_container_width=True):
            st.session_state["thread_id"] = thread
            st.session_state["message_history"] = load_conversation(thread)
            st.rerun()


# ========================= Main Chat Interface =========================

st.title("🤖 Agentic Research Assistant")
st.caption(f"Active Thread ID: `{st.session_state['thread_id']}`")

# Display previous messages from history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User chat input
user_input = st.chat_input("Ask a question about research or general topics...")

if user_input:
    # 1. Display and save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Stream assistant response from backend
    with st.chat_message("assistant"):
        ai_response = st.write_stream(
            stream_agent_response(user_input, st.session_state["thread_id"])
        )

    # 3. Save assistant message to history
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_response,
    })

    # Ensure current thread is tracked in sidebar list
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, st.session_state["thread_id"])