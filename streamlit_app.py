import sys
from pathlib import Path
from agent.agent import app, get_all_threads
import streamlit as st 
import uuid 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def generate_thread_id():
    return str(uuid.uuid4())

def add_new_thread_id(thread_id):
    if thread_id not in st.session_state['chat_threads']: 
        st.session_state['chat_threads'].append(thread_id)

def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # Add the new thread to the conversation list
    add_new_thread_id(st.session_state["thread_id"])

def load_conversation(thread_id):
    state = app.get_state(
        config={
            'configurable' : {
                'thread_id' : thread_id
            }
        }
    )

    return state.values.get("messages", [])

st.title('Agentic Research Assistant')

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

add_new_thread_id(st.session_state["thread_id"])

st.sidebar.title('Chat Threads')

for thread in st.session_state['chat_threads']: 
    if st.sidebar.button(str(thread), key=thread):
        st.session_state['thread_id'] = thread
        messages = load_conversation(thread)
        temp_messages = []


        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue


            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })


        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()



# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])



# Create the chat input box
user_input = st.chat_input("Type here")


# Run this block after the user submits a message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }



    # Create the assistant chat-message container
    with st.chat_message("assistant"):

        # Stream the assistant response token by token
        ai_message = st.write_stream(

            # Return only the content of AI message chunks
            message_chunk.content

            # Stream messages from the LangGraph chatbot
            for message_chunk, metadata in app.stream(
                {
                    # Send the latest user message to the chatbot
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                # Use the current conversation thread
                config=CONFIG,

                # Stream individual message chunks
                stream_mode="messages"
            )

            # Display only AI messages
            # This prevents tool and user messages from appearing
            if isinstance(message_chunk, AIMessage)
        )


    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

