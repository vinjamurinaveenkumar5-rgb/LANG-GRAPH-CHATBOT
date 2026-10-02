import os
import streamlit as st
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("GROQ API key loaded:", bool(os.getenv("GROQ_API_KEY")))

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq


# -----------------------------
# 1. Streamlit page
# -----------------------------

st.set_page_config(
    page_title="LangGraph ChatBot",
    page_icon="🤖"
)

st.title("🤖 LangGraph ChatBot")
st.write("Chat with your AI assistant")


# -----------------------------
# 2. Groq LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)


# -----------------------------
# 3. Chat State
# -----------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# 4. Chat Node
# -----------------------------

def chat_node(state: ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [response]
    }


# -----------------------------
# 5. Create LangGraph
# -----------------------------

checkpoint = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("Chat Node", chat_node)

graph.add_edge(START, "Chat Node")
graph.add_edge("Chat Node", END)

chatbot = graph.compile(
    checkpointer=checkpoint
)


# -----------------------------
# 6. Thread ID
# -----------------------------

thread_id = "1"

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# -----------------------------
# 7. Initialize chat history
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# 8. Display previous messages
# -----------------------------

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.write(message["content"])


# -----------------------------
# 9. Chat input
# -----------------------------

user_message = st.chat_input("Type your message...")


if user_message:

    # Display user message
    with st.chat_message("user"):
        st.write(user_message)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    # Send message to LangGraph
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_message)
            ]
        },
        config=config
    )

    # Get AI response
    ai_message = response["messages"][-1].content

    # Display AI response
    with st.chat_message("assistant"):
        st.write(ai_message)

    # Save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_message
    })