import streamlit as st
import asyncio
from client.mcp_client import MCPClient

st.set_page_config(page_title="Jeju MCP Chatbot", layout="centered")
st.title("Jeju MCP Chatbot :desert_island: ")

if "client" not in st.session_state:
    st.session_state.client = MCPClient()
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)
    st.session_state.loop.run_until_complete(
        st.session_state.client.connect_to_server("jeju_rag_server.py")
    )

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("질문을 입력하세요")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = st.session_state.loop.run_until_complete(
            st.session_state.client.process_query(user_input)
        )
    st.session_state.history.append(("bot", response))

for role, msg in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
