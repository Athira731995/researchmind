"""Step 11 - Chat UI. Run: streamlit run app/ui.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src import rag, agent
from src.memory import Memory

st.set_page_config(page_title="ResearchMind", page_icon="🧠")
st.title("🧠 ResearchMind")
st.caption("Ask questions about your documents. Answers are cited.")


@st.cache_resource
def get_store():
    return rag.build_store()


store = get_store()
mode = st.sidebar.radio("Mode", ["RAG (with memory)", "Agent (with calculator)"])
if st.sidebar.button("New conversation"):
    st.session_state.clear()
    st.rerun()

if "memory" not in st.session_state:
    st.session_state.memory = Memory()
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        for s in m.get("sources", []):
            st.caption(f"[{s['id']}] {s['file']}: {s['text']}...")

if question := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if mode.startswith("RAG"):
                out = rag.chat_turn(question, store, st.session_state.memory)
                reply, sources = out["answer"], out["sources"]
            else:
                reply, sources = agent.run_agent(question, store, verbose=False), []
        st.write(reply)
        for s in sources:
            st.caption(f"[{s['id']}] {s['file']}: {s['text']}...")
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})