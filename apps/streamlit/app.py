from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import streamlit as st

from shared.dashboard import execution_summary


st.set_page_config(
    page_title="Supply Chain Multi-Agent Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("Supply Chain Multi-Agent Copilot")
st.caption("LangGraph + Azure AI Foundry + Azure AI Search")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000")
    default_session = st.session_state.get("session_id", f"ui-{uuid.uuid4().hex[:8]}")
    session_id = st.text_input("Session ID", value=default_session)
    st.session_state.session_id = session_id
    timeout_seconds = st.number_input("Timeout (seconds)", min_value=10, max_value=300, value=120)

    col1, col2 = st.columns(2)
    if col1.button("New session", use_container_width=True):
        st.session_state.session_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.last_response = None
        st.rerun()
    if col2.button("Clear", use_container_width=True):
        try:
            httpx.delete(
                f"{api_url.rstrip('/')}/sessions/{session_id}",
                timeout=timeout_seconds,
            ).raise_for_status()
            st.session_state.messages = []
            st.session_state.last_response = None
            st.success("Session cleared.")
        except Exception as exc:
            st.error(f"Could not clear session: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None

chat_tab, observability_tab = st.tabs(["Chat", "Observability"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about inventory, suppliers, or logistics")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Running the agent workflow..."):
                try:
                    response = httpx.post(
                        f"{api_url.rstrip('/')}/copilot",
                        json={"session_id": session_id, "message": prompt},
                        timeout=timeout_seconds,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    answer = payload["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.last_response = payload
                except Exception as exc:
                    error = f"API request failed: {exc}"
                    st.error(error)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error}
                    )

    payload = st.session_state.last_response
    if payload:
        with st.expander("Evidence", expanded=True):
            evidence = payload.get("evidence", [])
            if not evidence:
                st.info("No evidence returned.")
            for item in evidence:
                st.markdown(
                    f"**{item['title']}**  \n"
                    f"Source: `{item['source']}`  \n"
                    f"Entity: `{item['entity_id']}`"
                )

with observability_tab:
    payload = st.session_state.last_response
    if not payload:
        st.info("Send a message to populate execution metrics.")
    else:
        summary = execution_summary(payload)
        cols = st.columns(5)
        cols[0].metric("Route", summary["route"])
        cols[1].metric("LLM calls", summary["llm_calls"])
        cols[2].metric("Tool calls", summary["tool_calls"])
        cols[3].metric("Tokens", summary["total_tokens"])
        duration = summary["duration_ms"]
        cols[4].metric("Duration", f"{duration / 1000:.2f}s" if duration else "-")

        st.subheader("Agent workflow")
        st.write("Specialist:", summary["specialist"])
        st.write("Participants:", ", ".join(summary["participants"]) or "none")
        st.write("Tools:", ", ".join(summary["tools"]) or "none")

        st.subheader("Specialist queries")
        queries = payload.get("specialist_queries", {})
        if queries:
            for specialist, query in queries.items():
                st.markdown(f"**{specialist}**")
                st.code(query)
        else:
            st.info("No specialist query planning for this route.")

        st.subheader("Raw execution")
        st.code(json.dumps(payload.get("execution", {}), indent=2, ensure_ascii=False), language="json")
