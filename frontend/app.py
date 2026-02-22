"""
RLM Document Chatbot - Streamlit Frontend

Based on "Recursive Language Models" by Zhang et al. (2025)
"""

import streamlit as st
import requests
import time
from typing import Optional, List

API_URL = "http://localhost:8000"


def init_state():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = None
    if "use_retrieval" not in st.session_state:
        st.session_state.use_retrieval = True


def check_health():
    """Check backend health."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def get_documents():
    """Get document list."""
    try:
        r = requests.get(f"{API_URL}/documents", timeout=10)
        st.session_state.documents = r.json() if r.status_code == 200 else []
    except:
        st.session_state.documents = []


def upload_file(file):
    """Upload a file."""
    try:
        r = requests.post(
            f"{API_URL}/upload",
            files={"file": (file.name, file.getvalue())},
            timeout=60,
        )
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None


def delete_doc(doc_id: str):
    """Delete a document."""
    try:
        r = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
        return r.status_code == 200
    except:
        return False


def chat(query: str, doc_id: Optional[str] = None, use_retrieval: bool = True):
    """Send chat request."""
    try:
        payload = {"query": query, "use_retrieval": use_retrieval}
        if doc_id:
            payload["doc_id"] = doc_id

        r = requests.post(f"{API_URL}/chat", json=payload, timeout=180)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def show_steps(steps: List[dict]):
    """Display RLM execution steps."""
    if not steps:
        return

    st.markdown("**Execution Trace:**")

    for i, step in enumerate(steps):
        step_type = step.get("step_type", "")
        content = step.get("content", "")

        if step_type == "code":
            with st.expander(f"Step {i+1}: CODE", expanded=False):
                st.code(content, language="python")

        elif step_type == "output":
            with st.expander(f"Step {i+1}: OUTPUT", expanded=False):
                st.text(content[:1500] + "..." if len(content) > 1500 else content)

        elif step_type == "llm_call":
            with st.expander(f"Step {i+1}: LLM_CALL", expanded=False):
                st.info(content)

        elif step_type == "final":
            st.success(f"**FINAL:** {content}")


def sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.title("RLM Chatbot")

        with st.expander("What is RLM?"):
            st.markdown("""
            **Recursive Language Models** store context
            as a variable instead of in the prompt.

            The LLM writes code to:
            - `print(context[:1000])` - peek
            - `re.findall(...)` - search
            - `llm_query(...)` - recursive call
            - `FINAL(answer)` - return answer

            [Paper](https://arxiv.org/abs/2512.24601)
            """)

        st.divider()

        # Upload
        st.subheader("Upload")
        file = st.file_uploader("File", type=["pdf", "txt", "md", "docx"])

        if file and st.button("Upload", type="primary"):
            with st.spinner("Processing..."):
                result = upload_file(file)
                if result:
                    st.success(f"Added: {result['filename']}")
                    get_documents()

        st.divider()

        # Documents
        st.subheader("Documents")
        if st.button("Refresh"):
            get_documents()

        if st.session_state.documents:
            options = ["All"] + [d["filename"] for d in st.session_state.documents]
            sel = st.selectbox("Filter:", options)

            if sel == "All":
                st.session_state.selected_doc = None
            else:
                idx = options.index(sel) - 1
                st.session_state.selected_doc = st.session_state.documents[idx]["doc_id"]

            for doc in st.session_state.documents:
                c1, c2 = st.columns([4, 1])
                c1.text(f"{doc['filename'][:20]}")
                if c2.button("X", key=f"d_{doc['doc_id']}"):
                    if delete_doc(doc["doc_id"]):
                        get_documents()
                        st.rerun()
        else:
            st.info("No documents")

        st.divider()
        st.subheader("Context Mode")
        mode = st.radio(
            "Use retrieval or full documents?",
            ["Retrieve top chunks (fast)", "Use full document(s) (slow)"],
            index=0 if st.session_state.use_retrieval else 1,
        )
        st.session_state.use_retrieval = (mode == "Retrieve top chunks (fast)")
        if not st.session_state.use_retrieval:
            st.caption("Full documents can be large and may slow responses.")

        st.divider()
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()


def main_chat():
    """Render chat interface."""
    st.title("RLM Document Chat")

    if st.session_state.selected_doc:
        doc = next((d for d in st.session_state.documents
                   if d["doc_id"] == st.session_state.selected_doc), None)
        if doc:
            st.info(f"Filtering: {doc['filename']}")

    # History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "steps" in msg:
                show_steps(msg["steps"])

    # Input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("RLM processing..."):
                start = time.time()
                result = chat(
                    prompt,
                    st.session_state.selected_doc,
                    st.session_state.use_retrieval,
                )
                elapsed = time.time() - start

            if "error" in result:
                st.error(result["error"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {result['error']}",
                })
            else:
                response = result.get("response", "")
                st.markdown(response)

                st.caption(
                    f"LLM calls: {result.get('llm_calls', 0)} | "
                    f"Context: {result.get('context_length', 0):,} chars | "
                    f"Time: {elapsed:.1f}s"
                )

                steps = result.get("steps", [])
                show_steps(steps)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "steps": steps,
                })


def error_page():
    """Show when backend not available."""
    st.error("Cannot connect to backend")

    st.markdown("""
    ### Start the backend:
    ```bash
    cd rlm-chatbot
    source venv/bin/activate
    uvicorn backend.main:app --port 8000
    ```

    ### Or use run.sh:
    ```bash
    ./run.sh
    ```
    """)

    if st.button("Retry"):
        st.rerun()


def main():
    st.set_page_config(
        page_title="RLM Chatbot",
        page_icon="brain",
        layout="wide",
    )

    init_state()

    health = check_health()
    if not health:
        error_page()
        return

    with st.sidebar:
        if health.get("status") == "healthy":
            st.success("Connected")
        else:
            st.warning("Degraded")
        st.divider()

    if not st.session_state.documents:
        get_documents()

    sidebar()
    main_chat()


if __name__ == "__main__":
    main()
