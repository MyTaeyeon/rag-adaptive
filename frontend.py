"""
Streamlit frontend for the Adaptive RAG system
============================================

This file implements a simple Streamlit application that interacts with the
FastAPI backend (defined in ``backend.py``) to manage collections of
documents, upload files, and issue queries.  The user can create, list and
delete collections, upload PDF/DOCX/plain-text files, and execute queries
against the selected collection.  Results are displayed with their text,
metadata, and fused scores.

To run the frontend:

```
streamlit run frontend.py
```

Ensure the FastAPI backend is running at the configured ``backend_url``.
"""

import io
import requests
import streamlit as st


# Base URL of the backend API
backend_url = "http://localhost:8000"


def list_collections() -> list[str]:
    try:
        resp = requests.get(f"{backend_url}/collections")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def create_collection(name: str) -> str:
    payload = {"name": name}
    resp = requests.post(f"{backend_url}/collections", json=payload)
    if resp.status_code != 200:
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    return resp.json().get("message", "Created")


def delete_collection(name: str) -> str:
    resp = requests.delete(f"{backend_url}/collections/{name}")
    if resp.status_code != 200:
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    return resp.json().get("message", "Deleted")


def upload_files(name: str, files: list[io.BytesIO]) -> str:
    # Prepare multipart form-data
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type)))
    resp = requests.post(f"{backend_url}/collections/{name}/documents", files=files_data)
    if resp.status_code != 200:
        return f"Error: {resp.json().get('detail', 'Unknown error')}"
    return resp.json().get("message", "Uploaded")


def query_collection(name: str, query: str) -> dict:
    """
    Issue a query to the backend and return the response.

    The backend returns a JSON object with ``k`` and ``results``.
    """
    payload = {"query": query}
    resp = requests.post(f"{backend_url}/collections/{name}/query", json=payload)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(f"Query failed: {resp.json().get('detail', 'Unknown error')}")
        return {}


def main() -> None:
    st.title("Adaptive RAG Demo")
    st.sidebar.header("Collections")
    if "collections" not in st.session_state:
        st.session_state.collections = list_collections()

    # Create new collection
    new_name = st.sidebar.text_input("New collection name:")
    if st.sidebar.button("Create collection") and new_name:
        msg = create_collection(new_name)
        st.sidebar.success(msg)
        st.session_state.collections = list_collections()

    # Refresh collections list
    if st.sidebar.button("Refresh collections"):
        st.session_state.collections = list_collections()

    # Select existing collection
    collections = st.session_state.collections
    selected = st.sidebar.selectbox("Select collection", [""] + collections)

    # Delete selected collection
    if selected and st.sidebar.button("Delete collection"):
        msg = delete_collection(selected)
        st.sidebar.success(msg)
        st.session_state.collections = list_collections()
        selected = ""

    if selected:
        st.header(f"Collection: {selected}")
        # Upload documents
        uploaded = st.file_uploader("Upload documents (PDF/DOCX/TXT)", accept_multiple_files=True)
        if uploaded and st.button("Upload"):
            msg = upload_files(selected, uploaded)
            st.success(msg)
        # Query
        query_text = st.text_input("Enter your query:")
        if st.button("Search") and query_text:
            resp = query_collection(selected, query_text)
            if resp:
                k_val = resp.get("k", 0)
                results = resp.get("results", [])
                st.markdown(f"**Adaptive k = {k_val} documents**")
                if results:
                    st.subheader("Results")
                    for i, r in enumerate(results, start=1):
                        st.markdown(f"**Result {i} (score {r['score']:.4f})**")
                        st.markdown(f"*Source:* {r['metadata']['source']} – chunk {r['metadata']['chunk_index']}")
                        st.write(r['text'])


if __name__ == "__main__":
    main()