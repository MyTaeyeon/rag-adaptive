import streamlit as st
import requests
import time
from typing import List, Dict, Any, Optional

BACKEND_URL = "http://localhost:8000"


def estimate_tokens(text: str) -> int:
    """
    Ước tính số token từ text.
    Rule of thumb: ~4 characters = 1 token (tiếng Anh)
    Tiếng Việt: ~2-3 characters = 1 token (vì có nhiều ký tự dấu)
    """
    # Ước tính đơn giản: chia số ký tự cho 4
    # Có thể cải thiện bằng cách dùng tiktoken nếu cần chính xác hơn
    return max(1, len(text) // 4)


def list_collections() -> List[str]:
    try:
        resp = requests.get(f"{BACKEND_URL}/collections")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def create_collection(name: str, language: str = "en") -> Dict[str, Any]:
    payload = {"name": name, "language": language}
    resp = requests.post(f"{BACKEND_URL}/collections", json=payload)
    if resp.status_code == 200:
        return {"success": True, "message": resp.json().get("message", "")}
    return {"success": False, "message": resp.json().get("detail", "Error")}


def delete_collection(name: str) -> Dict[str, Any]:
    resp = requests.delete(f"{BACKEND_URL}/collections/{name}")
    if resp.status_code == 200:
        return {"success": True, "message": resp.json().get("message", "")}
    return {"success": False, "message": resp.json().get("detail", "Error")}


def upload_files(collection_name: str, files: List) -> Dict[str, Any]:
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file.getvalue(), file.type or "application/octet-stream")))
    
    resp = requests.post(
        f"{BACKEND_URL}/collections/{collection_name}/documents",
        files=files_data,
        data={"use_semantic_chunking": "true"}
    )
    if resp.status_code == 200:
        return resp.json()
    return {"message": f"Error: {resp.json().get('detail', 'Unknown error')}", "num_files": 0, "num_chunks": 0}


def get_collection_info(name: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{BACKEND_URL}/collections/{name}/info")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def get_collection_chunks(name: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{BACKEND_URL}/collections/{name}/chunks?skip={skip}&limit={limit}")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total": 0, "chunks": []}


def query_collection(collection_name: str, query: str, n: Optional[int] = None) -> Dict[str, Any]:
    payload = {"query": query}
    if n is not None:
        payload["n"] = n
    resp = requests.post(f"{BACKEND_URL}/collections/{collection_name}/query", json=payload)
    if resp.status_code == 200:
        return resp.json()
    return {}


st.set_page_config(page_title="RAG System", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None

if "chunks_loaded" not in st.session_state:
    st.session_state.chunks_loaded = {}

with st.sidebar:
    st.title("Collections")
    
    collections_list = list_collections()
    
    with st.expander("Create Collection", expanded=False):
        new_collection_name = st.text_input("Collection Name", key="new_collection")
        new_collection_lang = st.selectbox("Language", ["en", "vi"], key="new_collection_lang")
        
        if st.button("Create", key="create_collection_btn"):
            if new_collection_name:
                result = create_collection(new_collection_name, new_collection_lang)
                if result["success"]:
                    st.success(result["message"])
                    st.session_state.selected_collection = new_collection_name
                    collections_list = list_collections()
                    st.rerun()
                else:
                    st.error(result["message"])
    
    st.divider()
    
    if collections_list:
        selected = st.selectbox(
            "Select Collection",
            collections_list,
            index=collections_list.index(st.session_state.selected_collection) if st.session_state.selected_collection in collections_list else 0,
            key="collection_select"
        )
        
        if selected != st.session_state.selected_collection:
            st.session_state.selected_collection = selected
            st.session_state.chunks_loaded = {}
            st.rerun()
        
        if selected:
            coll_info = get_collection_info(selected)
            st.caption(f"Language: {coll_info.get('language', 'en').upper()}")
            st.caption(f"Chunks: {coll_info.get('num_chunks', 0)}")
            
            # Adaptive iterations selector
            if "adaptive_n" not in st.session_state:
                st.session_state.adaptive_n = 5
            adaptive_n = st.selectbox(
                "Adaptive Iterations (n)",
                options=list(range(1, 11)),
                index=st.session_state.adaptive_n - 1,
                key="adaptive_n_selector",
                help="Number of non-hop LLM calls to calculate entropy (1-10)"
            )
            st.session_state.adaptive_n = adaptive_n
            
            st.divider()
            
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                key="file_uploader"
            )
            
            if uploaded_files and st.button("Upload", key="upload_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting upload...")
                progress_bar.progress(10)
                time.sleep(0.3)
                
                status_text.text("Reading documents...")
                progress_bar.progress(30)
                
                upload_result = upload_files(selected, uploaded_files)
                
                if "Error" not in upload_result.get("message", ""):
                    st.session_state.chunks_loaded = {}
                    
                    status_text.text("Chunking documents...")
                    progress_bar.progress(50)
                    time.sleep(0.5)
                    
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(70)
                    time.sleep(0.5)
                    
                    status_text.text("Building indices...")
                    progress_bar.progress(90)
                    time.sleep(0.5)
                    
                    status_text.text("Completed")
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success(f"Uploaded {upload_result.get('num_files', 0)} files, created {upload_result.get('num_chunks', 0)} chunks")
                    st.rerun()
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(upload_result.get("message", "Upload failed"))
            
            if coll_info.get("num_chunks", 0) > 0:
                st.divider()
                
                st.markdown("**Chunks Preview**")
                
                if selected not in st.session_state.chunks_loaded:
                    with st.spinner("Loading chunks..."):
                        chunks_data = get_collection_chunks(selected, limit=1000)
                        st.session_state.chunks_loaded[selected] = chunks_data
                else:
                    chunks_data = st.session_state.chunks_loaded[selected]
                
                total_chunks = chunks_data.get("total", 0)
                chunks = chunks_data.get("chunks", [])
                
                st.caption(f"Total: {total_chunks} chunks")
                
                chunk_container = st.container(height=400)
                with chunk_container:
                    for chunk_data in chunks[:100]:
                        chunk_num = chunk_data['chunk_index'] + 1
                        chunk_text = chunk_data['text']
                        num_tokens = estimate_tokens(chunk_text)
                        # Create expander title with token count (styled with markdown)
                        # Create expander title with token count (shown as plain text in title since HTML not supported)
                        expander_title = f"Chunk {chunk_num} ({num_tokens} tokens)"
                        with st.expander(expander_title, expanded=False):
                            # Display chunk info with styled markdown
                            st.markdown(f"**Chunk {chunk_num}** <span style='color: #808080; font-style: italic; font-weight: normal;'>({num_tokens} tokens)</span>", unsafe_allow_html=True)
                            st.caption(f"Source: {chunk_data['metadata'].get('source', 'Unknown')}")
                            # Display full chunk text without truncation to show semantic chunking differences
                            st.text(chunk_text)
                
                if total_chunks > 100:
                    st.caption(f"Showing first 100 of {total_chunks} chunks")
            
            st.divider()
            
            if st.button("Delete Collection", key="delete_collection_btn", type="secondary"):
                if st.session_state.get("confirm_delete"):
                    result = delete_collection(selected)
                    if result["success"]:
                        st.success(result["message"])
                        st.session_state.selected_collection = None
                        st.session_state.chunks_loaded = {}
                        st.rerun()
                    else:
                        st.error(result["message"])
                else:
                    st.session_state.confirm_delete = True
                    st.warning("Click again to confirm deletion")

if st.session_state.selected_collection:
    st.title("Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "pipeline_steps" in message:
                with st.expander("Pipeline Steps", expanded=False):
                    steps = message["pipeline_steps"]
                    
                    if "query_rewriting" in steps:
                        step_time = steps['query_rewriting'].get('time', 0)
                        with st.status(f"Step 1: Query Rewriting ({step_time}s)", expanded=False):
                            st.write(f"**Original:** {steps['query_rewriting'].get('original_query', '')}")
                            st.write(f"**Rewritten:** {steps['query_rewriting'].get('rewritten_query', '')}")
                    
                    if "adaptive_k_selection" in steps:
                        step_time = steps['adaptive_k_selection'].get('time', 0)
                        with st.status(f"Step 2: Adaptive K Selection ({step_time}s)", expanded=False):
                            st.metric("K selected", steps['adaptive_k_selection'].get('k_determined', steps['adaptive_k_selection'].get('k', 0)))
                            avg_entropy = steps['adaptive_k_selection'].get('average_entropy')
                            if avg_entropy is not None:
                                st.metric("Average Entropy", f"{avg_entropy:.3f}")
                            n_iterations = steps['adaptive_k_selection'].get('n', 0)
                            if n_iterations is not None and n_iterations > 0:
                                st.write(f"**Iterations:** {n_iterations}")
                                iterations_detail = steps['adaptive_k_selection'].get('iterations_detail', [])
                                if iterations_detail:
                                    st.write("**Iteration responses:**")
                                    for iter_detail in iterations_detail:
                                        run_num = iter_detail.get('run', 0)
                                        style = iter_detail.get('style', '')
                                        output = iter_detail.get('output', '')
                                        entropy = iter_detail.get('entropy', 0.0)
                                        with st.expander(f"Iteration {run_num} (entropy: {entropy:.3f})", expanded=False):
                                            st.caption(f"Style: {style}")
                                            st.markdown("**Response:**")
                                            st.markdown(output)
                                            st.caption(f"Entropy: {entropy:.3f}")
                    
                    if "retrieval" in steps:
                        step_time = steps['retrieval'].get('time', 0)
                        with st.status(f"Step 3: Retrieval ({step_time}s)", expanded=False):
                            st.write(f"Found {steps['retrieval'].get('num_results', 0)} documents")
                            preview_docs = steps['retrieval'].get('preview_documents', [])
                            if preview_docs:
                                st.write("**Preview documents:**")
                                for doc in preview_docs:
                                    with st.expander(f"Document {doc.get('index', 0)} (score: {doc.get('score', 0):.4f})", expanded=False):
                                        st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                                        st.text(doc.get('text', ''))
                    
                    if "answer_generation" in steps:
                        step_time = steps['answer_generation'].get('time', 0)
                        with st.status(f"Step 4: Answer Generation ({step_time}s)", expanded=False):
                            st.write(f"Model: {steps['answer_generation'].get('model', 'N/A')}")
                            st.write(f"Chunks used: {steps['answer_generation'].get('num_chunks_used', 0)}")
    
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                n = st.session_state.get("adaptive_n", 5)
                response = query_collection(st.session_state.selected_collection, prompt, n=n)
            
            if response:
                answer = response.get("answer", "")
                
                if answer:
                    words = answer.split()
                    answer_placeholder = st.empty()
                    streamed_text = ""
                    
                    for word in words:
                        streamed_text += word + " "
                        answer_placeholder.markdown(streamed_text + "▌")
                        time.sleep(0.02)
                    
                    answer_placeholder.markdown(streamed_text)
                    
                    pipeline_steps = response.get("pipeline_steps", {})
                    total_time = pipeline_steps.get("total_time", 0)
                    
                    st.markdown(f"<p style='color: #808080; margin-top: 10px; margin-bottom: 0;'>{total_time}s</p>", unsafe_allow_html=True)
                    
                    with st.expander("Pipeline Steps", expanded=False):
                        if "query_rewriting" in pipeline_steps:
                            step_time = pipeline_steps['query_rewriting'].get('time', 0)
                            with st.status(f"Step 1: Query Rewriting ({step_time}s)", expanded=False):
                                st.write(f"**Original:** {pipeline_steps['query_rewriting'].get('original_query', '')}")
                                st.write(f"**Rewritten:** {pipeline_steps['query_rewriting'].get('rewritten_query', '')}")
                        
                        if "adaptive_k_selection" in pipeline_steps:
                            step_time = pipeline_steps['adaptive_k_selection'].get('time', 0)
                            with st.status(f"Step 2: Adaptive K Selection ({step_time}s)", expanded=False):
                                st.metric("K selected", pipeline_steps['adaptive_k_selection'].get('k_determined', pipeline_steps['adaptive_k_selection'].get('k', 0)))
                                avg_entropy = pipeline_steps['adaptive_k_selection'].get('average_entropy')
                                if avg_entropy is not None:
                                    st.metric("Average Entropy", f"{avg_entropy:.3f}")
                                n_iterations = pipeline_steps['adaptive_k_selection'].get('n', 0)
                                if n_iterations is not None and n_iterations > 0:
                                    st.write(f"**Iterations:** {n_iterations}")
                                    iterations_detail = pipeline_steps['adaptive_k_selection'].get('iterations_detail', [])
                                    if iterations_detail:
                                        st.write("**Iteration responses:**")
                                        for iter_detail in iterations_detail:
                                            run_num = iter_detail.get('run', 0)
                                            style = iter_detail.get('style', '')
                                            output = iter_detail.get('output', '')
                                            entropy = iter_detail.get('entropy', 0.0)
                                            with st.expander(f"Iteration {run_num} (entropy: {entropy:.3f})", expanded=False):
                                                st.caption(f"Style: {style}")
                                                st.markdown("**Response:**")
                                                st.markdown(output)
                                                st.caption(f"Entropy: {entropy:.3f}")
                        
                        if "retrieval" in pipeline_steps:
                            step_time = pipeline_steps['retrieval'].get('time', 0)
                            with st.status(f"Step 3: Retrieval ({step_time}s)", expanded=False):
                                st.write(f"Found {pipeline_steps['retrieval'].get('num_results', 0)} documents")
                                preview_docs = pipeline_steps['retrieval'].get('preview_documents', [])
                                if preview_docs:
                                    st.write("**Preview documents:**")
                                    for doc in preview_docs:
                                        with st.expander(f"Document {doc.get('index', 0)} (score: {doc.get('score', 0):.4f})", expanded=False):
                                            st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                                            st.text(doc.get('text', ''))
                        
                        if "answer_generation" in pipeline_steps:
                            step_time = pipeline_steps['answer_generation'].get('time', 0)
                            with st.status(f"Step 4: Answer Generation ({step_time}s)", expanded=False):
                                st.write(f"Model: {pipeline_steps['answer_generation'].get('model', 'N/A')}")
                                st.write(f"Chunks used: {pipeline_steps['answer_generation'].get('num_chunks_used', 0)}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": streamed_text,
                        "pipeline_steps": pipeline_steps
                    })
                else:
                    st.write("No answer generated.")
else:
    st.info("Please select or create a collection from the sidebar to start chatting.")

