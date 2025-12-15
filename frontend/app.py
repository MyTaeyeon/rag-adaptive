import streamlit as st
import requests
import time
from typing import List, Dict, Any, Optional

# Control the streaming pace on the UI (seconds per update)
STREAM_DELAY_SECONDS = 0.05

BACKEND_URL = "http://localhost:2022"


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


def get_collection_documents(name: str) -> Dict[str, Any]:
    """Get list of uploaded documents with their sizes."""
    try:
        resp = requests.get(f"{BACKEND_URL}/collections/{name}/documents")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"total_documents": 0, "documents": []}


def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_available_models() -> List[str]:
    """Get list of available answer generation models."""
    try:
        resp = requests.get(f"{BACKEND_URL}/models")
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def query_collection(collection_name: str, query: str, n: Optional[int] = None, model: Optional[str] = None) -> Dict[str, Any]:
    payload = {"query": query}
    if n is not None:
        payload["n"] = n
    if model is not None:
        payload["model"] = model
    resp = requests.post(f"{BACKEND_URL}/collections/{collection_name}/query", json=payload)
    if resp.status_code == 200:
        return resp.json()
    return {}


PAGE_ICON_PATH = "./frontend/assets/page_icon.jpg"
ROLE_AVATARS = {
    "user": "./frontend/assets/user_icon.jpg",
    "assistant": "./frontend/assets/bot_icon.jpg",
}

# Page meta (must be before other Streamlit calls)
st.set_page_config(
    page_title="Adaptive RAG System",
    page_icon=PAGE_ICON_PATH,
    layout="wide",
    initial_sidebar_state="expanded",
)

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
                st.session_state.adaptive_n = 1
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
            
            # Collection Info Section
            if coll_info.get("num_chunks", 0) > 0:
                st.divider()
                st.markdown("**Collection Info**")
                
                # Get documents info
                docs_info = get_collection_documents(selected)
                total_chunks = coll_info.get("num_chunks", 0)
                documents = docs_info.get("documents", [])
                
                # Display total chunks
                st.caption(f"Total Chunks: {total_chunks}")
                
                # Display documents list
                if documents:
                    st.caption(f"Uploaded Documents ({len(documents)}):")
                    for doc in documents:
                        filename = doc.get("filename", "Unknown")
                        file_size = doc.get("file_size", 0)
                        num_chunks = doc.get("num_chunks", 0)
                        formatted_size = format_file_size(file_size)
                        st.text(f"• {filename} ({formatted_size}) - {num_chunks} chunks")
                else:
                    st.caption("No documents uploaded yet")
            
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
    
    # Model selector above chat input
    if "selected_model" not in st.session_state:
        # Initialize with default model from config
        st.session_state.selected_model = None
    
    available_models = get_available_models()
    if available_models:
        selected_model = st.selectbox(
            "Select Model",
            options=available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            key="model_selector",
            help="Choose the OpenAI model for answer generation"
        )
        st.session_state.selected_model = selected_model
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=ROLE_AVATARS.get(message["role"])):
            # Use container to ensure full markdown rendering without truncation
            msg_container = st.container()
            with msg_container:
                st.markdown(message["content"])
            
            if message["role"] == "assistant" and "pipeline_steps" in message:
                with st.expander("Pipeline Steps", expanded=False):
                    steps = message["pipeline_steps"]
                    
                    if "query_rewriting" in steps:
                        step_time = steps['query_rewriting'].get('time', 0)
                        with st.status(f"Step 1: Query Rewriting ({step_time}s)", expanded=True):
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
                                            # Render markdown directly - Streamlit should handle full content
                                            # If truncation occurs, it's likely due to content length limits
                                            st.markdown(output)
                                            st.caption(f"Entropy: {entropy:.3f}")
                    
                    if "retrieval" in steps:
                        step_time = steps['retrieval'].get('time', 0)
                        retrieval_details = steps['retrieval'].get('details', {})
                        num_results = steps['retrieval'].get('num_results', 0)
                        
                        with st.status(f"Step 3: Retrieval ({step_time}s)", expanded=False):
                            st.write(f"Found {num_results} documents")
                            
                            # If k = 0, only show the count and skip all sub-steps
                            if num_results == 0:
                                pass
                            else:
                                # 3.1 Dense Retrieval
                                if "dense_retrieval" in retrieval_details:
                                    dense_info = retrieval_details["dense_retrieval"]
                                    num_of_dense_chunk = dense_info.get('num_of_dense_chunk', dense_info.get('num_results', 0))
                                    with st.expander(f"3.1 Dense Retrieval ({dense_info.get('time', 0)}s)", expanded=False):
                                        st.write(f"Selected {num_of_dense_chunk} dense candidates")
                                        
                                        # Preview top 10 dense candidates
                                        top_10_dense = dense_info.get('top_10_candidates', [])
                                        if top_10_dense:
                                            st.write("**Preview top 10 highest dense score candidates**")
                                            for candidate in top_10_dense:
                                                with st.expander(
                                                    f"3.1.{candidate.get('index', 0)} (score: {candidate.get('score', 0):.4f})",
                                                    expanded=False
                                                ):
                                                    st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                    st.text(candidate.get('text', ''))
                                
                                # 3.2 Sparse Retrieval
                                if "sparse_retrieval" in retrieval_details:
                                    sparse_info = retrieval_details["sparse_retrieval"]
                                    num_of_sparse_chunk = sparse_info.get('num_of_sparse_chunk', sparse_info.get('num_results', 0))
                                    with st.expander(f"3.2 Sparse Retrieval ({sparse_info.get('time', 0)}s)", expanded=False):
                                        st.write(f"Selected {num_of_sparse_chunk} sparse candidates")
                                        
                                        # Preview top 10 sparse candidates
                                        top_10_sparse = sparse_info.get('top_10_candidates', [])
                                        if top_10_sparse:
                                            st.write("**Preview top 10 highest sparse score candidates**")
                                            for candidate in top_10_sparse:
                                                with st.expander(
                                                    f"3.2.{candidate.get('index', 0)} (score: {candidate.get('score', 0):.4f})",
                                                    expanded=False
                                                ):
                                                    st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                    st.text(candidate.get('text', ''))
                                
                                # 3.3 Hybrid Retrieval
                                if "hybrid_retrieval" in retrieval_details:
                                    hybrid_info = retrieval_details["hybrid_retrieval"]
                                    with st.expander("3.3 Hybrid Retrieval", expanded=False):
                                        # 3.3.1 RRF Fusion
                                        if "rrf_rerank" in hybrid_info:
                                            rrf_info = hybrid_info["rrf_rerank"]
                                            total_rrf_candidates = rrf_info.get('total_candidates_for_rrf', 0)
                                            with st.expander(f"3.3.1 RRF Fusion ({rrf_info.get('time', 0)}s)", expanded=False):
                                                st.write(f"Selected {total_rrf_candidates} candidates to calculate RRF score")
                                                
                                                # Preview top 10 RRF candidates
                                                top_10_rrf = rrf_info.get('top_10_rrf_candidates', [])
                                                if top_10_rrf:
                                                    st.write("**Preview top 10 highest RRF score candidates**")
                                                    for candidate in top_10_rrf:
                                                        with st.expander(
                                                            f"3.3.1.{candidate.get('index', 0)} (RRF score: {candidate.get('rrf_score', 0):.4f})",
                                                            expanded=False
                                                        ):
                                                            st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                            st.text(candidate.get('text', ''))
                                        
                                        # 3.3.2 Cross-Encoder Rerank
                                        if "cross_encoder" in hybrid_info:
                                            ce_info = hybrid_info["cross_encoder"]
                                            selected_candidates_count = ce_info.get('selected_candidates', {}).get('num_candidates', 0)
                                            with st.expander(f"3.3.2 Cross-Encoder Rerank ({ce_info.get('time', 0)}s)", expanded=False):
                                                st.write(f"Selected {selected_candidates_count} candidates with highest RRF score")
                                                st.write("Calculate Cross-Encoder Score")
                                                st.write("Ranking by cross-encoder score")
                                                
                                                # Preview top 10 cross-encoder candidates
                                                top_10_ce = ce_info.get('top_10_cross_encoder_candidates', [])
                                                if top_10_ce:
                                                    st.write("**Preview top 10 highest Cross-Encoder score candidates**")
                                                    for candidate in top_10_ce:
                                                        label = candidate.get('label', 'unknown')
                                                        label_color = "green" if label == "accepted" else "red"
                                                        with st.expander(
                                                            f"3.3.2.{candidate.get('index', 0)} (score: {candidate.get('rerank_score', 0):.4f}, state: {label})",
                                                            expanded=False
                                                        ):
                                                            st.markdown(f"**State:** <span style='color: {label_color}'>{label.upper()}</span>", unsafe_allow_html=True)
                                                            st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                            st.caption(f"RRF Score: {candidate.get('rrf_score', 0):.4f} | Cross-Encoder Score: {candidate.get('rerank_score', 0):.4f}")
                                                            st.text(candidate.get('text', ''))
                                
                                # 3.4 Final Top K Chunks
                                if "output" in retrieval_details:
                                    output_info = retrieval_details["output"]
                                    with st.expander(f"3.4 Final Top K Chunks ({output_info.get('num_results', 0)} docs)", expanded=False):
                                        for doc in output_info.get("documents", []):
                                            label = doc.get('label', 'accepted')
                                            label_color = "green" if label == "accepted" else "red"
                                            with st.expander(
                                                f"3.4.{doc.get('index', 0)} (score: {doc.get('score', 0):.4f}, state: {label})",
                                                expanded=False
                                            ):
                                                st.markdown(f"**State:** <span style='color: {label_color}'>{label.upper()}</span>", unsafe_allow_html=True)
                                                st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                                                st.caption(f"Cross-Encoder Score: {doc.get('rerank_score', 0):.4f}")
                                                st.text(doc.get('text', ''))
                            
                            # Backward compatibility: Show preview_documents if details not available
                            if not retrieval_details:
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
        
        with st.chat_message("user", avatar=ROLE_AVATARS["user"]):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=ROLE_AVATARS["assistant"]):
            with st.spinner("Thinking..."):
                n = st.session_state.get("adaptive_n", 1)
                model = st.session_state.get("selected_model")
                response = query_collection(st.session_state.selected_collection, prompt, n=n, model=model)
            
            if response:
                answer = response.get("answer", "")
                
                if answer:
                    # Stream by sentences to preserve markdown syntax and avoid text selection issues
                    answer_placeholder = st.empty()
                    
                    # Split by sentences (keep the separators)
                    import re
                    sentence_parts = re.split(r'([.!?]\s+)', answer)
                    
                    # Stream sentence by sentence
                    streamed_text = ""
                    for i in range(len(sentence_parts)):
                        streamed_text += sentence_parts[i]
                        # Update after each complete sentence
                        if i > 0 and (i % 2 == 0 or i == len(sentence_parts) - 1):
                            answer_placeholder.markdown(streamed_text)
                            time.sleep(STREAM_DELAY_SECONDS)
                    
                    # Final render to ensure complete markdown is displayed without truncation
                    # Clear placeholder and render final answer
                    answer_placeholder.empty()
                    st.markdown(answer)
                    
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
                                                # Render markdown directly - Streamlit should handle full content
                                                # If truncation occurs, it's likely due to content length limits
                                                st.markdown(output)
                                                st.caption(f"Entropy: {entropy:.3f}")
                        
                        if "retrieval" in pipeline_steps:
                            step_time = pipeline_steps['retrieval'].get('time', 0)
                            retrieval_details = pipeline_steps['retrieval'].get('details', {})
                            num_results = pipeline_steps['retrieval'].get('num_results', 0)
                            
                            with st.status(f"Step 3: Retrieval ({step_time}s)", expanded=False):
                                st.write(f"Found {num_results} documents")
                                
                                # If k = 0, only show the count and skip all sub-steps
                                if num_results == 0:
                                    pass
                                else:
                                    # 3.1 Dense Retrieval
                                    if "dense_retrieval" in retrieval_details:
                                        dense_info = retrieval_details["dense_retrieval"]
                                        num_of_dense_chunk = dense_info.get('num_of_dense_chunk', dense_info.get('num_results', 0))
                                        with st.expander(f"3.1 Dense Retrieval ({dense_info.get('time', 0)}s)", expanded=False):
                                            st.write(f"Selected {num_of_dense_chunk} dense candidates")
                                            
                                            # Preview top 10 dense candidates
                                            top_10_dense = dense_info.get('top_10_candidates', [])
                                            if top_10_dense:
                                                st.write("**Preview top 10 highest dense score candidates**")
                                                for candidate in top_10_dense:
                                                    with st.expander(
                                                        f"3.1.{candidate.get('index', 0)} (score: {candidate.get('score', 0):.4f})",
                                                        expanded=False
                                                    ):
                                                        st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                        st.text(candidate.get('text', ''))
                                    
                                    # 3.2 Sparse Retrieval
                                    if "sparse_retrieval" in retrieval_details:
                                        sparse_info = retrieval_details["sparse_retrieval"]
                                        num_of_sparse_chunk = sparse_info.get('num_of_sparse_chunk', sparse_info.get('num_results', 0))
                                        with st.expander(f"3.2 Sparse Retrieval ({sparse_info.get('time', 0)}s)", expanded=False):
                                            st.write(f"Selected {num_of_sparse_chunk} sparse candidates")
                                            
                                            # Preview top 10 sparse candidates
                                            top_10_sparse = sparse_info.get('top_10_candidates', [])
                                            if top_10_sparse:
                                                st.write("**Preview top 10 highest sparse score candidates**")
                                                for candidate in top_10_sparse:
                                                    with st.expander(
                                                        f"3.2.{candidate.get('index', 0)} (score: {candidate.get('score', 0):.4f})",
                                                        expanded=False
                                                    ):
                                                        st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                        st.text(candidate.get('text', ''))
                                    
                                    # 3.3 Hybrid Retrieval
                                    if "hybrid_retrieval" in retrieval_details:
                                        hybrid_info = retrieval_details["hybrid_retrieval"]
                                        with st.expander("3.3 Hybrid Retrieval", expanded=False):
                                            # 3.3.1 RRF Fusion
                                            if "rrf_rerank" in hybrid_info:
                                                rrf_info = hybrid_info["rrf_rerank"]
                                                total_rrf_candidates = rrf_info.get('total_candidates_for_rrf', 0)
                                                with st.expander(f"3.3.1 RRF Fusion ({rrf_info.get('time', 0)}s)", expanded=False):
                                                    st.write(f"Selected {total_rrf_candidates} candidates to calculate RRF score")
                                                    
                                                    # Preview top 10 RRF candidates
                                                    top_10_rrf = rrf_info.get('top_10_rrf_candidates', [])
                                                    if top_10_rrf:
                                                        st.write("**Preview top 10 highest RRF score candidates**")
                                                        for candidate in top_10_rrf:
                                                            with st.expander(
                                                                f"3.3.1.{candidate.get('index', 0)} (RRF score: {candidate.get('rrf_score', 0):.4f})",
                                                                expanded=False
                                                            ):
                                                                st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                                st.text(candidate.get('text', ''))
                                            
                                            # 3.3.2 Cross-Encoder Rerank
                                            if "cross_encoder" in hybrid_info:
                                                ce_info = hybrid_info["cross_encoder"]
                                                selected_candidates_count = ce_info.get('selected_candidates', {}).get('num_candidates', 0)
                                                with st.expander(f"3.3.2 Cross-Encoder Rerank ({ce_info.get('time', 0)}s)", expanded=False):
                                                    st.write(f"Selected {selected_candidates_count} candidates with highest RRF score")
                                                    st.write("Calculate Cross-Encoder Score")
                                                    st.write("Ranking by cross-encoder score")
                                                    
                                                    # Preview top 10 cross-encoder candidates
                                                    top_10_ce = ce_info.get('top_10_cross_encoder_candidates', [])
                                                    if top_10_ce:
                                                        st.write("**Preview top 10 highest Cross-Encoder score candidates**")
                                                        for candidate in top_10_ce:
                                                            label = candidate.get('label', 'unknown')
                                                            label_color = "green" if label == "accepted" else "red"
                                                            with st.expander(
                                                                f"3.3.2.{candidate.get('index', 0)} (score: {candidate.get('rerank_score', 0):.4f}, state: {label})",
                                                                expanded=False
                                                            ):
                                                                st.markdown(f"**State:** <span style='color: {label_color}'>{label.upper()}</span>", unsafe_allow_html=True)
                                                                st.caption(f"Source: {candidate.get('metadata', {}).get('source', 'Unknown')}")
                                                                st.caption(f"RRF Score: {candidate.get('rrf_score', 0):.4f} | Cross-Encoder Score: {candidate.get('rerank_score', 0):.4f}")
                                                                st.text(candidate.get('text', ''))
                                    
                                    # 3.4 Final Top K Chunks
                                    if "output" in retrieval_details:
                                        output_info = retrieval_details["output"]
                                        with st.expander(f"3.4 Final Top K Chunks ({output_info.get('num_results', 0)} docs)", expanded=False):
                                            for doc in output_info.get("documents", []):
                                                label = doc.get('label', 'accepted')
                                                label_color = "green" if label == "accepted" else "red"
                                                with st.expander(
                                                    f"3.4.{doc.get('index', 0)} (score: {doc.get('score', 0):.4f}, state: {label})",
                                                    expanded=False
                                                ):
                                                    st.markdown(f"**State:** <span style='color: {label_color}'>{label.upper()}</span>", unsafe_allow_html=True)
                                                    st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                                                    st.caption(f"Cross-Encoder Score: {doc.get('rerank_score', 0):.4f}")
                                                    st.text(doc.get('text', ''))
                                
                                # Backward compatibility: Show preview_documents if details not available
                                if not retrieval_details:
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
                        "content": answer,  # Use full answer, not streamed_text
                        "pipeline_steps": pipeline_steps
                    })
                else:
                    st.write("No answer generated.")
else:
    st.info("Please select or create a collection from the sidebar to start chatting.")

