"""Main FastAPI application with full RAG pipeline."""

import time
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from ..models.collection import Collection
from .models import (
    CollectionCreateRequest,
    QueryRequest,
    QueryResponse,
    UploadResponse
)
from ..utils.document_processing import read_file_to_text
from ..services.llm_service import rewrite_query, generate_answer
from ..config.config import get_config
from ..models.fusion import reciprocal_rank_fusion

# Global store of collections (in-memory for demonstration)
collections: Dict[str, Collection] = {}

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/collections", response_model=Dict[str, str])
async def create_collection(req: CollectionCreateRequest) -> Dict[str, str]:
    """Create a new collection with specified language and settings."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")
    if name in collections:
        raise HTTPException(status_code=400, detail="Collection already exists")
    
    config = get_config()
    collections[name] = Collection(
        name=name,
        language=req.language or config.api.default_language,
        dense_model_name=req.dense_model_name or config.api.default_dense_model_name or config.retrieval.dense_model_name,
        chunking_model_name=req.chunking_model_name or config.api.default_chunking_model_name or config.retrieval.chunking_model_name,
        similarity_threshold=req.similarity_threshold if req.similarity_threshold is not None else config.api.default_similarity_threshold
    )
    return {"message": f"Collection '{name}' created"}


@app.delete("/collections/{name}", response_model=Dict[str, str])
async def delete_collection(name: str) -> Dict[str, str]:
    """Delete a collection."""
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    del collections[name]
    return {"message": f"Collection '{name}' deleted"}


@app.get("/collections", response_model=List[str])
async def list_collections() -> List[str]:
    """List all collection names."""
    return list(collections.keys())


@app.get("/collections/{name}/info", response_model=Dict[str, Any])
async def get_collection_info(name: str) -> Dict[str, Any]:
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    coll = collections[name]
    return {
        "name": coll.name,
        "language": coll.language,
        "num_chunks": len(coll.chunks),
        "similarity_threshold": coll.similarity_threshold
    }


@app.get("/collections/{name}/documents", response_model=Dict[str, Any])
async def get_collection_documents(name: str) -> Dict[str, Any]:
    """
    Get list of uploaded documents with their sizes.
    Returns unique documents with total chunks per document.
    """
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    coll = collections[name]
    
    # Aggregate documents by source (filename)
    doc_info: Dict[str, Dict[str, Any]] = {}
    
    # First pass: collect document info and count chunks
    for meta in coll.metadata:
        source = meta.get("source", "unknown")
        file_size = meta.get("file_size", 0)
        
        if source not in doc_info:
            doc_info[source] = {
                "filename": source,
                "file_size": file_size,
                "file_type": meta.get("file_type", "unknown"),
                "num_chunks": 0
            }
        
        # Count chunks per document
        doc_info[source]["num_chunks"] += 1
    
    documents_list = list(doc_info.values())
    
    return {
        "total_documents": len(documents_list),
        "documents": documents_list
    }


@app.get("/collections/{name}/chunks", response_model=Dict[str, Any])
async def get_collection_chunks(name: str, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    coll = collections[name]
    
    chunks_with_meta = [
        {
            "chunk_index": i,
            "text": chunk,
            "metadata": coll.metadata[i]
        }
        for i, chunk in enumerate(coll.chunks)
    ]
    
    total = len(chunks_with_meta)
    paginated = chunks_with_meta[skip:skip+limit]
    
    return {
        "total": total,
        "chunks": paginated,
        "skip": skip,
        "limit": limit
    }


@app.post("/collections/{name}/documents", response_model=UploadResponse)
async def upload_documents(
    name: str,
    files: List[UploadFile] = File(...),
    use_semantic_chunking: bool = True
) -> UploadResponse:
    """
    Upload documents to a collection.
    
    Documents will be processed with semantic chunking if enabled.
    """
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    coll = collections[name]
    all_texts: List[str] = []
    all_meta: List[Dict] = []
    
    for f in files:
        try:
            # Read file content to get size
            file_data = f.file.read()
            file_size = len(file_data)
            # Reset file pointer for reading text
            f.file.seek(0)
            text = read_file_to_text(f)
            all_texts.append(text)
            all_meta.append({
                "source": f.filename or "unknown",
                "file_type": f.content_type or "unknown",
                "file_size": file_size  # Size in bytes
            })
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read '{f.filename}': {str(e)}"
            )
    
    if all_texts:
        coll.add_documents(all_texts, all_meta, use_semantic_chunking=use_semantic_chunking)
    
    return UploadResponse(
        message=f"Uploaded {len(files)} documents",
        num_files=len(files),
        num_chunks=len(coll.chunks)
    )


@app.post("/collections/{name}/query", response_model=QueryResponse)
async def run_query(name: str, req: QueryRequest) -> QueryResponse:
    """
    Execute full RAG pipeline query:
    
    1. Query rewriting (using LLM)
    2. Adaptive k selection
    3. Hybrid retrieval (BM25 + Dense)
    4. Reranking
    5. Answer generation (using GPT-4o)
    
    Returns detailed response with all pipeline steps.
    """
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    coll = collections[name]
    
    # Determine language (use request language if provided, otherwise collection language)
    query_language = req.language if req.language else coll.language
    
    total_start_time = time.time()
    
    # Step 1: Query Rewriting
    step1_start = time.time()
    rewriting_result = rewrite_query(req.query, language=query_language)
    rewritten_query = rewriting_result.get("rewritten_query", req.query)
    step1_time = round(time.time() - step1_start, 2)
    
    # Step 2: Adaptive k selection (tính từ khi có rewritten query đến khi có k)
    step2_start = time.time()  # Bắt đầu sau khi có rewritten query
    # Get n from request, default to config if not provided
    n = req.n if req.n is not None else None
    
    # Determine k value using adaptive controller
    config = get_config()
    k, adaptive_metadata = coll.controller.decide(req.query, coll.language, n=n)
    step2_time = round(time.time() - step2_start, 2)
    
    # Step 3: Retrieval (only if k > 0)
    step3_retrieval_start = time.time()
    if k == 0:
        results = []
        retrieval_time = 0.0
        retrieval_details = {
            "dense_retrieval": {"time": 0.0, "num_results": 0, "num_of_dense_chunk": 0},
            "sparse_retrieval": {"time": 0.0, "num_results": 0, "num_of_sparse_chunk": 0},
            "hybrid_retrieval": {
                "rrf_rerank": {"time": 0.0, "num_results": 0, "num_of_cross_encoder_candidates": 0, "num_of_dense_chunk": 0, "num_of_sparse_chunk": 0},
                "cross_encoder": {
                    "time": 0.0,
                    "selected_candidates": {"num_candidates": 0, "documents": []},
                    "top_k_results": {"num_results": 0, "documents": []}
                }
            },
            "output": {"documents": [], "num_results": 0}
        }
    else:
        # Perform retrieval
        retrieval_query = rewritten_query if rewritten_query else req.query
        
        # Calculate number of documents to retrieve from each method
        num_of_dense_chunk = max(1, int(config.fusion.dense_chunk_multiplier * k))
        num_of_sparse_chunk = max(1, int(config.fusion.sparse_chunk_multiplier * k))
        num_of_cross_encoder_candidates = max(k, int(config.fusion.cross_encoder_multiplier * k))
        
        # 1. Dense Retrieval
        dense_start = time.time()
        dense_results = coll.dense.retrieve(retrieval_query, num_of_dense_chunk)
        dense_time = round(time.time() - dense_start, 2)
        
        # Prepare top 10 dense candidates for preview
        top_10_dense_candidates = []
        for i, (idx, score) in enumerate(dense_results[:10]):
            top_10_dense_candidates.append({
                "index": i + 1,
                "text": coll.chunks[idx],
                "score": round(score, 4),
                "metadata": coll.metadata[idx]
            })
        
        # 2. Sparse Retrieval
        sparse_start = time.time()
        bm25_results = coll.bm25.retrieve(retrieval_query, num_of_sparse_chunk)
        sparse_time = round(time.time() - sparse_start, 2)
        
        # Prepare top 10 sparse candidates for preview
        top_10_sparse_candidates = []
        for i, (idx, score) in enumerate(bm25_results[:10]):
            top_10_sparse_candidates.append({
                "index": i + 1,
                "text": coll.chunks[idx],
                "score": round(score, 4),
                "metadata": coll.metadata[idx]
            })
        
        # 3. Hybrid Retrieval
        # 3.1 RRF Rerank
        rrf_start = time.time()
        # RRF fuse all candidates and get top num_of_cross_encoder_candidates
        fused = reciprocal_rank_fusion([bm25_results, dense_results], num_of_cross_encoder_candidates, k_rrf=config.fusion.k_rrf)
        rrf_time = round(time.time() - rrf_start, 2)
        
        # Prepare top 10 RRF candidates for preview
        top_10_rrf_candidates = []
        for i, (idx, rrf_score) in enumerate(fused[:10]):
            top_10_rrf_candidates.append({
                "index": i + 1,
                "text": coll.chunks[idx],
                "rrf_score": round(rrf_score, 4),
                "score": round(rrf_score, 4),
                "metadata": coll.metadata[idx]
            })
        
        # 3.2 Cross-Encoder
        cross_encoder_start = time.time()
        candidate_indices = [idx for idx, _ in fused]
        candidate_texts = [coll.chunks[idx] for idx in candidate_indices]
        rerank_scores = coll.reranker.rerank(retrieval_query, candidate_texts)
        cross_encoder_time = round(time.time() - cross_encoder_start, 2)
        
        # Combine RRF and reranker scores, then sort by rerank_score
        combined = [
            (idx, rrf_score, rr_score)
            for (idx, rrf_score), rr_score in zip(fused, rerank_scores)
        ]
        combined.sort(key=lambda x: x[2], reverse=True)
        
        # Get top k from cross-encoder results
        top_k_combined = combined[:k]
        
        # Prepare cross-encoder documents (all candidates for display)
        all_cross_encoder_documents = []
        for (idx, rrf_score), rr_score in zip(fused, rerank_scores):
            all_cross_encoder_documents.append({
                "index": len(all_cross_encoder_documents) + 1,
                "text": coll.chunks[idx],
                "rrf_score": round(rrf_score, 4),
                "rerank_score": round(rr_score, 4),
                "score": round(rr_score, 4),
                "label": "accepted" if rr_score >= 0 else "denied",
                "metadata": coll.metadata[idx]
            })
        
        # Prepare top 10 cross-encoder candidates for preview (sorted by rerank_score)
        top_10_cross_encoder_candidates = sorted(
            all_cross_encoder_documents,
            key=lambda x: x["rerank_score"],
            reverse=True
        )[:10]
        
        # Top k documents after cross-encoder reranking
        top_k_documents = []
        for idx, rrf_score, rr_score in top_k_combined:
            top_k_documents.append({
                "index": len(top_k_documents) + 1,
                "text": coll.chunks[idx],
                "rrf_score": round(rrf_score, 4),
                "rerank_score": round(rr_score, 4),
                "score": round(rr_score, 4),
                "label": "accepted" if rr_score >= 0 else "denied",
                "metadata": coll.metadata[idx]
            })
        
        # 4. Output (final results - top k after cross-encoder)
        output_documents = top_k_documents.copy()
        results = []
        for idx, rrf_score, rr_score in top_k_combined:
            results.append({
                "text": coll.chunks[idx],
                "metadata": coll.metadata[idx],
                "rrf_score": rrf_score,
                "rerank_score": rr_score,
                "score": rr_score,
            })
        
        retrieval_time = round(time.time() - step3_retrieval_start, 2)
        
        retrieval_details = {
            "dense_retrieval": {
                "time": dense_time,
                "num_results": len(dense_results),
                "num_of_dense_chunk": num_of_dense_chunk,
                "top_10_candidates": top_10_dense_candidates
            },
            "sparse_retrieval": {
                "time": sparse_time,
                "num_results": len(bm25_results),
                "num_of_sparse_chunk": num_of_sparse_chunk,
                "top_10_candidates": top_10_sparse_candidates
            },
            "hybrid_retrieval": {
                "rrf_rerank": {
                    "time": rrf_time,
                    "num_results": len(fused),
                    "num_of_cross_encoder_candidates": num_of_cross_encoder_candidates,
                    "num_of_dense_chunk": num_of_dense_chunk,
                    "num_of_sparse_chunk": num_of_sparse_chunk,
                    "total_candidates_for_rrf": num_of_dense_chunk + num_of_sparse_chunk,
                    "top_10_rrf_candidates": top_10_rrf_candidates
                },
                "cross_encoder": {
                    "time": cross_encoder_time,
                    "selected_candidates": {
                        "num_candidates": len(all_cross_encoder_documents),
                        "documents": all_cross_encoder_documents
                    },
                    "top_10_cross_encoder_candidates": top_10_cross_encoder_candidates,
                    "top_k_results": {
                        "num_results": len(top_k_documents),
                        "documents": top_k_documents
                    }
                }
            },
            "output": {
                "documents": output_documents,
                "num_results": len(output_documents)
            }
        }
    
    # Step 4: Answer generation (with or without results - Adaptive RAG can answer without context)
    answer_data = {}
    step3_start = time.time()
    # Always call generate_answer - it can handle both cases (with/without context)
    # This aligns with Adaptive RAG philosophy where model can answer using its own knowledge
    # Use model from request if provided, otherwise use config default
    answer_model = req.model if req.model else None
    answer_result = generate_answer(
        query=req.query,  # Use original query for answer generation
        context_chunks=results,  # Can be empty list - model will use its own knowledge
        language=query_language,
        model=answer_model
    )
    answer_data = {
        "answer": answer_result.get("answer", ""),
        "sources": answer_result.get("sources", []),
        "model": answer_result.get("model", "N/A"),
        "provider": answer_result.get("provider", "openai")
    }
    step3_time = round(time.time() - step3_start, 2)
    
    total_time = round(time.time() - total_start_time, 2)
    
    # Prepare preview documents for retrieval step
    preview_documents = []
    for i, result in enumerate(results):
        preview_documents.append({
            "index": i + 1,
            "text": result.get("text", ""),  # Hiển thị toàn bộ chunk, không truncate
            "score": round(result.get("score", 0), 4),
            "metadata": result.get("metadata", {})
        })
    
    # Prepare pipeline steps metadata
    pipeline_steps = {
        "query_rewriting": {
            "time": step1_time,
            "original_query": req.query,
            "rewritten_query": rewritten_query,
            "model": rewriting_result.get("model", "N/A")
        },
        "adaptive_k_selection": {
            "time": step2_time,
            "k": k,
            "average_entropy": adaptive_metadata.get("average_entropy"),
            "k_determined": adaptive_metadata.get("k_determined", k),
            "n": adaptive_metadata.get("n"),
            "iterations_detail": adaptive_metadata.get("iterations_detail", []),
            "k_min": adaptive_metadata.get("k_min"),
            "k_max": adaptive_metadata.get("k_max")
        },
        "retrieval": {
            "time": retrieval_time,
            "num_results": len(results),
            "retrieval_query": rewritten_query,
            "preview_documents": preview_documents,
            "details": retrieval_details
        },
        "answer_generation": {
            "time": step3_time,
            "model": answer_data.get("model", "gpt-4o"),
            "provider": answer_data.get("provider", "openai"),
            "num_chunks_used": len(results)
        },
        "total_time": total_time
    }
    
    return QueryResponse(
        original_query=req.query,
        rewritten_query=rewritten_query,
        k=k,
        entropy=adaptive_metadata.get("entropy"),
        results=results,
        answer=answer_data.get("answer"),
        sources=answer_data.get("sources", []),
        pipeline_steps=pipeline_steps
    )


@app.post("/collections/{name}/query/rewrite-only")
async def rewrite_only(name: str, req: QueryRequest) -> Dict[str, Any]:
    """Only perform query rewriting without retrieval."""
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    coll = collections[name]
    query_language = req.language if req.language else coll.language
    
    rewriting_result = rewrite_query(req.query, language=query_language)
    
    return {
        "original_query": req.query,
        "rewritten_query": rewriting_result.get("rewritten_query", req.query),
        "model": rewriting_result.get("model", "N/A")
    }


@app.get("/models", response_model=List[str])
async def get_available_models() -> List[str]:
    """Get list of available answer generation models."""
    config = get_config()
    return config.llm.available_answer_models


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1012)

