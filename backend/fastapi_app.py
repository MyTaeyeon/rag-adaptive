"""
backend/fastapi_app.py
Minimal FastAPI demo that wires the adaptive controller and retrieval templates.
Endpoints:
 - POST /query  { "question": "..." }
 - POST /upload_corpus  (multipart file)  -- template
 - GET  /status
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import time
from scripts.adaptive_controller import heuristic_k
from scripts.retrieve_hybrid import hybrid_retrieve

app = FastAPI(title="Adaptive RAG Demo")

class QueryRequest(BaseModel):
    question: str
    topk_hint: int = 0

@app.get("/status")
def status():
    return {"status":"ok", "note":"Demo backend running (template)"}

@app.post("/query")
def run_query(req: QueryRequest):
    t0 = time.time()
    ctrl = heuristic_k(req.question)
    k = ctrl["k"] if req.topk_hint == 0 else req.topk_hint
    # corpus path is fixed to samples/corpus.tsv in this demo
    retrieval = hybrid_retrieve(req.question, "samples/corpus.tsv", topk=k)
    # For demo, we don't call an LLM here; we return trace + top-5 merged ids
    result = {
        "question": req.question,
        "controller": ctrl,
        "retrieval": retrieval,
        "elapsed_seconds": round(time.time() - t0, 3)
    }
    return result
