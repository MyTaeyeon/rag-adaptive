"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal

Language = Literal["en", "vi"]


class CollectionCreateRequest(BaseModel):
    name: str
    language: Language = "en"
    dense_model_name: Optional[str] = None
    chunking_model_name: Optional[str] = None
    similarity_threshold: Optional[float] = None  # If None, uses config default


class QueryRequest(BaseModel):
    query: str
    language: Optional[Language] = None  # Override collection language if provided
    n: Optional[int] = None  # Number of adaptive iterations (1-10)
    model: Optional[str] = None  # Model name for answer generation
    session_id: Optional[str] = None  # Session ID for conversation history and memory


class QueryResponse(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    k: int
    entropy: Optional[float] = None
    results: List[Dict[str, Any]]
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    pipeline_steps: Dict[str, Any] = {}
    session_id: Optional[str] = None  # Session ID for conversation continuity


class UploadResponse(BaseModel):
    message: str
    num_files: int
    num_chunks: int

