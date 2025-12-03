from __future__ import annotations

import os
import math
from typing import List, Dict, Optional, Tuple, Any
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder  

import docx  # type: ignore

import pypdf  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

def english_normalize(text: str) -> str:
    """
    Lowercase the input string.

    This function provides a minimal normalisation suitable for English
    retrieval.  It only converts text to lowercase to perform case‑
    insensitive matching and deliberately preserves any accented characters.
    """
    return text.lower()

# Override the default normalisation function used throughout this module.
_normalize = english_normalize

class CrossEncoderReranker:
    """
    Light‑weight re‑ranker using a cross‑encoder model.

    Cross‑encoder models score query–document pairs jointly and often
    outperform simple cosine similarity for fine‑grained relevance.  If the
    ``sentence_transformers`` library and a suitable cross‑encoder model are
    available, this class provides a convenient wrapper.  When the
    cross‑encoder cannot be loaded (e.g. missing dependencies), the
    ``available`` flag will be ``False`` and the ``rerank`` method will
    return zero scores for all documents, effectively leaving the RRF
    ordering untouched.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.available: bool = False
        self.model: Optional[CrossEncoder] = None
        # Only attempt to load if dependencies are present
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
            except Exception:
                # Model could not be loaded; remain unavailable
                self.model = None
                self.available = False

    def rerank(self, query: str, docs: List[str]) -> List[float]:
        """
        Compute relevance scores for a list of documents with respect to a query.

        Parameters
        ----------
        query : str
            The user query.
        docs : List[str]
            Candidate documents to score.

        Returns
        -------
        List[float]
            A list of scores, one per document.  Higher is better.  If
            the re‑ranker is not available, all scores will be 0.0.
        """
        if not self.available or self.model is None or not docs:
            return [0.0] * len(docs)
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]


class AdaptiveController:
    """Simple heuristic to determine number of documents based on query length.

    Short queries fetch fewer documents, long queries fetch more.  You can
    adjust ``k_min`` and ``k_max`` to suit your dataset and use a more
    sophisticated strategy if desired (e.g. classification of queries).
    """

    def __init__(self, k_min: int = 5, k_max: int = 20) -> None:
        self.k_min = k_min
        self.k_max = k_max

    def decide(self, query: str) -> int:
        tokens = query.strip().split()
        length = len(tokens)
        if length <= 3:
            return self.k_min
        if length >= 15:
            return self.k_max
        ratio = (length - 3) / (15 - 3)
        k = int(self.k_min + ratio * (self.k_max - self.k_min))
        return max(self.k_min, min(self.k_max, k))


# ``unicodedata`` was used in the Vietnamese pipeline to strip accents.
# In the English‑optimised version we do not need it.


def _normalize(text: str) -> str:
    """Legacy normaliser for compatibility.

    In the English‑optimised pipeline we simply return the lower‑cased text via
    ``english_normalize``.  This placeholder exists to satisfy any calls to
    ``_normalize`` in fallback code paths without altering behaviour.
    """
    return english_normalize(text)


class BM25Retriever:
    """Simple BM25 implementation for sparse retrieval.

    This class builds term frequencies and inverse document frequencies for a set
    of documents and computes BM25 scores for queries.  It is not optimised
    for very large corpora but suffices for moderately sized collections.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avg_doc_len = 0.0
        self.N = 0

    def add_documents(self, docs: List[str]) -> None:
        """Add a list of documents to the index.

        Parameters
        ----------
        docs : List[str]
            List of document strings to index.
        """
        for doc in docs:
            # Normalise document for English retrieval
            norm = english_normalize(doc)
            terms = norm.split()
            tf: Dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            self.documents.append(doc)
            self.doc_freqs.append(tf)
            self.doc_len.append(len(terms))
        self.N = len(self.documents)
        if self.doc_len:
            self.avg_doc_len = sum(self.doc_len) / float(self.N)
        self._calculate_idf()

    def _calculate_idf(self) -> None:
        df: Dict[str, int] = {}
        for tf in self.doc_freqs:
            for term in tf:
                df[term] = df.get(term, 0) + 1
        self.idf = {}
        for term, freq in df.items():
            # add 1 to numerator and denominator to prevent division by zero
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return top-k document indices and scores for a query.

        Returns a list of tuples (doc_index, score) sorted by score
        descending.
        """
        # Normalise query for English retrieval
        query_terms = english_normalize(query).split()
        scores = np.zeros(self.N)
        for term in query_terms:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.doc_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_doc_len)
                scores[i] += idf * (numerator / denominator)
        # Get top-k indices
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_idx]


class DenseRetriever:
    """Dense retriever with optional SentenceTransformer fallback to TF‑IDF.

    If the `sentence_transformers` library is available and a model name is
    provided, this class will compute dense embeddings using that model and
    normalise them for cosine similarity.  Otherwise it falls back to using
    a TF‑IDF vectoriser for both indexing and querying.
    """

    def __init__(self, model_name: str = "sentence-transformers/colbert-distilroberta-v1") -> None:
        self.use_sentence_transformer = SentenceTransformer is not None
        if self.use_sentence_transformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
                self.use_sentence_transformer = False
        else:
            self.model = None
        # Fallback to TF‑IDF
        if not self.use_sentence_transformer:
            self.vectorizer: Optional[TfidfVectorizer] = TfidfVectorizer()
        else:
            self.vectorizer = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]) -> None:
        """Add documents and compute embeddings.

        Parameters
        ----------
        docs : List[str]
            Document strings to embed and index.
        """
        # Normalise documents for dense retrieval.  Even though the dense model
        # generally handles casing internally, normalising makes the TF‑IDF
        # fallback more robust to accents.
        norm_docs = [english_normalize(doc) for doc in docs]
        start_idx = len(self.documents)
        self.documents.extend(norm_docs)
        if self.use_sentence_transformer and self.model is not None:
            new_embeds = self.model.encode(norm_docs, batch_size=16, show_progress_bar=False)
            new_embeds = normalize(new_embeds)
            if self.embeddings is None:
                self.embeddings = new_embeds
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeds])
        else:
            # Fit or update TF‑IDF vectoriser
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer()
            # When adding new documents, refit vectoriser on all documents
            # This is not ideal for dynamic updates but acceptable for demonstration
            self.vectorizer.fit(self.documents)
            self.embeddings = self.vectorizer.transform(self.documents).toarray()

    def _embed_query(self, query: str) -> np.ndarray:
        if self.use_sentence_transformer and self.model is not None:
            # Normalise the query before encoding.  SentenceTransformer models
            # typically lower‑case internally, but this normalisation also
            # removes accents for languages like Vietnamese.
            vec = self.model.encode([english_normalize(query)], show_progress_bar=False)
            return normalize(vec)
        # Fall back to TF‑IDF
        assert self.vectorizer is not None
        return normalize(self.vectorizer.transform([english_normalize(query)]).toarray())

    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self.documents:
            return []
        q_emb = self._embed_query(query)
        # Cosine similarity via dot product on normalised vectors
        sims = np.dot(self.embeddings, q_emb.T).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        return [(int(idx), float(sims[idx])) for idx in top_idx]


    # NOTE: CrossEncoderReranker has been moved to module level.  This nested
    # definition is retained for backward compatibility but is unused.
    class CrossEncoderReranker:
        pass


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]], final_k: int, k_rrf: int = 60) -> List[Tuple[int, float]]:
    """Fuse multiple ranked lists of document indices using RRF.

    Parameters
    ----------
    ranked_lists : List[List[Tuple[int, float]]]
        Each inner list must be sorted by decreasing relevance.
    final_k : int
        Number of items to return after fusion.
    k_rrf : int
        Smoothing parameter controlling decay of rank contribution.

    Returns
    -------
    List[Tuple[int, float]]
        Fused list of (doc_index, score) pairs sorted by score descending.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            contrib = 1.0 / (k_rrf + rank + 1)
            scores[idx] = scores.get(idx, 0.0) + contrib
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:final_k]


def chunk_text(text: str, max_tokens: int = 128) -> List[str]:
    """Split text into chunks of up to ``max_tokens`` whitespace tokens.

    Parameters
    ----------
    text : str
        The input document text.
    max_tokens : int
        Maximum number of whitespace tokens per chunk.

    Returns
    -------
    List[str]
        A list of chunk strings.
    """
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_tokens:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


class Collection:
    """Represents a collection of documents and associated retrieval indices."""

    def __init__(self, name: str, dense_model_name: str | None = None) -> None:
        self.name = name
        self.chunks: List[str] = []  # each chunk is a piece of text
        self.metadata: List[Dict] = []  # metadata per chunk, e.g. original file and chunk index
        self.bm25 = BM25Retriever()
        model_name = dense_model_name or "sentence-transformers/colbert-distilroberta-v1"
        self.dense = DenseRetriever(model_name)
        self.controller = AdaptiveController()
        self.reranker = CrossEncoderReranker()

    def add_documents(self, docs: List[str], meta: List[Dict]) -> None:
        """Add document chunks with associated metadata."""
        self.chunks.extend(docs)
        self.metadata.extend(meta)
        self.bm25.add_documents(docs)
        self.dense.add_documents(docs)

    def query(self, query: str) -> Tuple[int, List[Dict]]:
        """Run the adaptive hybrid retrieval on a query.

        Returns a tuple ``(k, results)`` where ``k`` is the number of documents
        determined by the adaptive controller and ``results`` is a list of
        retrieved chunks with their scores and metadata.
        """
        # If there are no documents, return 0 and an empty list
        if not self.chunks:
            return 0, []
        # Determine how many documents to retrieve
        k = self.controller.decide(query)
        # Retrieve k documents from both sparse and dense retrievers
        bm25_results = self.bm25.retrieve(query, k)
        dense_results = self.dense.retrieve(query, k)
        # Fuse the results with RRF using the same k as final_k
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k)
        # Gather candidate texts and re‑rank with cross‑encoder
        candidate_indices = [idx for idx, _ in fused]
        candidate_texts = [self.chunks[idx] for idx in candidate_indices]
        rerank_scores = self.reranker.rerank(query, candidate_texts)
        # Combine RRF and reranker scores
        combined = [
            (idx, rrf_score, rr_score)
            for (idx, rrf_score), rr_score in zip(fused, rerank_scores)
        ]
        combined.sort(key=lambda x: x[2], reverse=True)
        results: List[Dict] = []
        for idx, rrf_score, rr_score in combined[:k]:
            data = {
                "text": self.chunks[idx],
                "metadata": self.metadata[idx],
                "rrf_score": rrf_score,
                "rerank_score": rr_score,
                # Unified 'score' field for frontend compatibility
                "score": rr_score,
            }
            results.append(data)
        return k, results


def extract_text_from_pdf(file_data: bytes) -> str:
    """Extract text from a PDF file using PyPDF2 if available."""
    if pypdf is None:
        raise RuntimeError("pypdf is not installed")
    reader = pypdf.PdfReader(BytesIO(file_data))
    texts = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def extract_text_from_docx(file_data: bytes) -> str:
    """Extract text from a DOCX file using python-docx if available."""
    if docx is None:
        raise RuntimeError("python-docx is not installed")
    from docx import Document as WordDocument  # type: ignore
    # Write to a temporary file because python-docx expects a path
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_data)
        tmp.flush()
        doc = WordDocument(tmp.name)
        text = "\n".join([p.text for p in doc.paragraphs])
        os.unlink(tmp.name)
    return text


def read_file_to_text(file: UploadFile) -> str:
    """Read an uploaded file and extract its text."""
    data = file.file.read()
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(data)
    if ext in (".docx", ".doc"):
        return extract_text_from_docx(data)
    # For plain text or unknown formats, assume utf-8 plain text
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin1", errors="ignore")


# Global store of collections (in-memory for demonstration)
collections: Dict[str, Collection] = {}


class CollectionCreateRequest(BaseModel):
    name: str

class QueryRequest(BaseModel):
    query: str


app = FastAPI()


@app.post("/collections")
async def create_collection(req: CollectionCreateRequest) -> Dict[str, str]:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Collection name cannot be empty")
    if name in collections:
        raise HTTPException(status_code=400, detail="Collection already exists")
    collections[name] = Collection(name)
    return {"message": f"Collection '{name}' created"}


@app.delete("/collections/{name}")
async def delete_collection(name: str) -> Dict[str, str]:
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    del collections[name]
    return {"message": f"Collection '{name}' deleted"}


@app.get("/collections")
async def list_collections() -> List[str]:
    return list(collections.keys())


@app.post("/collections/{name}/documents")
async def upload_documents(name: str, files: List[UploadFile] = File(...)) -> Dict[str, str]:
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    coll = collections[name]
    all_chunks: List[str] = []
    all_meta: List[Dict] = []
    for f in files:
        try:
            text = read_file_to_text(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read '{f.filename}': {e}")
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": f.filename, "chunk_index": idx})
    if all_chunks:
        coll.add_documents(all_chunks, all_meta)
    return {"message": f"Uploaded {len(files)} documents and added {len(all_chunks)} chunks"}


@app.post("/collections/{name}/query")
async def run_query(name: str, req: QueryRequest) -> Dict[str, Any]:
    """
    Execute a query against the specified collection using adaptive retrieval.

    Returns a JSON object with two keys:
    - ``k``: the number of documents selected by the adaptive controller.
    - ``results``: the list of retrieved document chunks with scores and metadata.
    """
    if name not in collections:
        raise HTTPException(status_code=404, detail="Collection not found")
    coll = collections[name]
    k, results = coll.query(req.query)
    return {"k": k, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)