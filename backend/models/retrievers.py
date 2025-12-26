"""Retrieval models: BM25 and Dense retrievers with language support."""

import math
from typing import List, Dict, Optional, Tuple, Literal

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..utils.normalization import normalize_text, Language


class BM25Retriever:
    """Simple BM25 implementation for sparse retrieval with language support."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        language: Language = "en"
    ) -> None:
        self.k1 = k1
        self.b = b
        self.language = language
        self.documents: List[str] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avg_doc_len = 0.0
        self.N = 0

    def add_documents(self, docs: List[str]) -> None:
        """Add a list of documents to the index."""
        for doc in docs:
            # Normalise document based on language
            norm = normalize_text(doc, self.language)
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
            self.idf[term] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)

    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Return top-k document indices and scores for a query."""
        query_terms = normalize_text(query, self.language).split()
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
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_idx]


class DenseRetriever:
    """Dense retriever with optional SentenceTransformer fallback to TF-IDF."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/colbert-distilroberta-v1",
        language: Language = "en"
    ) -> None:
        self.language = language
        self.use_sentence_transformer = SentenceTransformer is not None
        if self.use_sentence_transformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception:
                self.model = None
                self.use_sentence_transformer = False
        else:
            self.model = None
        
        if not self.use_sentence_transformer:
            self.vectorizer: Optional[TfidfVectorizer] = TfidfVectorizer()
        else:
            self.vectorizer = None
        self.embeddings: Optional[np.ndarray] = None
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]) -> None:
        """Add documents and compute embeddings."""
        norm_docs = [normalize_text(doc, self.language) for doc in docs]
        self.documents.extend(norm_docs)
        
        if self.use_sentence_transformer and self.model is not None:
            new_embeds = self.model.encode(
                norm_docs,
                batch_size=16,
                show_progress_bar=False
            )
            new_embeds = normalize(new_embeds)
            if self.embeddings is None:
                self.embeddings = new_embeds
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeds])
        else:
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(self.documents)
            self.embeddings = self.vectorizer.transform(self.documents).toarray()

    def _embed_query(self, query: str) -> np.ndarray:
        norm_query = normalize_text(query, self.language)
        if self.use_sentence_transformer and self.model is not None:
            vec = self.model.encode([norm_query], show_progress_bar=False)
            return normalize(vec)
        assert self.vectorizer is not None
        return normalize(self.vectorizer.transform([norm_query]).toarray())

    def retrieve(self, query: str, k: int) -> List[Tuple[int, float]]:
        if not self.documents:
            return []
        q_emb = self._embed_query(query)
        sims = np.dot(self.embeddings, q_emb.T).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        return [(int(idx), float(sims[idx])) for idx in top_idx]

