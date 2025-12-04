"""
retrieve_hybrid.py
Template for hybrid retrieval:
 - BM25 (Pyserini / Elastic)
 - Dense (ColBERTv2 / SentenceTransformers + FAISS)
 - Merge with RRF

This script provides a minimal TF-IDF fallback so you can run basic local experiments without heavy deps.
"""
import json
from typing import List, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from scripts.rrf_utils import rrf_merge

# Fallback lightweight retriever using TF-IDF on the provided corpus (samples/corpus.tsv)
class SimpleTfIdfRetriever:
    def __init__(self, docs: List[dict]):
        self.ids = [d["id"] for d in docs]
        self.passages = [d.get("text", d.get("passage","")) for d in docs]
        self.vec = TfidfVectorizer(stop_words="english", max_features=20000)
        if self.passages:
            self.mat = self.vec.fit_transform(self.passages)
        else:
            self.mat = None

    def retrieve(self, query: str, topk=10) -> List[Tuple[str, float]]:
        if self.mat is None:
            return []
        qv = self.vec.transform([query])
        scores = linear_kernel(qv, self.mat).flatten()
        idx = np.argsort(scores)[::-1][:topk]
        return [(self.ids[i], float(scores[i])) for i in idx]

def load_corpus(path: str):
    docs = []
    p = Path(path)
    if p.suffix == ".tsv":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    docs.append({"id": parts[0], "title": parts[1], "text": parts[2]})
    elif p.suffix in (".json", ".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            docs = json.load(f)
    return docs

def hybrid_retrieve(query: str, corpus_path: str, topk=10):
    # run TF-IDF as BM25 proxy (local small experiments)
    docs = load_corpus(corpus_path)
    tfidf = SimpleTfIdfRetriever(docs)
    hits_tfidf = [docid for docid, score in tfidf.retrieve(query, topk=topk)]
    # dense retrieval placeholder (use sentence-transformers or ColBERT in real runs)
    # For now, we'll reuse tfidf with different k to simulate diversity
    hits_dense = [docid for docid, score in tfidf.retrieve(query, topk=topk*2)][::2]
    merged = rrf_merge([hits_tfidf, hits_dense], rrf_k=60, topk=topk)
    return {"bm25": hits_tfidf, "dense": hits_dense, "merged": merged}

if __name__ == "__main__":
    out = hybrid_retrieve("what is reciprocal rank fusion", "samples/corpus.tsv", topk=5)
    print(json.dumps(out, indent=2))
