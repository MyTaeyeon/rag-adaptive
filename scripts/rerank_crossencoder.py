"""
rerank_crossencoder.py

Template for reranking top-k passages using a cross-encoder (e.g., sentence-transformers cross-encoder models).
"""
from sentence_transformers import CrossEncoder

def rerank(query, passages, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    model = CrossEncoder(model_name)
    inputs = [[query, p] for p in passages]
    scores = model.predict(inputs)
    pairs = list(zip(passages, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _s in pairs]

if __name__ == "__main__":
    q = "what is reciprocal rank fusion?"
    ps = ["RRF is ...", "Some other doc ...", "BM25 explanation ..."]
    print(rerank(q, ps))
