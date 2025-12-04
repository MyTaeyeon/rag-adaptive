"""
rrf_utils.py
Implement Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
"""
from collections import defaultdict
from typing import List, Dict

def rrf_merge(ranked_lists: List[List[str]], rrf_k: int = 60, topk: int = 100):
    scores = defaultdict(float)
    for lst in ranked_lists:
        for rank, docid in enumerate(lst, start=1):
            scores[docid] += 1.0 / (rrf_k + rank)
    # sort by score desc
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docid for docid, _score in items][:topk]

if __name__ == "__main__":
    a = ["d1","d2","d3","d4"]
    b = ["d3","d2","d5","d6"]
    print(rrf_merge([a,b], rrf_k=60, topk=10))
