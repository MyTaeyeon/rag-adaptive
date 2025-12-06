"""Fusion methods for combining retrieval results."""

from typing import List, Dict, Tuple, Optional
from ..config.config import get_config


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    final_k: int,
    k_rrf: Optional[int] = None
) -> List[Tuple[int, float]]:
    """
    Fuse multiple ranked lists of document indices using RRF.
    
    Parameters
    ----------
    ranked_lists : List[List[Tuple[int, float]]]
        Each inner list must be sorted by decreasing relevance.
    final_k : int
        Number of items to return after fusion.
    k_rrf : Optional[int]
        Smoothing parameter controlling decay of rank contribution.
        If None, uses config default.
        
    Returns
    -------
    List[Tuple[int, float]]
        Fused list of (doc_index, score) pairs sorted by score descending.
    """
    if k_rrf is None:
        config = get_config()
        k_rrf = config.fusion.k_rrf
    
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            contrib = 1.0 / (k_rrf + rank + 1)
            scores[idx] = scores.get(idx, 0.0) + contrib
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:final_k]

