"""Reranker model using cross-encoder."""

from typing import List, Optional
from sentence_transformers import CrossEncoder
from ..config.config import get_config


class CrossEncoderReranker:
    """
    Light-weight re-ranker using a cross-encoder model.
    
    Cross-encoder models score query–document pairs jointly and often
    outperform simple cosine similarity for fine-grained relevance.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        config = get_config()
        model_name = model_name or config.reranker.model_name
        
        self.available: bool = False
        self.model: Optional[CrossEncoder] = None
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
                self.available = True
            except Exception:
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
            A list of scores, one per document. Higher is better.
            If the re-ranker is not available, all scores will be 0.0.
        """
        if not self.available or self.model is None or not docs:
            return [0.0] * len(docs)
        pairs = [[query, d] for d in docs]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

