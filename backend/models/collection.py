"""Collection class for managing documents and retrieval."""

from typing import List, Dict, Optional, Tuple, Any, Literal

from .retrievers import BM25Retriever, DenseRetriever
from .reranker import CrossEncoderReranker
from .adaptive_controller import AdaptiveController
from .fusion import reciprocal_rank_fusion
from ..utils.text_processing import semantic_chunk, split_sentences
from ..utils.normalization import Language
from ..config.config import get_config

Language = Literal["en", "vi"]


class Collection:
    """Represents a collection of documents and associated retrieval indices."""

    def __init__(
        self,
        name: str,
        language: Language = "en",
        dense_model_name: Optional[str] = None,
        chunking_model_name: Optional[str] = None,
        similarity_threshold: float = 0.5
    ) -> None:
        self.name = name
        self.language = language
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []
        
        # Get configuration
        config = get_config()
        
        # Initialize retrievers with language support
        self.bm25 = BM25Retriever(language=language)
        
        # Select dense model based on language (language-aware selection)
        if dense_model_name:
            model_name = dense_model_name
        else:
            model_name = config.retrieval.get_dense_model(language)
        self.dense = DenseRetriever(model_name=model_name, language=language)
        
        # Initialize adaptive controller and reranker
        self.controller = AdaptiveController(
            k_min=config.adaptive.k_min,
            k_max=config.adaptive.k_max
        )
        
        # Select reranker model based on language (language-aware selection)
        reranker_model = config.reranker.get_model_name(language)
        self.reranker = CrossEncoderReranker(
            model_name=reranker_model
        )
        
        # Semantic chunking model (use same as dense model for consistency)
        self.chunking_model = None
        self.chunking_model_name = chunking_model_name or config.retrieval.chunking_model_name or model_name
        
        # Use provided similarity_threshold if given, otherwise use language-specific config default
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = config.chunking.get_similarity_threshold(language)

    def _get_chunking_model(self):
        """Lazy load chunking model."""
        if self.chunking_model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self.chunking_model = SentenceTransformer(self.chunking_model_name)
            except Exception:
                self.chunking_model = None
        return self.chunking_model

    def add_documents(
        self,
        docs: List[str],
        meta: List[Dict],
        use_semantic_chunking: bool = True
    ) -> None:
        """
        Add document chunks with associated metadata.
        
        Parameters
        ----------
        docs : List[str]
            List of full document texts (not pre-chunked)
        meta : List[Dict]
            Metadata for each document (one per document, not per chunk)
        use_semantic_chunking : bool
            Whether to use semantic chunking (True) or simple chunking (False)
        """
        all_chunks: List[str] = []
        all_meta: List[Dict] = []
        
        for doc_text, doc_meta in zip(docs, meta):
            if use_semantic_chunking:
                chunking_model = self._get_chunking_model()
                if chunking_model is not None:
                    config = get_config()
                    chunks = semantic_chunk(
                        doc_text,
                        chunking_model,
                        similarity_threshold=self.similarity_threshold,
                        min_chunk_size=config.chunking.min_chunk_size,
                        max_chunk_size=config.chunking.max_chunk_size,
                        language=self.language
                    )
                else:
                    # Fallback to simple sentence-based chunking
                    chunks = split_sentences(doc_text, language=self.language)
            else:
                # Simple sentence-based chunking
                chunks = split_sentences(doc_text, language=self.language)
            
            # Create metadata for each chunk
            for idx, chunk in enumerate(chunks):
                chunk_meta = {
                    **doc_meta,
                    "chunk_index": idx,
                    "num_chunks": len(chunks)
                }
                all_chunks.append(chunk)
                all_meta.append(chunk_meta)
        
        # Add to collection
        self.chunks.extend(all_chunks)
        self.metadata.extend(all_meta)
        
        # Update retrieval indices
        if all_chunks:
            self.bm25.add_documents(all_chunks)
            self.dense.add_documents(all_chunks)

    def query(
        self,
        query: str,
        use_rewritten_query: Optional[str] = None,
        n: Optional[int] = None
    ) -> Tuple[int, List[Dict], Dict]:
        """
        Run the adaptive hybrid retrieval on a query.
        
        Parameters
        ----------
        query : str
            Original user query
        use_rewritten_query : Optional[str]
            If provided, use this for retrieval instead of original query
            
        Returns
        -------
        Tuple[int, List[Dict], Dict]
            (k, results, metadata) where:
            - k: number of documents retrieved
            - results: list of retrieved chunks with scores and metadata
            - metadata: additional info (entropy, etc.)
        """
        if not self.chunks:
            return 0, [], {}
        
        # Use rewritten query for retrieval if provided, otherwise use original
        retrieval_query = use_rewritten_query if use_rewritten_query else query
        
        # Determine how many documents to retrieve (use original query for adaptive k)
        k, adaptive_metadata = self.controller.decide(query, self.language, n=n)
        
        # Skip retrieval if k is 0 - no need to retrieve, fuse, or rerank
        if k == 0:
            return k, [], adaptive_metadata
        
        # Retrieve k documents from both sparse and dense retrievers
        bm25_results = self.bm25.retrieve(retrieval_query, k)
        dense_results = self.dense.retrieve(retrieval_query, k)
        
        # Fuse the results with RRF
        config = get_config()
        fused = reciprocal_rank_fusion([bm25_results, dense_results], k, k_rrf=config.fusion.k_rrf)
        
        # Gather candidate texts and re-rank with cross-encoder
        candidate_indices = [idx for idx, _ in fused]
        candidate_texts = [self.chunks[idx] for idx in candidate_indices]
        rerank_scores = self.reranker.rerank(retrieval_query, candidate_texts)
        
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
                "score": rr_score,  # Unified score field
            }
            results.append(data)
        
        return k, results, adaptive_metadata

