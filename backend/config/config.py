"""
Configuration file for RAG Adaptive System.

This file contains all configurable parameters for the RAG system.
Modify values here to adjust system behavior without changing code.

All parameters are organized into logical groups:
- LLMConfig: Settings for OpenAI LLM (query rewriting, answer generation)
- RetrievalConfig: Settings for document retrieval models
- ChunkingConfig: Settings for semantic chunking
- AdaptiveConfig: Settings for adaptive k selection
- RerankerConfig: Settings for cross-encoder reranker
- FusionConfig: Settings for result fusion
- APIConfig: Default API settings
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class LLMConfig:
    """
    Configuration for LLM services (OpenAI GPT models).
    
    Used for:
    - Query rewriting: Rewrites user queries to improve retrieval
    - Answer generation: Generates final answers from retrieved context
    """
    
    # Model names
    # OpenAI model to use for query rewriting
    # Options: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", etc.
    # query_rewrite_model: str = "gpt-4o"
    query_rewrite_model: str = "gpt-4.1-nano-2025-04-14"
    
    
    # OpenAI model to use for answer generation
    answer_generation_model: str = "gpt-4o-mini-2024-07-18"
    
    # Available models for answer generation (used by API and frontend)
    # Older model 4.
    # Reasoning model 5.
    available_answer_models: List[str] = field(default_factory=lambda: [
        # Older model 4.
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4o-mini-2024-07-18",
        # Reasoning model 5.
        "gpt-5.1-2025-11-13",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
    ])
    
    def is_gpt5_model(self, model_name: str) -> bool:
        """Check if model is a GPT-5 model (reasoning model)."""
        return model_name.startswith("gpt-5")
    
    # Query Rewriting Parameters
    # Temperature controls randomness in query rewriting
    # Lower values (0.1-0.3): More focused, deterministic rewrites
    # Higher values (0.7-1.0): More creative, varied rewrites
    query_rewrite_temperature: float = 0.3
    
    # Maximum tokens for query rewriting response
    # Typical queries need 50-150 tokens, 200 provides buffer
    query_rewrite_max_tokens: int = 200
    
    # Answer Generation Parameters
    # Temperature for answer generation
    # Lower values (0.5-0.7): More factual, consistent answers
    # Higher values (0.8-1.0): More creative, varied answers
    answer_generation_temperature: float = 0.7
    
    # Maximum tokens for answer generation
    # Adjust based on expected answer length
    # 1000 tokens ≈ 750 words
    # For GPT-5.1 models with extended context, can use higher values
    # Note: If you see "finish_reason: length" errors, increase this value
    answer_generation_max_tokens: int = 4000  # Increased from 1000 to handle longer responses


@dataclass
class RetrievalConfig:
    """
    Configuration for document retrieval models.
    
    Used for:
    - Dense retrieval: Semantic similarity search using embeddings
    - Sparse retrieval: Keyword-based search using BM25
    
    Supports language-specific models for optimal performance.
    """
    
    # Dense Retrieval Models (Language-specific)
    # English: High-quality model optimized for English
    # Recommended: "sentence-transformers/all-mpnet-base-v2" (best quality)
    # Alternative: "sentence-transformers/all-MiniLM-L6-v2" (faster)
    dense_model_name_en: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Vietnamese: Specialized Vietnamese embedding model
    # Recommended: "keepitreal/vietnamese-sbert-v2" (best for Vietnamese)
    # Alternatives: "dangvantuan/vietnamese-embedding", "bge-base-zh-v1.5"
    dense_model_name_vi: str = "keepitreal/vietnamese-sbert-v2"
    
    # Legacy: Default model (used if language-specific not available)
    # Will be replaced by language-specific selection
    dense_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Chunking Model
    # Model used for semantic chunking (can be same as dense_model_name)
    # If None, will use language-specific dense_model_name
    chunking_model_name: Optional[str] = None
    
    # Default Similarity Threshold
    # Used for filtering retrieval results
    # Range: 0.0 to 1.0
    # Lower values: More results, may include less relevant docs
    # Higher values: Fewer results, more relevant docs
    default_similarity_threshold: float = 0.4
    
    def get_dense_model(self, language: str = "en") -> str:
        """Get dense model name based on language."""
        if language == "vi":
            return self.dense_model_name_vi
        else:
            return self.dense_model_name_en


@dataclass
class ChunkingConfig:
    """
    Configuration for semantic chunking of documents.
    
    Semantic chunking splits documents based on semantic similarity
    between sentences, creating more coherent chunks than fixed-size splitting.
    
    Supports language-specific thresholds for optimal chunking.
    Measured using len(string), so this is character count, not token count.

    """
    

    # Similarity Threshold for Semantic Chunking (Language-specific)
    # Range: 0.0 to 1.0
    similarity_threshold_en: float = 0.5
    
    # Vietnamese: May need slightly lower threshold due to different sentence structure
    similarity_threshold_vi: float = 0.5
    
    # Legacy: Default threshold (used if language-specific not available)
    similarity_threshold: float = 0.5
    
    # Minimum Chunk Size
    # Minimum number of characters per chunk
    min_chunk_size: int = 32
    
    # Maximum Chunk Size
    # Maximum number of characters per chunk
    # n*3 char ~ n chunk
    max_chunk_size: int = 512
    
    # Embedding Batch Size
    # Number of sentences to process in parallel when computing embeddings
    # Recommended: 16-64 depending on GPU memory
    embedding_batch_size: int = 32
    
    def get_similarity_threshold(self, language: str = "en") -> float:
        """Get similarity threshold based on language."""
        if language == "vi":
            return self.similarity_threshold_vi
        else:
            return self.similarity_threshold_en


@dataclass
class AdaptiveConfig:
    """
    Configuration for adaptive k selection using entropy-based method.
    
    Adaptive k selection uses multiple LLM calls with different styles to calculate
    entropy and determine optimal number of documents to retrieve.
    """
    
    # Minimum K Value
    # Minimum number of documents to retrieve
    k_min: int = 0
    
    # Maximum K Value
    # Maximum number of documents to retrieve
    k_max: int = 10
    
    # Default number of iterations (n)
    # Number of non-hop LLM calls to calculate entropy
    # Range: 1-10
    default_n: int = 3
    
    # Style candidates for adaptive analysis (Vietnamese)
    # 10 different response styles to test model uncertainty
    # Designed to create slightly different responses without extreme entropy outliers
    style_candidates_vi: List[str] = field(default_factory=lambda: [
        "Trả lời câu hỏi một cách ngắn gọn và rõ ràng",
        "Trả lời câu hỏi một cách ngắn gọn, sử dụng ngôn ngữ tự nhiên",
        "Trả lời câu hỏi một cách ngắn gọn với cách diễn đạt đơn giản",
        "Trả lời câu hỏi một cách ngắn gọn, tập trung vào thông tin chính",
        "Trả lời câu hỏi một cách ngắn gọn với cấu trúc câu rõ ràng",
        "Trả lời câu hỏi một cách ngắn gọn, sử dụng từ ngữ phù hợp",
        "Trả lời câu hỏi một cách ngắn gọn với cách trình bày logic",
        "Trả lời câu hỏi một cách ngắn gọn, đảm bảo tính chính xác",
        "Trả lời câu hỏi một cách ngắn gọn với cách diễn đạt dễ hiểu",
        "Trả lời câu hỏi một cách ngắn gọn, cung cấp thông tin cần thiết"
    ])
    
    # Style candidates for adaptive analysis (English)
    # 10 different response styles to test model uncertainty
    # Designed to create slightly different responses without extreme entropy outliers
    style_candidates_en: List[str] = field(default_factory=lambda: [
        "Answer briefly and clearly",
        "Answer briefly using natural language",
        "Answer briefly with simple expression",
        "Answer briefly, focusing on key information",
        "Answer briefly with clear sentence structure",
        "Answer briefly using appropriate vocabulary",
        "Answer briefly with logical presentation",
        "Answer briefly, ensuring accuracy",
        "Answer briefly with easy-to-understand expression",
        "Answer briefly, providing necessary information"
    ])
    # Phase 1 Model (for adaptive analysis)
    # Model used for non-hop calls to calculate entropy
    # Must support logprobs parameter
    # Options (fastest to slowest):
    # - "gpt-4o-mini": Fastest, cheapest, recommended for speed
    # - "gpt-3.5-turbo": Very fast, cheaper alternative
    # - "gpt-4-turbo": Faster than gpt-4o, better quality than mini
    # - "gpt-4o": Slower but highest quality (original default)
    phase1_model: str = "gpt-4o"
    # phase1_model: str = "gpt-4o-mini"
    
    # Phase 1 Temperature
    # Temperature for adaptive analysis calls
    phase1_temperature: float = 0.01
    
    # Phase 1 Max Tokens
    # Maximum tokens for adaptive analysis responses
    phase1_max_tokens: int = 64
    
    # Entropy Maximum Value
    # Maximum expected entropy value for normalization
    # Entropy values above this will be clamped to 1.0
    # Typical range: 0.2-0.5 bits/token for GPT models
    # Lower values: System will retrieve more documents (more conservative)
    # Higher values: System will retrieve fewer documents (more aggressive)
    entropy_max: float = 0.5


@dataclass
class RerankerConfig:
    """
    Configuration for cross-encoder reranker.
    
    Reranker improves retrieval quality by scoring query-document pairs
    more accurately than simple cosine similarity.
    
    Supports language-specific models for optimal performance.
    """
    
    # Reranker Model Names (Language-specific)
    # English: Higher quality model for English
    # Recommended: "cross-encoder/ms-marco-MiniLM-L-12-v2" (better quality than L-6)
    model_name_en: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Vietnamese: Multilingual model that supports Vietnamese
    # Recommended: "cross-encoder/ms-marco-MiniLM-L-12-v2" (multilingual support)
    # Alternative: "bge-reranker-base" (if available, better for Vietnamese)
    model_name_vi: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    # Legacy: Default model (used if language-specific not available)
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    def get_model_name(self, language: str = "en") -> str:
        """Get reranker model name based on language."""
        if language == "vi":
            return self.model_name_vi
        else:
            return self.model_name_en


@dataclass
class FusionConfig:
    """
    Configuration for result fusion methods.
    
    Fusion combines results from multiple retrieval methods (BM25, Dense)
    to improve overall retrieval quality.
    """
    
    # Reciprocal Rank Fusion (RRF) Parameter
    # Smoothing parameter controlling how rank contributes to final score
    # Higher values (60-100): More smoothing, less difference between ranks
    # Lower values (20-40): Less smoothing, more emphasis on top ranks
    # Recommended: 60 for balanced fusion
    k_rrf: int = 40
    
    # Number of documents to retrieve from each method before RRF fusion
    # These are multipliers based on adaptive k value
    # Formula: num_of_dense_chunk = dense_chunk_multiplier * k_adaptive
    # Formula: num_of_sparse_chunk = sparse_chunk_multiplier * k_adaptive
    # Higher values: More candidates for RRF, potentially better quality but slower
    # Lower values: Fewer candidates, faster but may miss good documents
    # Recommended: 20-50 for balanced trade-off between accuracy and latency
    dense_chunk_multiplier: float = 30.0
    sparse_chunk_multiplier: float = 30.0
    
    # Number of candidates to select from RRF for cross-encoder reranking
    # Formula: num_of_cross_encoder_candidates = cross_encoder_multiplier * k_adaptive
    # These candidates will be reranked by cross-encoder, then top k will be selected
    # Higher values: More candidates reranked, potentially better quality but slower
    # Lower values: Fewer candidates reranked, faster but may miss good documents
    # Recommended: 3-10 for balanced trade-off
    cross_encoder_multiplier: float = 10.0


@dataclass
class APIConfig:
    """
    Default configuration for API endpoints.
    
    These are default values used when creating collections or making queries
    if not explicitly specified in the request.
    """
    
    # Default Language
    # Default language for collections and queries
    # Options: "en" (English), "vi" (Vietnamese)
    default_language: str = "en"
    
    # Default Similarity Threshold
    # Default similarity threshold for new collections
    # Can be overridden when creating a collection
    default_similarity_threshold: float = 0.5
    
    # Default Dense Model Name
    # Default dense model for new collections
    # Can be overridden when creating a collection
    default_dense_model_name: Optional[str] = None
    
    # Default Chunking Model Name
    # Default chunking model for new collections
    # Can be overridden when creating a collection
    default_chunking_model_name: Optional[str] = None


@dataclass
class RAGConfig:
    """
    Main configuration class containing all sub-configurations.
    
    This is the primary configuration object used throughout the system.
    Import and use this class to access all configuration parameters.
    
    Example:
        from backend.config.config import RAGConfig
        
        config = RAGConfig()
        model_name = config.llm.query_rewrite_model
        k_min = config.adaptive.k_min
    """
    
    def __init__(self):
        """Initialize all configuration sub-modules."""
        self.llm = LLMConfig()
        self.retrieval = RetrievalConfig()
        self.chunking = ChunkingConfig()
        self.adaptive = AdaptiveConfig()
        self.reranker = RerankerConfig()
        self.fusion = FusionConfig()
        self.api = APIConfig()


# Global configuration instance
# Import this to use configuration throughout the application
_config_instance: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """
    Get the global configuration instance.
    
    Returns
    -------
    RAGConfig
        The global configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig()
    return _config_instance


def reload_config() -> RAGConfig:
    """
    Reload configuration (creates new instance).
    
    Useful for testing or when configuration needs to be refreshed.
    
    Returns
    -------
    RAGConfig
        New configuration instance
    """
    global _config_instance
    _config_instance = RAGConfig()
    return _config_instance

