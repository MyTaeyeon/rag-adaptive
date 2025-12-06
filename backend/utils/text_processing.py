"""Text processing utilities for sentence splitting and chunking."""

import re
from typing import List, Tuple, Optional
from typing import Literal
from ..config.config import get_config

Language = Literal["en", "vi"]


def split_sentences(text: str, language: Language = "en") -> List[str]:
    """
    Split text into sentences based on language.
    
    Parameters
    ----------
    text : str
        Input text to split
    language : Language
        Language code: "en" for English, "vi" for Vietnamese
        
    Returns
    -------
    List[str]
        List of sentences
    """
    if language == "vi":
        # Vietnamese sentence splitting
        # Common sentence endings: . ! ? and Vietnamese-specific patterns
        # Remove dots in numbers, dates, abbreviations
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĂÂÊÔƠƯĐẠẢÃÀÁẲẬẮẰẴẶẸẺẼÈÉÊỂỄỆỌỎÕÒÓỒỐỔỖỘỜỞỠỢỤỦŨÙÚỨỪỬỮỰỲỴỶỸÝ])', text)
    else:
        # English sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def semantic_chunk(
    text: str,
    embedding_model,
    similarity_threshold: Optional[float] = None,
    min_chunk_size: Optional[int] = None,
    max_chunk_size: Optional[int] = None,
    language: Language = "en"
) -> List[str]:
    """
    Split text into chunks based on semantic similarity between sentences.
    
    This function:
    1. Splits text into sentences
    2. Computes embeddings for each sentence
    3. Calculates semantic similarity between consecutive sentences
    4. Groups sentences into chunks when similarity is above threshold
    5. Splits chunks when similarity is below threshold
    
    Parameters
    ----------
    text : str
        Input text to chunk
    embedding_model
        SentenceTransformer model or similar for computing embeddings
    similarity_threshold : Optional[float]
        Threshold for semantic similarity (0.0 to 1.0)
        Above threshold: merge sentences into same chunk
        Below threshold: split into different chunks
        If None, uses config default.
    min_chunk_size : Optional[int]
        Minimum number of characters per chunk
        If None, uses config default.
    max_chunk_size : Optional[int]
        Maximum number of characters per chunk
        If None, uses config default.
    language : Language
        Language code for sentence splitting
        
    Returns
    -------
    List[str]
        List of semantic chunks
    """
    config = get_config()
    similarity_threshold = similarity_threshold if similarity_threshold is not None else config.chunking.similarity_threshold
    min_chunk_size = min_chunk_size if min_chunk_size is not None else config.chunking.min_chunk_size
    max_chunk_size = max_chunk_size if max_chunk_size is not None else config.chunking.max_chunk_size
    
    # Split into sentences
    sentences = split_sentences(text, language=language)
    
    if not sentences:
        return []
    
    if len(sentences) == 1:
        return sentences
    
    # Compute embeddings for all sentences
    try:
        sentence_embeddings = embedding_model.encode(
            sentences,
            batch_size=config.chunking.embedding_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    except Exception:
        # Fallback: simple chunking by size if embedding fails
        return _fallback_chunk_by_size(text, max_chunk_size)
    
    # Compute similarity matrix between consecutive sentences
    chunks: List[str] = []
    current_chunk: List[str] = []
    
    for i in range(len(sentences) - 1):
        current_chunk.append(sentences[i])
        
        # Calculate cosine similarity between consecutive sentences
        similarity = _cosine_similarity(
            sentence_embeddings[i],
            sentence_embeddings[i + 1]
        )
        
        current_text = " ".join(current_chunk)
        current_size = len(current_text)
        
        # Decision logic:
        # 1. If similarity below threshold AND current chunk is large enough -> split
        # 2. If similarity above threshold -> continue adding to current chunk
        # 3. If chunk exceeds max size -> split regardless of similarity
        
        if similarity < similarity_threshold and current_size >= min_chunk_size:
            # Split: save current chunk and start new one
            chunks.append(current_text)
            current_chunk = []
        elif current_size >= max_chunk_size:
            # Force split if chunk too large
            chunks.append(current_text)
            current_chunk = [sentences[i + 1]] if i + 1 < len(sentences) else []
            continue
    
    # Add the last sentence
    if current_chunk:
        current_chunk.append(sentences[-1])
        chunks.append(" ".join(current_chunk))
    else:
        chunks.append(sentences[-1])
    
    # Post-processing: merge very small chunks with neighbors
    chunks = _merge_small_chunks(chunks, min_chunk_size)
    
    return chunks


def _cosine_similarity(vec1, vec2) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    except Exception:
        # Fallback if numpy not available
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))


def _fallback_chunk_by_size(text: str, max_size: int) -> List[str]:
    """Fallback chunking method by character size."""
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > max_size and current:
            chunks.append(" ".join(current))
            current = [word]
            current_size = len(word)
        else:
            current.append(word)
            current_size += word_size
    
    if current:
        chunks.append(" ".join(current))
    
    return chunks


def _merge_small_chunks(chunks: List[str], min_size: int) -> List[str]:
    """Merge chunks that are too small with their neighbors."""
    if not chunks:
        return chunks
    
    merged: List[str] = []
    i = 0
    
    while i < len(chunks):
        current = chunks[i]
        
        # If chunk is too small, try to merge with next chunk
        if len(current) < min_size and i < len(chunks) - 1:
            merged_chunk = current + " " + chunks[i + 1]
            merged.append(merged_chunk)
            i += 2
        else:
            merged.append(current)
            i += 1
    
    return merged

