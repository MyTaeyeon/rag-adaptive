"""
Text Chunking Module

Chia context thành các chunks với overlap để chuẩn bị cho retrieval.
Hỗ trợ cả character-based và sentence-based chunking.
"""

import sys
import re
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Chia text thành các chunks với overlap (character-based).
    
    Args:
        text: Text cần chia
        chunk_size: Kích thước mỗi chunk (mặc định từ config)
        chunk_overlap: Số ký tự overlap giữa các chunks (mặc định từ config)
    
    Returns:
        List các chunks
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP
    
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
            
        start = end - chunk_overlap
    
    return chunks


def chunk_text_by_sentences(
    text: str, 
    max_chunk_size: int = None, 
    overlap_sentences: int = 1
) -> List[str]:
    """
    Chunk text theo câu, giữ ngữ nghĩa nguyên vẹn.
    Tốt hơn character-based vì không cắt giữa câu.
    
    Args:
        text: Text cần chia
        max_chunk_size: Kích thước tối đa mỗi chunk (mặc định từ config.CHUNK_SIZE)
        overlap_sentences: Số câu overlap giữa các chunks
    
    Returns:
        List các chunks (mỗi chunk là 1 hoặc nhiều câu hoàn chỉnh)
    """
    if max_chunk_size is None:
        max_chunk_size = config.CHUNK_SIZE
    
    if not text:
        return []
    
    # Split by sentence endings (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text else []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # Nếu thêm câu này vượt quá max_chunk_size và đã có content
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Lưu chunk hiện tại
            chunks.append(' '.join(current_chunk))
            
            # Keep last N sentences for overlap
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) + 1 for s in current_chunk)  # +1 for space
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space
    
    # Thêm chunk cuối cùng
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

