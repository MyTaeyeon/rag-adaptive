"""
Text Chunking Module

Chia context thành các chunks với overlap để chuẩn bị cho retrieval.
"""

import sys
from typing import List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Chia text thành các chunks với overlap.
    
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

