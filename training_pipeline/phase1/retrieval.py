"""
Retrieval System Module

Retrieval system đơn giản, in-memory với embedding và similarity search.
Chỉ lưu trong RAM, xóa sau mỗi sample.
"""

import sys
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from openai import OpenAI

sys.path.append(str(Path(__file__).parent.parent))
import config
from chunking import chunk_text


class InMemoryRetrieval:
    """
    Retrieval system đơn giản, in-memory.
    Lưu embeddings trong RAM, tự động xóa sau khi sử dụng.
    """
    
    def __init__(self):
        self.embedding_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.embedding_model = config.EMBEDDING_MODEL
        self.chunks = []
        self.embeddings = None
    
    def index_context(self, context: str) -> None:
        """
        Chunk context và tạo embeddings, lưu trong RAM.
        
        Args:
            context: Context text cần index
        """
        self.chunks = chunk_text(context)
        
        if not self.chunks:
            self.embeddings = np.array([])
            return
        
        embeddings_list = []
        for chunk in self.chunks:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=chunk
            )
            embeddings_list.append(response.data[0].embedding)
        
        self.embeddings = np.array(embeddings_list)
    
    def retrieve(self, query: str, k: int) -> List[Dict[str, any]]:
        """
        Retrieve k chunks gần nhất với query.
        
        Args:
            query: Query text
            k: Số lượng chunks cần retrieve
        
        Returns:
            List các chunks với metadata:
            [
                {
                    "text": "...",
                    "similarity_score": 0.95,
                    "chunk_index": 0
                },
                ...
            ]
        """
        if not self.chunks or len(self.chunks) == 0:
            return []
        
        if self.embeddings.size == 0:
            return []
        
        query_response = self.embedding_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        query_embedding = np.array(query_response.data[0].embedding)
        
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "text": self.chunks[idx],
                "similarity_score": float(similarities[idx]),
                "chunk_index": int(idx)
            })
        
        return results
    
    def clear(self) -> None:
        """
        Xóa tất cả chunks và embeddings khỏi RAM.
        """
        self.chunks = []
        self.embeddings = None

