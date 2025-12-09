"""
Biến thể tính entropy từ N token đầu tiên (không sửa logic adaptive controller).
Module này được sử dụng riêng cho thí nghiệm entropy statistics.
"""

import math
from typing import List, Optional


def sequence_entropy_from_first_n_tokens(
    token_logprobs: List[float], 
    num_tokens: int = 5
) -> float:
    """
    Tính entropy từ N token đầu tiên (thay vì tất cả tokens).
    
    Lý do: Token đầu tiên phản ánh tốt hơn độ không chắc chắn ban đầu của model
    khi trả lời query. Token sau bị ảnh hưởng bởi phần đã generate.
    
    Args:
        token_logprobs: Danh sách logprobs của các token (natural log)
        num_tokens: Số token đầu tiên để tính entropy (mặc định: 5)
    
    Returns:
        Entropy trung bình (bits/token) từ N token đầu
    """
    if not token_logprobs:
        return 0.0
    
    # Chỉ lấy num_tokens đầu tiên
    # Nếu không đủ tokens, lấy tất cả
    tokens_to_use = token_logprobs[:num_tokens]
    
    if not tokens_to_use:
        return 0.0
    
    # Tính entropy từ N token đầu tiên
    ln2 = math.log(2.0)
    surprisals = [(-lp) / ln2 for lp in tokens_to_use]
    return sum(surprisals) / len(surprisals)


def calculate_entropy_from_response(
    logprobs: List[float],
    num_tokens: int = 5
) -> float:
    """
    Wrapper function để tính entropy từ response logprobs.
    Tương tự như sequence_entropy_from_token_logprobs nhưng chỉ dùng N token đầu.
    
    Args:
        logprobs: Danh sách logprobs từ API response
        num_tokens: Số token đầu tiên để tính (mặc định: 5)
    
    Returns:
        Entropy trung bình (bits/token)
    """
    return sequence_entropy_from_first_n_tokens(logprobs, num_tokens)

