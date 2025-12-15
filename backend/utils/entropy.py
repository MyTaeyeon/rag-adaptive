"""Entropy calculation utilities for adaptive RAG."""

import math
from typing import List


def sequence_entropy_from_token_logprobs(token_logprobs: List[float]) -> float:
    """
    Tính entropy trung bình (bits/token) từ logprobs của từng token.
    
    - OpenAI trả về logprobs theo cơ số e (natural log).
    - Với mỗi token, xác suất p_i = exp(logprob_i).
    - Surprisal (độ bất ngờ) theo base-2: s_i = -log2(p_i) = -logprob_i / ln(2).
    - Entropy xấp xỉ: H ≈ (1/T) * Σ s_i.
    
    Kết quả: H ~ 0 với câu rất chắc chắn, tăng dần khi model kém chắc chắn.
    
    Args:
        token_logprobs: Danh sách logprobs của các token (natural log)
    
    Returns:
        Entropy trung bình (bits/token)
    """
    if not token_logprobs:
        return 0.0
    
    ln2 = math.log(2.0)
    surprisals = [(-lp) / ln2 for lp in token_logprobs]
    return sum(surprisals) / len(surprisals)


def entropy_to_k(average_entropy: float, k_min: int, k_max: int, entropy_max: float = 0.3) -> int:
    """
    Chuyển đổi average entropy thành số documents cần retrieve (k).
    
    Entropy cao → model không chắc chắn → cần nhiều documents hơn.
    Entropy thấp → model chắc chắn → cần ít documents hơn.
    
    Args:
        average_entropy: Entropy trung bình từ n iterations
        k_min: Số documents tối thiểu
        k_max: Số documents tối đa
        entropy_max: Giá trị entropy tối đa để normalize (default: 0.3)
    
    Returns:
        Số documents cần retrieve (k)
    """
    # Quy tắc ngưỡng: rất chắc chắn → k_min; rất không chắc → k_max
    if average_entropy <= 0.25:
        return k_min
    if average_entropy >= 0.45:
        return k_max

    # Linear mapping trong vùng chuyển tiếp [0.2, 0.4]
    normalized_entropy = (average_entropy - 0.25) / 0.2  # map 0.2->0, 0.4->1
    k = int(k_min + normalized_entropy * (k_max - k_min)) + 1
    return max(k_min, min(k_max, k))

