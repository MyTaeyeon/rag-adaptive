"""
Adaptive RAG Module - Training Logic

Module này chứa logic chính để tính toán entropy từ token logprobs của OpenAI model
và chuyển đổi entropy thành giá trị k (số lượng chunk/document cần retrieve trong RAG).
"""

from dotenv import load_dotenv
import os
import math
from typing import List
from collections import Counter

from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
openai_client: OpenAI = OpenAI(api_key=OPENAI_API_KEY)
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")



def text_to_token_distribution(text: str, normalize: bool = True) -> List[float]:
    """Chuyển text thành phân bố token (theo tần suất từ) dùng để xấp xỉ entropy.
        Dùng khi api không trả về  log probs, tránh vỡ pipeline
    Args:
        text: Văn bản cần phân tích
        normalize: Nếu True, chuẩn hóa tần suất thành xác suất (tổng = 1)
    
    Returns:
        Danh sách tần suất hoặc xác suất của các từ
    """
    tokens = [t for t in text.strip().split() if t]
    if not tokens:
        return []
    total = len(tokens)
    # Đếm tần suất
    counts = Counter(tokens)
    freqs = [c for c in counts.values()]
    if normalize:
        freqs = [c / total for c in freqs]
    return freqs


def shannon_entropy(probabilities: List[float]) -> float:
    """Tính Shannon entropy (log base 2) từ một danh sách xác suất.
    
    Args:
        probabilities: Danh sách các xác suất
    
    Returns:
        Shannon entropy (bits)
    """
    if not probabilities:
        return 0.0
    # Lọc các p=0 tránh log(0)
    return -sum(p * math.log2(p) for p in probabilities if p > 0)


def calculate_entropy(answer: str) -> float:
    """Tính entropy xấp xỉ từ câu trả lời của model.

    Bước:
    - Chuyển text thành phân bố tần suất từ.
    - Tính Shannon entropy trên phân bố đó.
    
    Args:
        answer: Câu trả lời từ model
    
    Returns:
        Entropy xấp xỉ (bits)
    """
    probs = text_to_token_distribution(answer)
    return shannon_entropy(probs)


# ==== Helper Functions ====

def _build_style_prompt(query: str, style: str) -> str:
    """Tạo prompt yêu cầu model trả lời cùng nội dung nhưng phong cách khác nhau.
    
    Args:
        query: Câu hỏi của user
        style: Phong cách trả lời (ví dụ: "ngắn gọn, súc tích")
    
    Returns:
        Prompt đã được format
    """
    return (
        f"Trả lời câu hỏi sau với phong cách {style}. "
        f"Giữ nội dung chính xác ngắn gọn.\n\n"
        f"Câu hỏi: {query}"
    )


def _build_default_prompt(query: str) -> str:
    """Tạo prompt mặc định cho câu hỏi.
    
    Args:
        query: Câu hỏi của user
    
    Returns:
        Prompt mặc định
    """
    return f"Trả lời ngắn gọn, trực tiếp:\n\nCâu hỏi: {query}"


def sequence_entropy_from_token_logprobs(token_logprobs: List[float]) -> float:
    """Tính entropy trung bình (bits/token) từ logprobs của từng token.

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


# ==== Core Functions ====

def _generate_answer(
    query: str,
    style_hint: str | None = None,
    backend: str = "openai",
    model_name: str | None = None,
) -> tuple[str, float]:
    """Gọi OpenAI model để sinh 1 câu trả lời + entropy từ logprobs.

    - Chỉ hỗ trợ backend="openai" để đảm bảo entropy dựa trên token logprobs.
    - model_name: tên model OpenAI hỗ trợ logprobs (mặc định dùng DEFAULT_OPENAI_MODEL).
    
    Args:
        query: Câu hỏi của user
        style_hint: Gợi ý phong cách trả lời (None = mặc định)
        backend: Backend sử dụng (hiện chỉ hỗ trợ "openai")
        model_name: Tên model OpenAI (None = dùng DEFAULT_OPENAI_MODEL)
    
    Returns:
        Tuple (answer_text, entropy)
    
    Raises:
        ValueError: Nếu backend không phải "openai"
    """
    if backend.lower() != "openai":
        raise ValueError("Only OpenAI backend with logprobs is supported in this pipeline")

    if style_hint:
        prompt = _build_style_prompt(query, style_hint)
    else:
        prompt = _build_default_prompt(query)

    if model_name is None:
        model_name = DEFAULT_OPENAI_MODEL

    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.01,
        logprobs=True,
        top_logprobs=5,
    )
    choice = resp.choices[0]
    text = choice.message.content or ""

    token_logprobs: List[float] = []
    if choice.logprobs and choice.logprobs.content:
        for t in choice.logprobs.content:
            if t.logprob is not None:
                token_logprobs.append(t.logprob)

    if token_logprobs:
        entropy = sequence_entropy_from_token_logprobs(token_logprobs)
    else:
        # Fallback rất hiếm khi cần, nhưng vẫn giữ để pipeline không vỡ
        entropy = calculate_entropy(text)

    return text, entropy


def entropy_to_k(entropy: float, k_min: int = 0, k_max: int = 10) -> int:
    """Chuyển entropy (bits/token) → k với log-scale để phân biệt tốt hơn.

    Yêu cầu thực tế:
    - Câu rất chắc chắn (entropy ~ 0.01–0.05) → k nên ~ 0.
    - Câu khó hơn (entropy ~ 0.2–0.5) → k nên tăng rõ ràng (2–4).
    - Câu rất mơ hồ / khó (entropy > 1–2) → k gần k_max.

    Ta dùng log-scale: e' = log(1 + entropy), rồi chuẩn hoá e' vào [0, 1].
    Điều này làm khác biệt nhỏ ở vùng entropy thấp vẫn nhìn thấy rõ hơn.
    
    Args:
        entropy: Entropy (bits/token)
        k_min: Giá trị k tối thiểu
        k_max: Giá trị k tối đa
    
    Returns:
        Giá trị k (số lượng chunk/document cần retrieve)
    """
    if entropy <= 0:
        return k_min

    max_entropy_base = 0.3
    e_norm = min(entropy / max_entropy_base, 1.0)

    k_float = k_min + e_norm * (k_max - k_min)
    return int(round(k_float))


def adaptive_k(
    query: str,
    n: int = 1,
    k_min: int = 0,
    k_max: int = 10,
    backend: str = "openai",
    model_name: str | None = None,
) -> int:
    """Hàm chính: chọn k dựa trên entropy từ câu trả lời của model OpenAI.

    - query: câu hỏi của user.
    - n: số lần gọi model để ước lượng entropy.
      + Nếu n = 1: gọi 1 lần, lấy entropy của 1 câu trả lời.
      + Nếu n > 1 (vd n = 3): gọi n lần với các phong cách khác nhau,
        lấy entropy trung bình của n câu trả lời.
    - k_min, k_max: giới hạn dưới/trên cho k.
    - backend: hiện chỉ hỗ trợ "openai" để đảm bảo dùng logprobs.
    - model_name: tên model OpenAI (mặc định dùng DEFAULT_OPENAI_MODEL).

    Trả về: k (số lượng chunk/document nên retrieve trong RAG).
    
    Args:
        query: Câu hỏi của user
        n: Số lần gọi model để ước lượng entropy (>= 1)
        k_min: Giá trị k tối thiểu
        k_max: Giá trị k tối đa
        backend: Backend sử dụng (hiện chỉ hỗ trợ "openai")
        model_name: Tên model OpenAI (None = dùng DEFAULT_OPENAI_MODEL)
    
    Returns:
        Giá trị k (số lượng chunk/document cần retrieve trong RAG)
    
    Raises:
        ValueError: Nếu backend không phải "openai" hoặc n <= 0
    """
    if backend.lower() != "openai":
        raise ValueError("adaptive_k currently only supports backend='openai'")

    if n <= 0:
        raise ValueError("n must be >= 1")

    style_candidates = [
        "ngắn gọn, súc tích",
        "chi tiết, giải thích từng bước",
        "ví dụ minh hoạ thân thiện",
        "phân tích phản biện",
        "giải thích cho người mới bắt đầu",
    ]

    entropies: List[float] = []

    for i in range(n):
        style = None
        if n > 1:
            # Chọn style khác nhau cho mỗi lần gọi
            style = style_candidates[i % len(style_candidates)]

        _, e = _generate_answer(
            query=query,
            style_hint=style,
            backend=backend,
            model_name=model_name,
        )
        entropies.append(e)

    avg_entropy = sum(entropies) / len(entropies)
    k = entropy_to_k(avg_entropy, k_min=k_min, k_max=k_max)

    return k

