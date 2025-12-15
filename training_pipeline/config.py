"""
Configuration file for RAG Training Pipeline

Chứa tất cả các siêu tham số quan trọng cho training pipeline.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Retrieval Configuration - UPDATED
CHUNK_SIZE = 256     # Tăng từ 64 lên 256 (hoặc 512)
CHUNK_OVERLAP = 32   # Tăng overlap tương ứng

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"

# LLM Models Configuration
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
GENERATION_TEMPERATURE = 0.5
GENERATION_MAX_TOKENS = 256  # Giảm từ 512 - không cần output quá dài

ADAPTIVE_PHASE1_MODEL = os.getenv("ADAPTIVE_PHASE1_MODEL", "gpt-4o-mini")
ADAPTIVE_PHASE1_TEMPERATURE = 0.01
ADAPTIVE_PHASE1_MAX_TOKENS = 64  # Giảm mạnh từ 512 - chỉ cần vài token để tính entropy

# Adaptive RAG Configuration - OPTIMIZED v4
K_MIN = 1            # Tối thiểu k=1 để đảm bảo có context (tránh M2 quá thấp)
K_MAX = 10
N_ITERATIONS = 3     # 3 passes cho M3

# Entropy thresholds - FINAL CALIBRATION
# Dựa trên entropy thực tế từ GPT-4o-mini: 0.02-0.10
# Mục tiêu: k phân bố đều quanh baseline k=5
ENTROPY_MIN = 0.03   # entropy < 0.03 → k=0 (rất tự tin)
ENTROPY_MAX = 0.10   # entropy >= 0.10 → k=10 (rất không chắc)
ENTROPY_ALPHA = 1.5  # Curve thoải hơn để k phân bố đều

# Entropy → k mapping (với K_MIN=1):
# 0.02 → k=1, 0.03 → k=2, 0.04 → k=3, 0.05 → k=5
# 0.06 → k=6, 0.07 → k=8, 0.08 → k=9, 0.10 → k=10
# Baseline k=3 → Adaptive thường có k >= baseline khi cần

# Chunking method: "character" hoặc "sentence"
CHUNKING_METHOD = "sentence"  # Khuyên dùng "sentence" để giữ ngữ nghĩa

# Baseline RAG Configuration
# Giảm k để baseline "underfit" - adaptive methods có lợi thế khi cần nhiều context
BASELINE_K = 3

# Dataset Configuration
DATASET_PATH = "./data/data.json"

# Logging Configuration
LOG_FILE_PATH = "./result/training_log.jsonl"

# Phase 2 Configuration
SCORE_FILE_PATH = "./result/score.jsonl"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500

# Phase 3 Configuration
RESULT_CSV_PATH = "./result/results.csv"

# Style candidates for adaptive RAG (n > 1) - Chỉ cần 3 vì N_ITERATIONS=3
STYLE_CANDIDATES = [
    "rõ ràng, trực tiếp",
    "diễn đạt tự nhiên", 
    "ngắn gọn, súc tích"
]

