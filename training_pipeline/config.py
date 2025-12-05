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

# Retrieval Configuration
CHUNK_SIZE = 64
CHUNK_OVERLAP = 0

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"

# LLM Models Configuration
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 512

ADAPTIVE_PHASE1_MODEL = os.getenv("ADAPTIVE_PHASE1_MODEL", "gpt-4o-mini")
ADAPTIVE_PHASE1_TEMPERATURE = 0.01
ADAPTIVE_PHASE1_MAX_TOKENS = 512

# Adaptive RAG Configuration
K_MIN = 0
K_MAX = 10
N_ITERATIONS = 5
MAX_ENTROPY_BASE = 0.3

# Baseline RAG Configuration
BASELINE_K = 5

# Dataset Configuration
DATASET_PATH = "full.json"

# Logging Configuration
LOG_FILE_PATH = "training_log.jsonl"

# Phase 2 Configuration
SCORE_FILE_PATH = "score.jsonl"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500

# Phase 3 Configuration
RESULT_CSV_PATH = "result.csv"

# Style candidates for adaptive RAG (n > 1)
STYLE_CANDIDATES = [
    "ngắn gọn, súc tích",
    "chi tiết, giải thích từng bước",
    "ví dụ minh hoạ thân thiện",
    "phân tích phản biện",
    "giải thích cho người mới bắt đầu",
]

