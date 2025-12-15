# Hướng Dẫn Sử Dụng Training Pipeline

## Chạy Pipeline

Pipeline gồm 3 phase: Generation (Phase 1), Evaluation (Phase 2), và Statistics (Phase 3).

### Phase 1: Generation & Retrieval

Chạy 3 phương pháp RAG và lưu kết quả vào `training_log.jsonl`:

```bash
python phase1/main_phase1.py
```

Output: File `training_log.jsonl` chứa kết quả của 3 methods cho mỗi sample.

### Phase 2: Evaluation

Đọc kết quả từ Phase 1, gọi LLM judge để đánh giá và lưu scores vào `score.jsonl`:

```bash
python phase2/main_phase2.py
```
Input: File `training_log.jsonl` từ Phase 1
Output: File `score.jsonl` chứa scores cho 3 methods

### Phase 3: Statistics & CSV Export

Đọc kết quả từ Phase 1 và Phase 2, chuyển đổi thành CSV và tạo thống kê:

```bash
python phase3/convert_to_csv.py
```

Input: File `training_log.jsonl` và `score.jsonl`
Output: File `result.csv` chứa tất cả metrics

Để xem thống kê và visualization:

```bash
python phase3/statistics.py
```

Output: Thống kê mô tả và các biểu đồ so sánh

## Mục Đích

So sánh 3 phương pháp RAG:
1. Baseline RAG (k=3 cố định)
2. Adaptive RAG Mini (n=1, k thích ứng)
3. Adaptive RAG Full (n=5, k thích ứng)

## Yêu Cầu

- Python 3.8+
- OpenAI API key (set trong environment variable hoặc .env file)
- Packages: openai, numpy, python-dotenv, pandas (cho Phase 3)

Cài đặt:

```bash
pip install openai numpy python-dotenv pandas matplotlib
```

## Cấu Hình

Tất cả siêu tham số trong file `config.py`:
- CHUNK_SIZE, CHUNK_OVERLAP: Chunking context
- K_MIN, K_MAX: Giới hạn k cho adaptive methods
- BASELINE_K: k cố định cho baseline
- Model configurations (generation, judge)
- JUDGE_MODEL: Model dùng để judge (mặc định: gpt-4o)

## File Formats

### training_log.jsonl (Phase 1 output)

Mỗi dòng là một JSON object:

```json
{
  "sample_id": 0,
  "original_id": "nq_89919",
  "input_data": {"query": "...", "ground_truth": "...", "context": "..."},
  "stage_1_results": {
    "method_1_baseline": {...},
    "method_2_adaptive_mini": {...},
    "method_3_adaptive_n5": {...}
  }
}
```

### score.jsonl (Phase 2 output)

Mỗi dòng là một JSON object:

```json
{
  "sample_id": 0,
  "evaluation": {
    "scores": {
      "method_1": {"factual_accuracy": 0.85, "completeness": 0.80, "relevance": 0.90, "overall_score": 0.85},
      "method_2": {...},
      "method_3": {...}
    }
  }
}
```

### result.csv (Phase 3 output)

File CSV với các cột:
- sample_id
- m1_latency, m2_latency, m3_latency
- m1_total_input_token, m2_total_input_token, m3_total_input_token
- m1_total_output_token, m2_total_output_token, m3_total_output_token
- m1_k, m2_k, m3_k
- m1_judge_score, m2_judge_score, m3_judge_score

## Resume Capability

Phase 1 tự động resume từ sample bị lỗi:

- Nếu bị lỗi ở sample X, chạy lại script sẽ tự động tiếp tục từ sample X
- File log có format: các sample thành công + dòng cuối `{"failed_sample_id": X}` nếu có lỗi
- Chỉ cần chạy lại `python phase1/main_phase1.py` để resume

## Kiểm Tra Kết Quả

```bash
# Đếm số samples đã xử lý
wc -l training_log.jsonl

# Xem sample đầu tiên
head -n 1 training_log.jsonl | python -m json.tool

# Xem scores
head -n 1 score.jsonl | python -m json.tool
```

## Troubleshooting

### Lỗi API Key

Set API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Hoặc tạo file `.env`:

```
OPENAI_API_KEY=your-api-key
```

### Lỗi Module

Cài đặt dependencies:

```bash
pip install openai numpy python-dotenv
```

### Lỗi Import

Đảm bảo chạy script từ đúng thư mục hoặc sử dụng đường dẫn đầy đủ.

## Tổng Hợp Kết Quả Thực Nghiệm

Kết quả được thu thập từ 100 samples trên dataset European Union Law.

### Tổng Quan

- **Số lượng samples**: 100
- **Dataset**: European Union Law
- **3 phương pháp so sánh**:
  - **M1 (Baseline RAG)**: k=3 cố định
  - **M2 (Adaptive RAG Mini)**: n=1, k thích ứng
  - **M3 (Adaptive RAG Full)**: n=5, k thích ứng

### Kết Quả Chi Tiết

#### 1. Latency (Thời gian xử lý)

| Method | Mean Latency (ms) |
|--------|-------------------|
| M1 (Baseline) | 1,727.7 |
| M2 (Adaptive Mini) | 2,774.1 |
| M3 (Adaptive Full) | 5,027.7 |

**Nhận xét**: Baseline có latency thấp nhất do không có bước phân tích entropy. Adaptive Full có latency cao nhất do phải thực hiện 5 lần phân tích entropy.

#### 2. Token Usage (Sử dụng token)

**Input Tokens:**
| Method | Mean | Total |
|--------|------|-------|
| M1 (Baseline) | 248 | 24,826 |
| M2 (Adaptive Mini) | 567 | 56,730 |
| M3 (Adaptive Full) | 738 | 73,810 |

**Output Tokens:**
| Method | Mean | Total |
|--------|------|-------|
| M1 (Baseline) | 38 | 3,782 |
| M2 (Adaptive Mini) | 56 | 5,554 |
| M3 (Adaptive Full) | 97 | 9,690 |

**Nhận xét**: Adaptive methods sử dụng nhiều token hơn do cần thêm context cho việc phân tích entropy và retrieve nhiều documents hơn khi cần thiết.

#### 3. Judge Score (Điểm đánh giá chất lượng)

| Method | Average Score | Accuracy (score >= 0.5) |
|--------|---------------|-------------------------|
| M1 (Baseline) | 0.828 | 82.0% |
| M2 (Adaptive Mini) | 0.888 | 89.0% |
| M3 (Adaptive Full) | 0.908 | 91.0% |

**Nhận xét**: 
- Adaptive methods cho kết quả tốt hơn về chất lượng, với M3 (Adaptive Full) đạt điểm cao nhất (0.908)
- Accuracy của M3 cao hơn M1 khoảng 9%, cho thấy adaptive retrieval giúp cải thiện đáng kể chất lượng câu trả lời

### Kết Luận

1. **Về chất lượng**: Adaptive RAG (đặc biệt là Adaptive Full) cho kết quả tốt hơn Baseline, với điểm judge score cao hơn và accuracy tốt hơn.

2. **Về hiệu suất**: Baseline có latency và token usage thấp nhất, phù hợp cho các ứng dụng cần tốc độ và chi phí thấp.

3. **Trade-off**: Adaptive methods đánh đổi latency và token usage để đạt được chất lượng tốt hơn. Adaptive Mini (M2) là sự cân bằng tốt giữa chất lượng và hiệu suất.

### Files Kết Quả

- **`result/results.csv`**: File CSV chứa tất cả metrics chi tiết cho 100 samples
- **`result/training_log.jsonl`**: Log chi tiết từ Phase 1
- **`result/score.jsonl`**: Scores từ Phase 2
- **`phase3/plots/`**: Các biểu đồ visualization so sánh metrics

Xem chi tiết thống kê và visualization trong file `test.ipynb`.