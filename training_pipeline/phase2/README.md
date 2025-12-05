# Phase 2: Evaluation Pipeline

## Mục Đích

Phase 2 đọc kết quả từ `training_log.jsonl` (đã được tạo từ Phase 1), gọi LLM judge để đánh giá 3 câu trả lời từ 3 methods và lưu scores vào `score.jsonl`.

## Cách Chạy

```bash
cd phase2
python main_phase2.py
```

Hoặc từ thư mục gốc:

```bash
python phase2/main_phase2.py
```

## Output

Script sẽ tạo file `score.jsonl` trong thư mục gốc với format:

```json
{
  "sample_id": 0,
  "original_id": "nq_89919",
  "dataset_source": "naturalquestions",
  "evaluation": {
    "scores": {
      "method_1": {
        "factual_accuracy": 0.85,
        "completeness": 0.80,
        "relevance": 0.90,
        "overall_score": 0.85
      },
      "method_2": {...},
      "method_3": {...}
    },
    "judge_metadata": {
      "judge_model": "gpt-4o",
      "judge_tokens": 500,
      "judge_latency_ms": 800
    }
  },
  "timestamp": "..."
}
```

## Cấu Hình

Các config cho phase2 được đặt trong `config.py` ở thư mục gốc:
- `JUDGE_MODEL`: Model dùng để judge (mặc định: "gpt-4o")
- `JUDGE_TEMPERATURE`: Temperature cho judge (mặc định: 0.0)
- `SCORE_FILE_PATH`: Đường dẫn file output (mặc định: "score.jsonl")

