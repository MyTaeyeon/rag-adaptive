# Mô Tả Training Pipeline - So Sánh 3 Phương Pháp RAG

## MỤC ĐÍCH TRAINING

Pipeline này được thiết kế để so sánh hiệu suất của 3 phương pháp RAG (Retrieval-Augmented Generation) nhằm đánh giá hiệu quả của phương pháp adaptive k so với baseline cố định. Mục tiêu chính là:

- **Đánh giá hiệu quả**: So sánh độ chính xác (accuracy) của 3 phương pháp
- **Đo lường chi phí**: Theo dõi token count và cost cho mỗi phương pháp
- **Đánh giá hiệu năng**: Đo latency (thời gian xử lý) của từng phương pháp

Kết quả sẽ giúp xác định phương pháp nào cho hiệu quả tốt nhất về độ cân bằng giữa accuracy, cost và latency.

---

## 3 PHƯƠNG PHÁP CẦN SO SÁNH

1. **Method 1 - Baseline RAG**: RAG với k cố định k=5
2. **Method 2 - Adaptive RAG Mini**: RAG với k tính từ entropy (n=1, gọi API 1 lần)
3. **Method 3 - Adaptive RAG Full**: RAG với k tính từ entropy trung bình (n=5, gọi API 5 lần và lấy trung bình)

---

## METRICS ĐÁNH GIÁ

Để đánh giá kết quả của 3 phương pháp, ta quan tâm tới:

- **Cost - Token Count**: Tổng số token input và output cho mỗi phương pháp
- **Latency**: Thời gian mỗi phương pháp cần để xử lý 1 query (từ đầu đến cuối)
- **Accuracy**: Độ chính xác đầu ra khi truy vấn k tài liệu và trả về kết quả

**Lưu ý về Stage 2 (Evaluation)**: Sau khi có kết quả từ 3 phương pháp, sẽ sử dụng LLM judge (ví dụ GPT-4) để đánh giá và chấm điểm 3 câu trả lời dựa trên ground truth và context. Stage 2 sẽ được triển khai sau.

---

## QUÁ TRÌNH TRAINING

Training pipeline được chia thành 2 stages:

### Stage 1: Generation & Retrieval
- Chạy 3 phương pháp để lấy k và sinh câu trả lời từ k documents được retrieve
- Track metrics: token count, latency, và output của từng phương pháp

### Stage 2: Evaluation (Triển khai sau)
- Lấy 3 câu trả lời từ 3 phương pháp
- Đưa vào LLM judge cùng với ground truth và context để chấm điểm (0-1)

---

## CHI TIẾT CÁC PHƯƠNG PHÁP

### Method 1: Baseline RAG (k=5)

**Quy trình:**
1. Nhận query
2. Lập tức retrieve 5 documents (k=5)
3. Đưa 5 documents vào context cho LLM
4. Call LLM để lấy câu trả lời

**Metrics tracking:**
- **Latency**: Tổng thời gian từ đầu đến cuối = thời gian retrieval + thời gian call API
- **Token count**: 
  - `total_input_tokens`: Token của prompt (query + context)
  - `total_output_tokens`: Token của câu trả lời

---

### Method 2: Adaptive RAG Mini (n=1)

**Quy trình:**
1. Nhận query
2. **Phase 1**: 
   - Gọi API lần 1 để lấy câu trả lời ban đầu
   - Tính entropy từ logprobs của câu trả lời
   - Tính k dựa trên entropy (dùng hàm `entropy_to_k` trong `training.py`)
3. **Phase 2**:
   - Retrieve k documents dựa trên k đã tính
   - Đưa k documents vào context cho LLM
   - Call API lần 2 để lấy câu trả lời cuối cùng

**Metrics tracking:**
- **Latency**: Tổng thời gian từ đầu đến cuối = call LLM phase 1 + tính entropy + retrieval k + call LLM phase 2
- **Token count**:
  - `total_input_tokens`: Token từ phase 1 + phase 2
  - `total_output_tokens`: Token từ phase 1 + phase 2
- **Additional tracking**:
  - Output từ call LLM phase 1
  - Entropy đã tính được
  - k đã xác định

---

### Method 3: Adaptive RAG Full (n=5)

**Quy trình:**
1. Nhận query
2. **Phase 1**:
   - Gọi API 5 lần với 5 phong cách khác nhau (dùng style_candidates từ `training.py`)
   - Tính entropy cho từng lần gọi
   - Tính entropy trung bình
   - Tính k dựa trên entropy trung bình
3. **Phase 2**:
   - Retrieve k documents dựa trên k đã tính
   - Đưa k documents vào context cho LLM
   - Call API để lấy câu trả lời cuối cùng

**Metrics tracking:**
- **Latency**: Tổng thời gian từ đầu đến cuối = 5 lần call LLM phase 1 + tính entropy + retrieval k + call LLM phase 2
- **Token count**:
  - `total_input_tokens`: Token từ 5 lần call phase 1 + phase 2
  - `total_output_tokens`: Token từ 5 lần call phase 1 + phase 2
- **Additional tracking**:
  - Output từ 5 lần call LLM phase 1
  - Entropy của từng lần
  - Entropy trung bình
  - k đã xác định

---

## RETRIEVAL SYSTEM

**Lưu ý quan trọng**: Retrieval system không thuộc domain chính của bài toán (domain chính là việc chọn k). Do đó, để đảm bảo tính công bằng khi so sánh, ta sẽ sử dụng một retrieval system đơn giản, nhẹ, và dễ triển khai cho cả 3 phương pháp. Điều này đảm bảo:

- Tất cả 3 phương pháp sử dụng cùng một retrieval mechanism
- Tránh bias từ việc sử dụng retrieval system phức tạp
- Tập trung vào việc so sánh hiệu quả của adaptive k vs fixed k

**Chi tiết retrieval:**
- **Context trong data**: Context trong `full.json` chưa được chunk
- **Chunking**: Cần chunk context trước khi retrieve
  - `chunk_size`: Siêu tham số tùy chỉnh trong `config.py`
  - `chunk_overlap`: Siêu tham số tùy chỉnh trong `config.py`
- **Xử lý mỗi sample**:
  - Khi đến một sample, thực hiện chunk context của sample đó
  - Không cần lưu database, chỉ lưu vector embeddings trong RAM
  - Sau khi xử lý xong sample, xóa phần vector đi và tiếp tục sample tiếp theo
- **Retrieval strategy**: Sử dụng similarity search đơn giản (cosine similarity) trên vector embeddings của chunks

---

## CONFIGURATION

### LLM Models Configuration

Tất cả cấu hình về LLM models sẽ được đặt trong file `config.py`, bao gồm:

- Model name cho generation (method 1, 2, 3)
- Model name cho adaptive phase 1 (nếu khác)
- Model parameters (temperature, max_tokens, etc.)
- API keys và settings
---

## TRACKING & LOGGING

### Token Tracking

Tracking chi tiết về token usage:

- **total_input_tokens**: Tổng số token input (prompt tokens) cho toàn bộ quá trình
- **total_output_tokens**: Tổng số token output (completion tokens) cho toàn bộ quá trình

### Latency Tracking

Latency được đo đơn giản:
- Bắt đầu đo thời gian khi bắt đầu chạy một method
- Kết thúc đo khi method hoàn thành và có output
- **latency_ms**: Tổng thời gian (milliseconds) từ đầu đến cuối

Không cần breakdown latency phức tạp, chỉ cần tổng thời gian thực tế.

---

## LOGGING MECHANISM

### Cơ chế Logging cho Training

Hệ thống logging được thiết kế để đảm bảo tính liên tục và khả năng resume:

1. **Logging mỗi sample**: 
   - Mỗi sample sau khi chạy xong (cả 3 methods) sẽ được lưu ngay vào file log
   - File log là JSONL format (mỗi dòng là một JSON object)

2. **Error handling**:
   - Nếu sample i bị lỗi:
     - Ghi lại sample_id (id đã chuẩn hóa) vào cuối file log
     - Dừng lại, không ghi thêm gì nữa
     - File log sẽ có format: các sample thành công (full JSON) + dòng cuối là sample_id bị lỗi

3. **Resume capability**:
   - Hàm `main()` có cơ chế resume từ sample bị lỗi
   - Đọc file log, tìm sample_id cuối cùng (có thể là sample bị lỗi)
   - Tiếp tục training từ sample đó

4. **ID normalization**:
   - ID trong data có thể chưa chuẩn hóa
   - Hàm `main()` sau khi đọc data sẽ chuẩn hóa ID sample từ `0` đến `len(dataset)-1`
   - Sử dụng ID đã chuẩn hóa này trong toàn bộ quá trình training và logging

### Log File Format

**File log**: `training_log.jsonl`

**Format mỗi dòng (sample thành công)**:
```json
{
  "sample_id": 0,
  "original_id": "nq_89919",
  "dataset_source": "naturalquestions",
  "input_data": {
    "query": "...",
    "ground_truth": "...",
    "context": "..."
  },
  "stage_1_results": {
    "method_1_baseline": {...},
    "method_2_adaptive_mini": {...},
    "method_3_adaptive_n5": {...}
  },
  "metadata": {
    "timestamp": "...",
    "latency_ms": 4500
  }
}
```

**Dòng cuối (nếu có lỗi)**:
```json
{"failed_sample_id": 5}
```

---

## TRACKING STRUCTURE CHO MỖI SAMPLE

### Structure tổng quan

```json
{
  "sample_id": 0,
  "original_id": "nq_89919",
  "dataset_source": "naturalquestions",
  "input_data": {
    "query": "when do booth and brennan sleep together for the first time",
    "ground_truth": "...",
    "context": "..."
  },
  "stage_1_results": {
    "method_1_baseline": {...},
    "method_2_adaptive_mini": {...},
    "method_3_adaptive_n5": {...}
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00Z",
    "processing_status": "success"
  }
}
```

### Method 1 - Baseline Structure

```json
{
  "method_1_baseline": {
    "description": "Fixed k=5",
    "config": {
      "k_fixed": 5
    },
    "execution_metrics": {
      "latency_ms": 1250,
      "total_input_tokens": 280,
      "total_output_tokens": 70
    },
    "final_output": "Thủ đô của Pháp là Paris, nổi tiếng với tháp Eiffel và kinh đô thời trang.",
    "status": "success"
  }
}
```

### Method 2 - Adaptive Mini Structure

```json
{
  "method_2_adaptive_mini": {
    "description": "Adaptive k based on single pass entropy",
    "config": {
      "k_min": 0,
      "k_max": 10
    },
    "execution_metrics": {
      "latency_ms": 2100,
      "total_input_tokens": 440,
      "total_output_tokens": 40
    },
    "phase_1_analysis": {
      "llm_output_raw": "Tôi nghĩ là Paris.",
      "calculated_entropy": 0.45,
      "k_determined": 3
    },
    "final_output": "Paris là thủ đô nước Pháp. Nơi đây nổi tiếng với các công trình kiến trúc như tháp Eiffel.",
    "status": "success"
  }
}
```

### Method 3 - Adaptive N5 Structure

```json
{
  "method_3_adaptive_n5": {
    "description": "Adaptive k based on average entropy of 5 passes",
    "config": {
      "n_iterations": 5,
      "k_min": 0,
      "k_max": 10
    },
    "execution_metrics": {
      "latency_ms": 4500,
      "total_input_tokens": 750,
      "total_output_tokens": 100
    },
    "phase_1_analysis": {
      "n_iterations": 5,
      "average_entropy": 0.52,
      "k_determined": 4,
      "iterations_detail": [
        {
          "run": 1,
          "style": "ngắn gọn, súc tích",
          "output": "Paris.",
          "entropy": 0.2
        },
        {
          "run": 2,
          "style": "chi tiết, giải thích từng bước",
          "output": "Là Paris.",
          "entropy": 0.3
        },
        {
          "run": 3,
          "style": "ví dụ minh hoạ thân thiện",
          "output": "Thủ đô Pháp là Paris.",
          "entropy": 0.6
        },
        {
          "run": 4,
          "style": "phân tích phản biện",
          "output": "Paris nhé.",
          "entropy": 0.8
        },
        {
          "run": 5,
          "style": "giải thích cho người mới bắt đầu",
          "output": "Thành phố Paris.",
          "entropy": 0.7
        }
      ]
    },
    "final_output": "Thủ đô của Pháp chính là Paris. Thành phố này nổi tiếng thế giới về nghệ thuật và thời trang.",
    "status": "success"
  }
}
```

---

## WORKFLOW

### Stage 1: Generation & Retrieval

```
1. Load dataset từ full.json
2. Chuẩn hóa ID: 0 -> len(dataset)-1
3. Check log file để xác định điểm resume (nếu có)
4. For each sample (từ điểm resume):
   a. Load sample data (query, ground_truth, context)
   b. Chunk context (dùng chunk_size, chunk_overlap từ config)
   c. Tạo vector embeddings cho chunks (lưu trong RAM)
   
   d. Run Method 1 (Baseline):
      - Retrieve k=5 chunks
      - Generate answer với LLM
      - Track metrics
   
   e. Run Method 2 (Adaptive Mini):
      - Phase 1: Generate + calculate entropy
      - Determine k from entropy
      - Phase 2: Retrieve k chunks + generate
      - Track metrics
   
   f. Run Method 3 (Adaptive N5):
      - Phase 1: Generate 5 times + avg entropy
      - Determine k from avg entropy
      - Phase 2: Retrieve k chunks + generate
      - Track metrics
   
   g. Lưu kết quả vào log file (JSONL format)
   h. Xóa vector embeddings từ RAM
   
5. Nếu có lỗi ở sample i:
   - Ghi failed_sample_id vào cuối log file
   - Dừng lại
```

### Stage 2: Evaluation (Triển khai sau)

```
1. Load tất cả stage_1_results từ log file
2. For each sample:
   a. Chuẩn bị input cho LLM judge:
      - query
      - ground_truth
      - context (retrieved chunks)
      - 3 answers từ 3 methods
   b. Call LLM judge để đánh giá
   c. Lưu kết quả evaluation
3. Tính toán aggregate statistics
```

---

## CẤU TRÚC FILE CONFIG

File `config.py` sẽ chứa các cấu hình sau:

```python
# Retrieval Config
CHUNK_SIZE = 512  # Siêu tham số tùy chỉnh
CHUNK_OVERLAP = 50  # Siêu tham số tùy chỉnh

# LLM Models Config
GENERATION_MODEL = "gpt-4o-mini"
GENERATION_TEMPERATURE = 0.7
GENERATION_MAX_TOKENS = 512

ADAPTIVE_PHASE1_MODEL = "gpt-4o-mini"
ADAPTIVE_PHASE1_TEMPERATURE = 0.01
ADAPTIVE_PHASE1_MAX_TOKENS = 512

# Adaptive RAG Config
K_MIN = 0
K_MAX = 10
N_ITERATIONS = 5  # Cho method 3

# Baseline RAG Config
BASELINE_K = 5

# Logging Config
LOG_FILE_PATH = "training_log.jsonl"
```

---

## IMPLEMENTATION CHECKLIST

### Phase 1: Setup & Configuration
- [ ] Tạo file `config.py` với tất cả configs
- [ ] Setup retrieval system đơn giản (chunking + embedding + similarity search)
- [ ] Setup LLM clients
- [ ] Implement error handling

### Phase 2: Stage 1 Implementation
- [ ] Implement chunking function
- [ ] Implement retrieval system (in-memory)
- [ ] Implement Method 1 (Baseline)
- [ ] Implement Method 2 (Adaptive Mini)
- [ ] Implement Method 3 (Adaptive N5)
- [ ] Implement logging mechanism (JSONL format)
- [ ] Implement ID normalization
- [ ] Implement resume capability

### Phase 3: Testing & Validation
- [ ] Test với một vài samples nhỏ
- [ ] Validate log file format
- [ ] Test resume mechanism
- [ ] Validate metrics tracking

### Phase 4: Stage 2 Implementation (Sau)
- [ ] Design LLM judge prompt
- [ ] Implement judge call
- [ ] Implement evaluation aggregation

---

## NOTES

- **Focus**: Tập trung vào so sánh hiệu quả của adaptive k vs fixed k
- **Simplicity**: Sử dụng retrieval system đơn giản để tránh bias
- **Resumability**: Đảm bảo có thể resume từ điểm lỗi
- **Logging**: Log ngay sau mỗi sample để tránh mất dữ liệu

