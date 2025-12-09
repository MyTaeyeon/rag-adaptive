# Entropy Statistics - Full Token Analysis

Thư mục này chứa toàn bộ thí nghiệm thống kê entropy sử dụng tất cả tokens trong response để tính entropy.

## Bối cảnh

Trong hệ thống Adaptive RAG, chúng ta sử dụng entropy để đánh giá độ không chắc chắn của model khi trả lời câu hỏi, từ đó xác định số lượng documents cần retrieve (k).

**Phương pháp hiện tại:** Tính entropy trung bình từ TẤT CẢ tokens trong response của LLM (non-hop call).

**Mục tiêu:** Xác định range entropy thực tế từ dữ liệu để thiết kế công thức mapping entropy → k phù hợp.

## Cấu trúc Files

- `entropy_stats_en.ipynb`: Notebook thống kê entropy cho queries tiếng Anh
- `entropy_stats_vi.ipynb`: Notebook thống kê entropy cho queries tiếng Việt
- `test_queries_en.json`: 101 test queries tiếng Anh (34 easy, 34 medium, 33 hard)
- `test_queries_vi.json`: 101 test queries tiếng Việt (34 dễ, 34 trung_bình, 33 khó)
- `entropy_analysis_results_en.csv`: Kết quả chi tiết entropy cho từng query (EN)
- `entropy_analysis_results_vi.csv`: Kết quả chi tiết entropy cho từng query (VI)
- `entropy_summary_en.json`: Thống kê tổng hợp (EN)

## Phương pháp

1. **Thu thập dữ liệu:**
   - Sử dụng `AdaptiveController.decide()` với `n=1` (tiết kiệm API calls)
   - Mỗi query được gọi API 1 lần để lấy entropy
   - Temperature = 0.01 (deterministic), max_tokens = 64

2. **Tính entropy:**
   - Formula: `H = average(-logprob_i / ln(2))` cho TẤT CẢ tokens
   - Unit: bits/token
   - Entropy cao → query khó → cần nhiều documents hơn

3. **Thống kê:**
   - Range: min, max
   - Central tendency: mean, median
   - Dispersion: std, percentiles (5th, 25th, 50th, 75th, 90th, 95th, 99th)
   - Phân tích theo độ khó: easy, medium, hard

4. **Visualization:**
   - Scatter plot: Sample ID (x-axis) vs Entropy (y-axis)
   - Colors: Red (easy), Blue (medium), Yellow (hard)

## Kết quả

### English Data:
- **Entropy Range:** [0.0944, 0.5355]
- **Mean:** ~0.28
- **Median:** ~0.25
- **Observation:** Entropy tăng tuyến tính với độ khó query (phân tách rõ ràng giữa easy, medium, hard)

### Vietnamese Data:
- **Entropy Range:** [0.1584, 0.4207]
- **Mean:** ~0.27
- **Median:** ~0.27
- **Observation:** Kết quả nhiễu, phân tách giữa các mức độ khó không rõ ràng như EN data

## Nhận xét

### Vấn đề phát hiện:

1. **English queries:** Phương pháp hiện tại hoạt động tốt, entropy phản ánh đúng độ khó

2. **Vietnamese queries:** 
   - Entropy phân bố không rõ ràng giữa các mức độ khó
   - Nguyên nhân có thể do:
     - Token đầu tiên có entropy cao (phản ánh độ khó ban đầu)
     - Token giữa/cuối có entropy thấp (khi đã vào flow trả lời, model tự tin hơn)
     - Trung bình entropy bị kéo xuống, không phản ánh đúng độ khó thực sự

3. **Giả thuyết:**
   - Với query khó, model không chắc chắn ở token đầu tiên → entropy cao
   - Nhưng khi trả lời dài, các token sau có entropy thấp (câu trả lời nghe có vẻ đúng hơn)
   - Temperature = 0.01 không ngăn được hiện tượng này
   - Output dài làm trung bình entropy thấp cho mọi query

## Hướng đi

Để giải quyết vấn đề này, chúng ta đề xuất thử nghiệm phương pháp mới:

### Phương pháp mới: First N Tokens
- **Ý tưởng:** Chỉ tính entropy từ N token đầu tiên (N=5 mặc định)
- **Lý do:**
  - Token đầu phản ánh tốt hơn độ không chắc chắn ban đầu của model
  - Token sau bị ảnh hưởng bởi phần đã generate, không phản ánh độ khó query
  - Nhiều nghiên cứu về uncertainty estimation cũng chỉ dùng token đầu

### Thử nghiệm tiếp theo:
Thư mục `../entropy_stats_first_n_token/` sẽ chứa:
- Biến thể tính entropy từ N token đầu tiên (N=5)
- Sử dụng cùng dataset để đảm bảo nhất quán
- So sánh kết quả với phương pháp hiện tại

## Sử dụng

### Chạy notebook:
1. Chạy Cells 1-4: Thu thập dữ liệu (chỉ cần chạy 1 lần)
2. Chạy Cells 5-7: Đọc kết quả đã lưu và phân tích

### Xem kết quả:
- Xem CSV files để có dữ liệu chi tiết
- Xem scatter plot để thấy phân bố entropy trực quan
- So sánh với kết quả trong `entropy_stats_first_n_token/`
