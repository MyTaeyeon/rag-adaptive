# Entropy Statistics - Full Token Analysis (n_iter=5)

## 1. Tổng quan

Thư mục này chứa thí nghiệm thống kê entropy sử dụng tất cả tokens trong response để tính entropy, với n_iterations=5 để cải thiện stability và giảm noise thông qua averaging.

## 2. Bối cảnh

Sau khi phân tích các phương pháp:

- **Full Token (n=1):** Phân tách rõ ràng cho English, nhiễu cho Vietnamese
- **First N Tokens (n=1):** Không cải thiện, thậm chí tệ hơn full_token

Thí nghiệm này kiểm tra giả thuyết: **n_iter càng cao, stability của curve càng ổn định.**

## 3. Cấu trúc Files

- `entropy_stats_en.ipynb`: Notebook thống kê entropy cho queries tiếng Anh
- `entropy_stats_vi.ipynb`: Notebook thống kê entropy cho queries tiếng Việt
- `test_queries_en.json`: 101 test queries tiếng Anh (34 easy, 34 medium, 33 hard)
- `test_queries_vi.json`: 101 test queries tiếng Việt (34 dễ, 34 trung_bình, 33 khó)
- `entropy_analysis_results_en.csv`: Kết quả chi tiết entropy cho từng query (EN)
- `entropy_analysis_results_vi.csv`: Kết quả chi tiết entropy cho từng query (VI)

## 4. Phương pháp

### 4.1 Thu thập dữ liệu

- Sử dụng `AdaptiveController.decide()` với n=5 (5 lần gọi API)
- Mỗi query được gọi API 5 lần với các styles khác nhau
- Temperature = 0.01 (deterministic), max_tokens = 64
- Tính trung bình entropy qua 5 lần gọi

### 4.2 Tính entropy

- Formula: `H = average(-logprob_i / ln(2))` cho TẤT CẢ tokens
- Unit: bits/token
- Tính entropy cho mỗi lần gọi, sau đó average qua 5 calls
- Entropy cao → query khó → cần nhiều documents hơn

### 4.3 Thống kê

- Range: min, max
- Central tendency: mean, median
- Dispersion: std, percentiles (5th, 25th, 50th, 75th, 90th, 95th, 99th)
- Phân tích theo độ khó: easy, medium, hard
- Thêm entropy_std để đo độ biến thiên qua các calls

### 4.4 Visualization

- Scatter plot: Sample ID (x-axis) vs Entropy (y-axis)
- Colors: Red (easy), Blue (medium), Yellow (hard)

## 5. So sánh với Full Token (n=1)

| Aspect | Full Token (n=1) | Full Token (n=5) |
|--------|------------------|------------------|
| API calls per query | 1 | 5 |
| Entropy calculation | Single value | Average over 5 calls |
| Stability | Lower (single sample) | Higher (averaged) |
| Noise | Higher | Lower (reduced by averaging) |
| Cost | Lower | Higher (5x API calls) |
| Mục đích | Baseline, tiết kiệm | Cải thiện stability |

## 6. Kết quả

### 6.1 Giả thuyết được chứng minh

**n_iter càng cao, stability của curve càng ổn định.**

### 6.2 English Data

| Metric | Value |
|--------|-------|
| Entropy Range | [0.1856, 0.3993] |
| Mean | ~0.28 |
| Median | ~0.28 |
| Entropy Std (across 5 iterations) | Rất thấp (gần 0) |

**Nhận xét:**
- Entropy ổn định hơn so với n=1
- Phân tách rõ ràng giữa easy, medium, hard
- Curve mượt mà, ít noise và outliers

### 6.3 Vietnamese Data

| Metric | Value |
|--------|-------|
| Entropy Range | [0.1856, 0.3993] |
| Mean | ~0.28 |
| Median | ~0.28 |
| Entropy Std (across 5 iterations) | Rất thấp (gần 0) |

**Nhận xét:**
- Cải thiện đáng kể so với n=1: Entropy ổn định hơn nhiều
- Phân tách giữa các mức độ khó rõ ràng hơn
- Curve ổn định, giảm noise đáng kể
- Averaging qua 5 calls giúp loại bỏ outliers và biến thiên ngẫu nhiên

## 7. So sánh với các phương pháp khác

### 7.1 Full Token (n=1) vs Full Token (n=5)

| Metric | n=1 | n=5 | Cải thiện |
|--------|-----|-----|-----------|
| Stability | Thấp | Cao | Rõ ràng |
| Noise level | Cao | Thấp | Giảm đáng kể |
| Vietnamese separation | Kém | Tốt hơn | Cải thiện |
| English separation | Tốt | Tốt hơn | Duy trì và cải thiện |
| Entropy std | N/A | Rất thấp | Stability cao |

### 7.2 First N Tokens (n=1) vs Full Token (n=5)

- **First N Tokens:** Không có correlation rõ ràng, phân tách kém
- **Full Token (n=5):** Correlation rõ ràng, phân tách tốt, stability cao

## 8. Kết luận

### 8.1 Phương pháp tối ưu

**Full Token với n_iter=5**

### 8.2 Lý do

1. **Stability cao:**
   - Entropy std rất thấp qua 5 iterations
   - Curve mượt mà, ít biến thiên ngẫu nhiên
   - Kết quả ổn định và đáng tin cậy

2. **Phân tách độ khó tốt:**
   - English: Phân tách rõ ràng giữa easy/medium/hard
   - Vietnamese: Cải thiện đáng kể so với n=1
   - Entropy phản ánh đúng độ khó query

3. **Giảm noise:**
   - Averaging qua 5 calls loại bỏ outliers
   - Giảm ảnh hưởng của biến thiên ngẫu nhiên
   - Kết quả nhất quán hơn

4. **Trade-off hợp lý:**
   - Chi phí API cao hơn 5 lần so với n=1
   - Đổi lại có stability và độ tin cậy cao hơn nhiều
   - Đáng giá cho production system

### 8.3 Quyết định cuối cùng

**Sử dụng Full Token method với n_iter=5 cho hệ thống Adaptive RAG.**

- Phương pháp này đã được chứng minh là tốt nhất qua 3 thí nghiệm
- Stability cao, phân tách độ khó rõ ràng
- Phù hợp cho cả English và Vietnamese queries
- Kết thúc quá trình tuning entropy calculation

## 9. Sử dụng

### 9.1 Chạy notebook

1. Chạy Cells 1-4: Thu thập dữ liệu (101 queries × 5 calls = 505 API calls)
2. Chạy Cells 5-7: Đọc kết quả đã lưu và phân tích

### 9.2 Xem kết quả

- Xem CSV files để có dữ liệu chi tiết
- Xem scatter plot để thấy phân bố entropy trực quan
- So sánh với kết quả trong `entropy_stats_full_token/` (n=1)

## 10. Lưu ý

- Dataset giống hệt full_token để đảm bảo so sánh công bằng
- Sử dụng cùng logic tính entropy như full_token (tất cả tokens)
- Chỉ khác ở số lần gọi API (n=5 thay vì n=1)
- Chi phí API cao hơn 5 lần so với n=1, nhưng đổi lại có stability cao hơn nhiều
- Đây là phương pháp được khuyến nghị cho production
