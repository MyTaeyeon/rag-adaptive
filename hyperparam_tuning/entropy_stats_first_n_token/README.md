# Entropy Statistics - First N Tokens Analysis

Thư mục này chứa thí nghiệm thống kê entropy sử dụng chỉ N token đầu tiên (N=5) từ response để tính entropy.

## Bối cảnh

Sau khi phân tích entropy với phương pháp **Full Token** (xem `../entropy_stats_full_token/`), chúng ta phát hiện:

- **English queries:** Entropy tăng tuyến tính với độ khó (phân tách rõ ràng)
- **Vietnamese queries:** Kết quả nhiễu, phân tách không rõ ràng giữa các mức độ khó

### Vấn đề:

Với query khó:
- Token đầu tiên có entropy cao (model không chắc chắn)
- Token giữa/cuối có entropy thấp (khi đã vào flow trả lời, model tự tin hơn)
- Trung bình entropy bị kéo xuống, không phản ánh đúng độ khó

### Giải pháp đề xuất:

Chỉ tính entropy từ **N token đầu tiên** (N=5):
- Token đầu phản ánh tốt hơn độ không chắc chắn ban đầu
- Token sau bị ảnh hưởng bởi phần đã generate
- Phương pháp này được sử dụng trong nhiều nghiên cứu về uncertainty estimation

## Cấu trúc Files

- `entropy_first_n_token.py`: Module tính entropy từ N token đầu tiên
- `entropy_stats_en.ipynb`: Notebook thống kê entropy cho queries tiếng Anh
- `entropy_stats_vi.ipynb`: Notebook thống kê entropy cho queries tiếng Việt
- `test_queries_en.json`: 101 test queries tiếng Anh (cùng dataset như full_token)
- `test_queries_vi.json`: 101 test queries tiếng Việt (cùng dataset như full_token)
- `entropy_analysis_results_en.csv`: Kết quả chi tiết entropy cho từng query (EN)
- `entropy_analysis_results_vi.csv`: Kết quả chi tiết entropy cho từng query (VI)

## Phương pháp

1. **Thu thập dữ liệu:**
   - Gọi API 1 lần mỗi query (n=1) để tiết kiệm token
   - Mỗi lần gọi sử dụng `generate_with_logprobs()` trực tiếp
   - Temperature = 0.01, max_tokens = 5 (chỉ generate 5 token cần thiết)

2. **Tính entropy:**
   - Chỉ lấy 5 token đầu tiên từ logprobs
   - Formula: `H = average(-logprob_i / ln(2))` cho 5 token đầu
   - Entropy từ 1 lần gọi API

3. **Thống kê:**
   - Tương tự như full_token analysis
   - So sánh với kết quả từ full_token method

## Kết quả

### English Data:
- **Entropy Range:** [0.0000, 0.5796]
- **Mean:** ~0.13
- **Median:** ~0.11
- **Observation:** 
  - Phân tách giữa easy/medium/hard không rõ ràng như full_token method
  - Entropy range rộng hơn nhưng phân bố không đồng đều
  - Nhiều easy queries có entropy rất thấp (gần 0), một số có entropy cao bất thường
  - Hard queries có entropy cao hơn nhưng overlap nhiều với medium

### Vietnamese Data:
- **Entropy Range:** [0.0, 0.6]
- **Mean:** ~0.15
- **Median:** ~0.10
- **Observation:**
  - Phân tách giữa các mức độ khó vẫn không rõ ràng
  - Không cải thiện so với full_token method
  - Entropy phân bố không có correlation rõ ràng với độ khó

## Đánh giá

### ❌ Kết quả không đạt kỳ vọng:

1. **Không có correlation rõ ràng:**
   - Entropy từ 5 token đầu không phản ánh tốt độ khó query
   - Phân tách giữa easy/medium/hard không rõ ràng hơn full_token method
   - Nhiều outliers và noise trong dữ liệu

2. **Vấn đề với phương pháp:**
   - Chỉ 5 token đầu có thể không đủ để đánh giá độ khó
   - Token đầu tiên có thể bị ảnh hưởng bởi prompt structure, không phải độ khó query
   - Với n=1, không có averaging nên kết quả không ổn định

3. **So sánh với Full Token:**
   - **Full Token (n=1):** Phân tách rõ ràng cho English, nhiễu cho Vietnamese
   - **First N Tokens (n=1):** Phân tách không rõ ràng cho cả English và Vietnamese
   - **Kết luận:** First N Tokens method không tốt hơn Full Token method

### Nguyên nhân có thể:

1. **Sample size nhỏ:** Chỉ 5 token có thể không đủ để capture uncertainty
2. **Token đầu bị ảnh hưởng:** Token đầu tiên có thể bị ảnh hưởng bởi prompt structure, style, hoặc format response
3. **Không có averaging:** Với n=1, không có trung bình qua nhiều calls nên kết quả không ổn định
4. **Model behavior:** Model có thể có pattern khác nhau ở token đầu so với token sau, nhưng pattern này không nhất thiết liên quan đến độ khó query

## Tổng kết

### Kết luận:

**Phương pháp First N Tokens (N=5, n=1) không cải thiện kết quả so với Full Token method.**

- Không giải quyết được vấn đề phân tách độ khó cho Vietnamese queries
- Làm giảm chất lượng phân tách cho English queries
- Entropy không có correlation rõ ràng với độ khó query

### Hướng đi tiếp theo:

1. **Thử nghiệm Full Token với n_iter=5:**
   - Sử dụng tất cả tokens nhưng average qua 5 lần gọi API
   - Có thể cải thiện stability và giảm noise
   - Xem thư mục `../entropy_stats_full_token_n_iter/`

2. **Các phương pháp khác:**
   - Thử nghiệm với N lớn hơn (10, 15 tokens)
   - Kết hợp entropy từ nhiều phần của response
   - Sử dụng các metrics khác ngoài entropy

3. **Tập trung vào Full Token method:**
   - Cải thiện công thức mapping entropy → k
   - Xử lý riêng cho Vietnamese queries
   - Tối ưu hóa với n_iter > 1

## Sử dụng

### Chạy notebook:
1. Chạy Cells 1-4: Thu thập dữ liệu (101 queries × 1 call = 101 API calls)
2. Chạy Cells 5-7: Đọc kết quả đã lưu và phân tích

### Xem kết quả:
- Xem CSV files để có dữ liệu chi tiết
- Xem scatter plot để thấy phân bố entropy trực quan
- So sánh với kết quả trong `entropy_stats_full_token/`

## Lưu ý

- Module `entropy_first_n_token.py` là biến thể riêng, không sửa logic trong `adaptive_controller`
- Dataset giống hệt full_token để đảm bảo so sánh công bằng
- N=5 là giá trị mặc định, đã được test nhưng không cho kết quả tốt
- Phương pháp này không được khuyến nghị sử dụng trong production
