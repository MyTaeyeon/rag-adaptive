# Hyperparameter Tuning cho Entropy-based Adaptive RAG

## 1. Tổng quan

Thư mục này chứa các thí nghiệm và phân tích để tối ưu hóa phương pháp tính entropy và mapping entropy → k trong hệ thống Adaptive RAG.

## 2. Cấu trúc thí nghiệm

### 2.1 entropy_stats_full_token

**Phương pháp Baseline:** Tính entropy từ TẤT CẢ tokens trong response

- Thí nghiệm ban đầu để xác định entropy range
- Sử dụng n=1 (1 API call per query)
- **Kết quả:**
  - EN: Entropy tăng tuyến tính với độ khó (tốt)
  - VI: Kết quả nhiễu, phân tách không rõ ràng

Xem chi tiết trong `entropy_stats_full_token/README.md`

### 2.2 entropy_stats_first_n_token

**Phương pháp cải tiến:** Tính entropy chỉ từ N token đầu tiên (N=5)

- Giải quyết vấn đề entropy bị kéo xuống bởi token sau
- Sử dụng n=1 (1 API call per query) để tiết kiệm
- **Kết quả:**
  - EN: Phân tách không rõ ràng, không cải thiện
  - VI: Kết quả tệ hơn, không có correlation rõ ràng
  - **Kết luận:** Phương pháp không hiệu quả

Xem chi tiết trong `entropy_stats_first_n_token/README.md`

### 2.3 entropy_stats_full_token_n_iter

**Phương pháp tối ưu:** Tính entropy từ TẤT CẢ tokens với n_iter=5

- Sử dụng tất cả tokens nhưng average qua 5 lần gọi API
- **Kết quả:**
  - EN: Stability cao, phân tách rõ ràng, cải thiện so với n=1
  - VI: Cải thiện đáng kể, phân tách rõ ràng hơn n=1
  - Entropy std rất thấp, chứng tỏ stability cao
  - **Kết luận:** Phương pháp tốt nhất

Xem chi tiết trong `entropy_stats_full_token_n_iter/README.md`

## 3. So sánh tổng hợp

### 3.1 Bảng so sánh các phương pháp

| Phương pháp | n_iter | Tokens sử dụng | EN Quality | VI Quality | Stability | Cost |
|-------------|--------|----------------|------------|------------|-----------|------|
| Full Token | 1 | Tất cả | Tốt | Kém | Thấp | Thấp |
| First N Tokens | 1 | 5 đầu tiên | Kém | Kém | Thấp | Thấp |
| Full Token n_iter | 5 | Tất cả | Tốt hơn | Tốt | Cao | Cao |

### 3.2 Đánh giá chi tiết

| Tiêu chí | Full Token (n=1) | First N Tokens (n=1) | Full Token (n=5) |
|----------|------------------|----------------------|------------------|
| Phân tách EN | Tốt | Kém | Tốt hơn |
| Phân tách VI | Kém | Kém | Tốt |
| Stability | Thấp | Thấp | Cao |
| Noise level | Cao | Cao | Thấp |
| Entropy std | N/A | N/A | Rất thấp |
| Correlation với độ khó | Có (EN), Không (VI) | Không | Có (cả EN và VI) |
| Chi phí API | Thấp | Thấp | Cao (5x) |

## 4. Kết luận và quyết định

### 4.1 Phương pháp được chọn

**Full Token với n_iter=5**

### 4.2 Lý do

1. **Stability cao nhất:**
   - Entropy std rất thấp qua 5 iterations
   - Curve mượt mà, ít biến thiên ngẫu nhiên
   - Kết quả ổn định và đáng tin cậy

2. **Phân tách độ khó tốt nhất:**
   - English: Phân tách rõ ràng, tốt hơn n=1
   - Vietnamese: Cải thiện đáng kể so với n=1
   - Entropy phản ánh đúng độ khó query cho cả hai ngôn ngữ

3. **Giảm noise hiệu quả:**
   - Averaging qua 5 calls loại bỏ outliers
   - Giảm ảnh hưởng của biến thiên ngẫu nhiên
   - Kết quả nhất quán hơn

4. **Giả thuyết được chứng minh:**
   - n_iter càng cao, stability của curve càng ổn định
   - Được xác nhận qua thí nghiệm với n=5

### 4.3 Trade-off

- **Chi phí:** API calls tăng 5 lần so với n=1
- **Lợi ích:** Stability và độ tin cậy tăng đáng kể
- **Đánh giá:** Trade-off hợp lý, đáng giá cho production system

### 4.4 Quyết định cuối cùng

**Sử dụng Full Token method với n_iter=5 cho hệ thống Adaptive RAG production.**

- Phương pháp này đã được chứng minh là tốt nhất qua 3 thí nghiệm
- Stability cao, phân tách độ khó rõ ràng
- Phù hợp cho cả English và Vietnamese queries
- **Kết thúc quá trình tuning entropy calculation**

## 5. Workflow thí nghiệm

1. **Bắt đầu với Full Token (n=1):**
   - Chạy thí nghiệm baseline để hiểu entropy range
   - Xác định vấn đề với phương pháp hiện tại
   - Phát hiện: EN tốt, VI nhiễu

2. **Thử nghiệm First N Tokens (n=1):**
   - Tạo biến thể tính entropy từ N token đầu
   - Sử dụng cùng dataset để so sánh công bằng
   - Kết quả: Không cải thiện, thậm chí tệ hơn

3. **Thử nghiệm Full Token (n=5):**
   - Kiểm tra giả thuyết: n_iter càng cao, stability càng tốt
   - Sử dụng tất cả tokens nhưng average qua 5 calls
   - Kết quả: Cải thiện đáng kể, stability cao

4. **Phân tích và quyết định:**
   - So sánh entropy range giữa 3 phương pháp
   - Đánh giá phân tách độ khó (easy/medium/hard)
   - Quyết định phương pháp tối ưu: Full Token (n=5)

## 6. Sử dụng

### 6.1 Chạy thí nghiệm Full Token (n=1)

```bash
cd entropy_stats_full_token
jupyter notebook entropy_stats_en.ipynb  # hoặc entropy_stats_vi.ipynb
```

### 6.2 Chạy thí nghiệm First N Tokens (n=1)

```bash
cd entropy_stats_first_n_token
jupyter notebook entropy_stats_en.ipynb  # hoặc entropy_stats_vi.ipynb
```

### 6.3 Chạy thí nghiệm Full Token (n=5)

```bash
cd entropy_stats_full_token_n_iter
jupyter notebook entropy_stats_en.ipynb  # hoặc entropy_stats_vi.ipynb
```

## 7. Lưu ý

- Mỗi thí nghiệm có README riêng với chi tiết phương pháp và kết quả
- Dataset được chia sẻ giữa các thí nghiệm để đảm bảo so sánh công bằng
- Kết quả được lưu vào CSV để phân tích sau
- Phương pháp cuối cùng (Full Token n=5) được khuyến nghị cho production
