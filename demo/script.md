# KỊCH BẢN DEMO – DOMAIN AI & ĐỜI SỐNG

*(Tài liệu PDF tiếng Anh, < 5MB, nhưng demo – thuyết minh hoàn toàn bằng tiếng Việt)*

---

## 📄 TÀI LIỆU PDF SỬ DỤNG (2 FILE)=

1. **Introduction to Artificial Intelligence**

   * AI là gì
   * Các lĩnh vực ứng dụng
   * Lịch sử phát triển cơ bản

2. **Ethical Issues in Artificial Intelligence**

   * Bias (thiên lệch)
   * Privacy (quyền riêng tư)
   * Transparency (tính minh bạch)

👉 Hai tài liệu **cùng domain nhưng khác góc nhìn**

---

## 🧩 TỔNG QUAN FLOW 

![Image](https://miro.medium.com/v2/resize%3Afit%3A1400/0%2AMwqEsP6YWzxVmaPT.gif)

![Image](https://miro.medium.com/1%2AKJz54o_gzMRTaxs4kkfaSg.png)

![Image](https://www.k2view.com/hs-fs/hubfs/K2%20RAG%20diagram%40150x-100.jpg?height=913\&name=K2+RAG+diagram%40150x-100.jpg\&width=2094)

# 🟢 MỨC 1 – CÂU HỎI DỄ (KHÔNG CẦN RAG)

👉 **Mục tiêu:** Chứng minh *retrieval là không cần thiết với câu hỏi phổ thông*

### 4–5 câu 

### Câu 1

> **“AI là viết tắt của từ gì?”**

### Câu 2

> **“Artificial Intelligence nghĩa là gì?”**

### Câu 3

> **“AI có phải là trí thông minh của con người không?”**

### Câu 4

> **“AI có phải là một lĩnh vực của khoa học máy tính không?”**

### (Có thể thêm)

> **“AI có tồn tại trước Internet không?”**

---

### 🧠 Hệ thống xử lý (giống nhau cho các câu dễ)

* Câu hỏi:

  * Ngắn
  * Định nghĩa trực tiếp
* Entropy: **thấp**
* **Adaptive Controller quyết định:**

  * `k = 0`
  * ❌ Không truy hồi PDF
  * ✅ LLM trả lời trực tiếp

### 🖥️ Giao diện hiển thị:

* Entropy: Thấp
* Retrieved docs: **0**
* Latency: rất thấp
* Token cost: thấp

### 🎓 Nói với thầy cô:

> “Nếu với những câu hỏi này mà hệ thống vẫn truy hồi tài liệu,
> thì đó là **lãng phí tài nguyên**.
> Adaptive RAG giúp **loại bỏ retrieval không cần thiết**.”

---

# 🔵 MỨC 2 – CÂU HỎI TRUNG BÌNH (CẦN RAG NHẸ)

👉 **Mục tiêu:** Thể hiện Adaptive k không phải chỉ có 0 hoặc rất lớn

### 🎤 Demo 3–4 câu

### Câu 1

> **“AI hiện nay được ứng dụng trong những lĩnh vực nào?”**

### Câu 2

> **“AI được sử dụng trong đời sống hằng ngày ra sao?”**

### Câu 3

> **“Những ví dụ phổ biến của AI trong xã hội hiện đại là gì?”**

### Câu 4 (tuỳ chọn)

> **“AI ảnh hưởng như thế nào đến công việc của con người?”**

---

### 🧠 Hệ thống xử lý:

* Câu hỏi:

  * Không quá khó
  * Nhưng cần **liệt kê / tổng hợp**
* Entropy: **trung bình**
* Adaptive Controller:

  * `k = 2–3`

### 🔍 Retrieval:

* BM25: tìm “applications”, “AI use”
* ColBERT: hiểu “impact”, “daily life”
* RRF hợp nhất

### 🎓 Nói:

> “Với nhóm câu hỏi này, hệ thống **vẫn truy hồi**,
> nhưng **chỉ lấy ít tài liệu**, đủ để tổng hợp thông tin.”

---

# 🔴 MỨC 3 – CÂU HỎI KHÓ (BẮT BUỘC RAG)

👉 **Mục tiêu:** Thể hiện rõ sức mạnh Adaptive + Hybrid

### 🎤 Demo 2–3 câu

### Câu 1

> **“Những vấn đề đạo đức chính của trí tuệ nhân tạo là gì?”**

### Câu 2

> **“Vì sao AI có thể gây ra thiên lệch (bias) trong quyết định?”**

### Câu 3

> **“AI ảnh hưởng như thế nào đến quyền riêng tư của con người?”**

---

### 🧠 Hệ thống xử lý:

* Câu hỏi trừu tượng
* Cần:

  * Định nghĩa
  * Giải thích
  * Ví dụ
* Entropy: **cao**
* Adaptive Controller:

  * `k = 6–8`

### 🔍 Retrieval:

* BM25 → “ethical issues”, “privacy”
* ColBERT → “social impact”, “fairness”
* RRF → chọn tài liệu đa góc nhìn

### 🎓 Nói:

> “Nếu chỉ dùng một tài liệu hoặc k nhỏ,
> câu trả lời sẽ **thiếu chiều sâu**.
> Adaptive RAG tự động tăng k để đảm bảo chất lượng.”

---

# ⚫ MỨC 4 – CÂU HỎI NHIỄU (ANTI-HALLUCINATION)

👉 **Mục tiêu:** Chứng minh hệ thống không bịa

### 🎤 Demo 1–2 câu

> **“AI có đang bí mật điều khiển chính phủ các nước không?”**
> **“AI có ý thức và cảm xúc như con người không?”**

### 🧠 Hệ thống:

* Entropy cao
* Truy hồi nhưng **không có bằng chứng**
* Trả lời: từ chối / trung lập

### 🎓 Nói:

> “Hệ thống **không cố trả lời cho mọi câu hỏi**,
> đây là yếu tố rất quan trọng trong các hệ thống AI thực tế.”