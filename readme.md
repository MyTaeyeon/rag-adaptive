# Adaptive RAG System

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

**Backend (FastAPI):**
```bash
python run_backend.py
```
Backend sẽ chạy tại `http://localhost:8000`

**Frontend (Streamlit):**
```bash
streamlit run frontend/app.py
```
Frontend sẽ tự động mở trong trình duyệt

---

## Project Overview

Adaptive RAG (Retrieval-Augmented Generation) System là một hệ thống RAG thông minh với khả năng tự động điều chỉnh số lượng tài liệu cần truy xuất dựa trên đặc điểm của câu truy vấn. Hệ thống sử dụng pipeline hybrid retrieval kết hợp giữa sparse retrieval (BM25) và dense retrieval (semantic embeddings) để đạt được độ chính xác cao trong việc tìm kiếm tài liệu liên quan.

## RAG Pipeline

Hệ thống thực hiện pipeline 5 bước để xử lý mỗi câu truy vấn:

### 1. Query Rewriting
Sử dụng **GPT-4o** để viết lại và mở rộng câu truy vấn của người dùng, giúp cải thiện hiệu suất retrieval bằng cách:
- Giữ nguyên ý nghĩa và mục đích ban đầu
- Mở rộng từ khóa quan trọng với các từ đồng nghĩa
- Làm rõ ý định nếu truy vấn mơ hồ

### 2. Adaptive K Selection
Tự động xác định số lượng tài liệu cần truy xuất (k) dựa trên:
- Độ dài câu truy vấn (số tokens)
- Entropy (độ đa dạng từ vựng)
- Độ phức tạp của câu hỏi

Hệ thống sử dụng heuristic algorithm để điều chỉnh k trong khoảng từ k_min đến k_max, đảm bảo truy xuất đủ tài liệu cho các câu hỏi phức tạp nhưng không lãng phí tài nguyên cho câu hỏi đơn giản.

### 3. Hybrid Retrieval
Kết hợp hai phương pháp retrieval để tận dụng ưu điểm của cả hai:

**Sparse Retrieval (BM25):**
- Tìm kiếm dựa trên từ khóa, phù hợp cho exact matching
- Hiệu quả với các truy vấn có từ khóa cụ thể

**Dense Retrieval (Semantic Embeddings):**
- Sử dụng model chuyên biệt cho từng ngôn ngữ:
  - **Tiếng Anh**: `all-mpnet-base-v2` - Model chất lượng cao cho English
  - **Tiếng Việt**: `vietnamese-sbert-v2` - Model được fine-tune cho Vietnamese
- Tìm kiếm dựa trên ngữ nghĩa, phù hợp cho semantic similarity
- Hiệu quả với các truy vấn cần hiểu ngữ cảnh và từ đồng nghĩa

Kết quả từ hai phương pháp được kết hợp bằng **Reciprocal Rank Fusion (RRF)** để tạo ra danh sách tài liệu được xếp hạng tốt nhất.

### 4. Reranking
Sử dụng **Cross-Encoder reranker (cross-encoder/ms-marco-MiniLM-L-12-v2)** để đánh giá lại độ liên quan của các tài liệu đã được retrieve. Cross-Encoder xử lý query-document pair cùng lúc, cho độ chính xác cao hơn so với cosine similarity đơn thuần. Model này hỗ trợ tốt cả tiếng Anh và tiếng Việt.

### 5. Answer Generation
Sử dụng **GPT-4o** để sinh câu trả lời dựa trên các tài liệu đã được retrieve và rerank. LLM được hướng dẫn:
- Chỉ sử dụng thông tin từ các tài liệu được cung cấp
- Trích dẫn nguồn tài liệu khi có thể
- Trả lời tự nhiên và dễ hiểu

## AI Models & Technologies

Hệ thống sử dụng **language-aware model selection** để tự động chọn model tối ưu cho từng ngôn ngữ (tiếng Anh và tiếng Việt), đảm bảo hiệu suất tốt nhất cho cả hai ngôn ngữ.

### Model Selection Strategy

| Pipeline Step | Tiếng Anh (EN) | Tiếng Việt (VI) | Lý do lựa chọn |
|---------------|----------------|-----------------|----------------|
| **Query Rewriting** | GPT-4o | GPT-4o | GPT-4o hỗ trợ tốt cả hai ngôn ngữ, có khả năng viết lại và mở rộng câu hỏi hiệu quả |
| **Dense Embedding** | all-mpnet-base-v2 | vietnamese-sbert-v2 | Model chuyên biệt cho từng ngôn ngữ đảm bảo hiệu suất tối ưu |
| **Semantic Chunking** | all-mpnet-base-v2 | vietnamese-sbert-v2 | Dùng chung với embedding model để đảm bảo consistency |
| **Reranker** | ms-marco-L-12-v2 | ms-marco-L-12-v2 | Multilingual model với chất lượng cao cho cả hai ngôn ngữ |
| **Answer Generation** | GPT-4o | GPT-4o | GPT-4o sinh câu trả lời tự nhiên và chính xác cho cả hai ngôn ngữ |
| **Sparse Retrieval** | BM25 | BM25 | BM25 hoạt động tốt cho cả hai ngôn ngữ với normalization phù hợp |

### Large Language Models (LLMs)

**GPT-4o (OpenAI)**
- **Sử dụng**: Query rewriting và answer generation cho cả tiếng Anh và tiếng Việt
- **Query Rewriting**: 
  - Temperature: 0.3 (focused, deterministic)
  - Max tokens: 200
  - Mục đích: Viết lại và mở rộng câu hỏi để cải thiện retrieval
- **Answer Generation**:
  - Temperature: 0.7 (balanced creativity and accuracy)
  - Max tokens: 1000
  - Mục đích: Sinh câu trả lời dựa trên context đã retrieve

**Ưu điểm**:
- Hỗ trợ đa ngôn ngữ xuất sắc, đặc biệt là tiếng Việt
- Hiểu ngữ cảnh và ngữ nghĩa tốt
- Sinh văn bản tự nhiên và chính xác

### Embedding Models

#### Tiếng Anh: `sentence-transformers/all-mpnet-base-v2`

**Đặc điểm**:
- Model chất lượng cao, được train trên large English corpus
- Vector dimension: 768
- Hiệu suất tốt cho semantic similarity search
- Tốc độ: Trung bình (chậm hơn MiniLM nhưng nhanh hơn các model lớn)

**Ưu điểm**:
- Độ chính xác cao cho semantic retrieval
- Hiểu tốt từ đồng nghĩa và ngữ cảnh
- Phù hợp cho RAG systems

**Nhược điểm**:
- Tốc độ chậm hơn các model nhẹ hơn
- Yêu cầu GPU để inference nhanh

#### Tiếng Việt: `keepitreal/vietnamese-sbert-v2`

**Đặc điểm**:
- Model được fine-tune chuyên biệt cho tiếng Việt
- Xử lý tốt dấu thanh và từ ghép tiếng Việt
- Hiệu suất cao trên Vietnamese benchmarks

**Ưu điểm**:
- **Tối ưu cho tiếng Việt**: Được train trên Vietnamese corpus
- **Xử lý dấu thanh tốt**: Hiểu được sự khác biệt giữa "má", "mà", "mả"
- **Từ ghép**: Xử lý tốt cấu trúc từ ghép tiếng Việt
- **Chất lượng demo tốt**: Cho kết quả retrieval chính xác cho tiếng Việt

**Nhược điểm**:
- Chỉ hỗ trợ tiếng Việt, không dùng được cho ngôn ngữ khác
- Có thể cần thời gian download model lần đầu

**Alternative models cho tiếng Việt**:
- `dangvantuan/vietnamese-embedding`: Model phổ biến, ổn định
- `bge-base-zh-v1.5`: Multilingual model, hỗ trợ tiếng Việt nhưng không chuyên biệt

### Reranking Models

**Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-12-v2`**

**Đặc điểm**:
- Cross-encoder model xử lý query-document pair cùng lúc
- Được train trên MS MARCO dataset (multilingual)
- Vector dimension: 384
- Tốc độ: Nhanh hơn các model lớn hơn

**Ưu điểm**:
- **Độ chính xác cao**: Cross-encoder cho kết quả chính xác hơn cosine similarity
- **Multilingual**: Hỗ trợ tốt cả tiếng Anh và tiếng Việt
- **Cân bằng tốc độ/chất lượng**: Tốc độ nhanh với chất lượng tốt

**So sánh với alternatives**:
- `ms-marco-MiniLM-L-6-v2`: Nhanh hơn nhưng chất lượng thấp hơn
- `ms-marco-electra-base`: Chất lượng cao hơn nhưng chậm hơn đáng kể

### Retrieval Algorithms

**BM25 (Best Matching 25)**
- **Sparse retrieval**: Dựa trên term frequency và inverse document frequency
- **Ưu điểm**: 
  - Tốc độ rất nhanh, không cần GPU
  - Hiệu quả cho exact matching và keyword search
  - Hoạt động tốt cho cả tiếng Anh và tiếng Việt với normalization phù hợp
- **Nhược điểm**:
  - Không hiểu ngữ nghĩa (semantic)
  - Không xử lý từ đồng nghĩa
  - Phụ thuộc vào tokenization

**Reciprocal Rank Fusion (RRF)**
- **Mục đích**: Kết hợp kết quả từ BM25 và Dense retrieval
- **Ưu điểm**: 
  - Tận dụng ưu điểm của cả hai phương pháp
  - Cải thiện recall và precision
  - Robust với các loại queries khác nhau
- **Parameter**: k_rrf = 60 (smoothing parameter)

### Semantic Chunking

Hệ thống sử dụng **semantic chunking** thay vì fixed-size chunking để tạo ra các chunks có ý nghĩa hơn:

**Cơ chế**:
1. Tính toán embeddings cho từng câu trong document
2. Tính cosine similarity giữa các câu liên tiếp
3. Nhóm các câu có similarity cao vào cùng một chunk
4. Tách chunk khi similarity thấp hoặc đạt max_chunk_size

**Cấu hình theo ngôn ngữ**:
- **Tiếng Anh**: 
  - Similarity threshold: 0.7
  - Min chunk size: 64 characters
  - Max chunk size: 512 characters
- **Tiếng Việt**:
  - Similarity threshold: 0.65 (thấp hơn do cấu trúc câu khác)
  - Min chunk size: 64 characters
  - Max chunk size: 512 characters

**Ưu điểm**:
- Chunks có ý nghĩa và liên kết về mặt ngữ nghĩa
- Giảm noise khi retrieve
- Cải thiện chất lượng context cho LLM

### Phân tích Model Selection

#### Tại sao chọn model chuyên biệt cho từng ngôn ngữ?

**Embedding Models**:
- **Tiếng Anh**: `all-mpnet-base-v2` được train chủ yếu trên English corpus, cho hiệu suất tốt nhất cho tiếng Anh
- **Tiếng Việt**: `vietnamese-sbert-v2` được fine-tune trên Vietnamese data, hiểu rõ đặc thù của tiếng Việt (dấu thanh, từ ghép, cấu trúc câu)

**Kết quả**:
- **Tiếng Anh**: Độ chính xác retrieval cao, hiểu tốt ngữ nghĩa và từ đồng nghĩa
- **Tiếng Việt**: Demo cho kết quả tốt, xử lý chính xác các truy vấn tiếng Việt, không bị nhầm lẫn bởi dấu thanh

**Reranker**:
- Dùng chung model multilingual vì cross-encoder models thường được train trên multilingual data và hoạt động tốt cho cả hai ngôn ngữ
- `ms-marco-L-12-v2` là sự cân bằng tốt giữa chất lượng và tốc độ

**LLM**:
- GPT-4o hỗ trợ xuất sắc cả hai ngôn ngữ, không cần model riêng
- Temperature và max_tokens giống nhau cho cả hai ngôn ngữ vì GPT-4o xử lý tốt cả hai

## Tech Stack

- **Backend**: FastAPI - RESTful API framework cho Python
- **Frontend**: Streamlit - Interactive web application framework

## Configuration

Tất cả các tham số có thể được cấu hình trong `backend/config/config.py`, bao gồm:
- Model names và parameters
- Chunking thresholds và sizes
- Adaptive k selection ranges
- Temperature và max_tokens cho LLM
- Reranker và fusion parameters
