# Adaptive RAG System - Documentation

## 1. Tổng quan dự án

### 1.1. Giới thiệu

Adaptive RAG (Retrieval-Augmented Generation) System là một hệ thống RAG thông minh có khả năng tự động điều chỉnh số lượng tài liệu cần truy xuất dựa trên đặc điểm của câu truy vấn. Hệ thống sử dụng pipeline hybrid retrieval kết hợp giữa sparse retrieval (BM25) và dense retrieval (semantic embeddings) để đạt được độ chính xác cao trong việc tìm kiếm tài liệu liên quan.

### 1.2. Kiến trúc hệ thống

- **Backend**: FastAPI - RESTful API framework cho Python
- **Frontend**: Streamlit - Interactive web application framework
- **Storage**: In-memory collections (có thể mở rộng sang vector database)
- **Models**: Language-aware model selection cho tiếng Anh và tiếng Việt

### 1.3. Các thành phần chính

- **Collection Management**: Quản lý các collection tài liệu
- **Document Processing**: Xử lý và chunking tài liệu
- **Adaptive Controller**: Điều khiển thông minh số lượng documents cần retrieve
- **Hybrid Retrieval**: Kết hợp BM25 và Dense retrieval
- **Reranking**: Đánh giá lại độ liên quan
- **Answer Generation**: Sinh câu trả lời từ LLM

## 2. Flow tổng quan của hệ thống

### 2.1. Flow chính của hệ thống

```
[User Query]
     |
     v
[Query Rewriting] (GPT-4o)
     |
     v
[Adaptive K Selection] (Entropy-based)
     |
     v
[Hybrid Retrieval]
     |----------> [BM25 Retrieval]
     |----------> [Dense Retrieval]
     |
     v
[Reciprocal Rank Fusion (RRF)]
     |
     v
[Cross-Encoder Reranking]
     |
     v
[Answer Generation] (GPT-4o / Gemini Flash Lite)
     |
     v
[Response to User]
```

### 2.2. Các phase chính

**Phase 1: Document Ingestion**
- Upload documents (PDF, DOCX, TXT)
- Semantic chunking hoặc sentence-based chunking
- Tạo embeddings cho chunks
- Xây dựng indices cho BM25 và Dense retrieval

**Phase 2: Query Processing**
- Query rewriting để cải thiện retrieval
- Adaptive k selection dựa trên entropy
- Hybrid retrieval (BM25 + Dense)
- Fusion và reranking

**Phase 3: Answer Generation**
- Sinh câu trả lời từ retrieved context
- Có thể trả lời không cần context nếu câu hỏi đơn giản
- Trích dẫn nguồn khi có thể

## 3. Chi tiết các chức năng Backend

### 3.1. API Endpoints

#### 3.1.1. Collection Management

**POST /collections**
- Tạo collection mới
- Parameters:
  - `name`: Tên collection (bắt buộc)
  - `language`: "en" hoặc "vi" (mặc định: "en")
  - `dense_model_name`: Tên model cho dense retrieval (optional)
  - `chunking_model_name`: Tên model cho chunking (optional)
  - `similarity_threshold`: Ngưỡng similarity cho chunking (optional)
- Response: `{"message": "Collection 'name' created"}`

**DELETE /collections/{name}**
- Xóa collection
- Response: `{"message": "Collection 'name' deleted"}`

**GET /collections**
- Liệt kê tất cả collection names
- Response: `List[str]`

**GET /collections/{name}/info**
- Lấy thông tin collection
- Response:
```json
{
  "name": "string",
  "language": "string",
  "num_chunks": 0,
  "similarity_threshold": 0.0
}
```

**GET /collections/{name}/chunks**
- Lấy danh sách chunks của collection
- Parameters:
  - `skip`: Số chunks bỏ qua (mặc định: 0)
  - `limit`: Số chunks tối đa trả về (mặc định: 100)
- Response:
```json
{
  "total": 0,
  "chunks": [],
  "skip": 0,
  "limit": 0
}
```

#### 3.1.2. Document Upload

**POST /collections/{name}/documents**
- Upload documents vào collection
- Parameters:
  - `files`: List[UploadFile] - Danh sách files (PDF, DOCX, TXT)
  - `use_semantic_chunking`: bool (mặc định: true)
- Response:
```json
{
  "message": "string",
  "num_files": 0,
  "num_chunks": 0
}
```
- Xử lý:
  - Đọc text từ files
  - Semantic chunking hoặc sentence-based chunking
  - Tạo embeddings
  - Cập nhật indices

#### 3.1.3. Query Processing

**POST /collections/{name}/query**
- Thực hiện full RAG pipeline
- Request Body:
```json
{
  "query": "string",
  "language": "string",  // Optional: Override collection language
  "n": 0  // Optional: Số iterations cho adaptive (1-10)
}
```
- Response:
```json
{
  "original_query": "string",
  "rewritten_query": "string",
  "k": 0,
  "entropy": 0.0,
  "results": [],
  "answer": "string",
  "sources": [],
  "pipeline_steps": {}
}
```
- Pipeline steps bao gồm:
  - `query_rewriting`: Thời gian, original/rewritten query, model
  - `adaptive_k_selection`: k, entropy, iterations detail
  - `retrieval`: Số lượng results, preview documents
  - `answer_generation`: Model, số chunks sử dụng

**POST /collections/{name}/query/rewrite-only**
- Chỉ thực hiện query rewriting
- Response:
```json
{
  "original_query": "string",
  "rewritten_query": "string",
  "model": "string"
}
```

### 3.2. Core Components

#### 3.2.1. Collection Class

**Location**: `backend/models/collection.py`

**Chức năng:**
- Quản lý documents và chunks
- Khởi tạo retrievers (BM25, Dense) theo language
- Khởi tạo adaptive controller
- Khởi tạo reranker theo language
- Xử lý semantic chunking

**Methods:**

- `__init__(name, language, dense_model_name, chunking_model_name, similarity_threshold)`
  - Tự động chọn model theo language
  - Khởi tạo BM25Retriever, DenseRetriever
  - Khởi tạo AdaptiveController
  - Khởi tạo CrossEncoderReranker

- `add_documents(docs, meta, use_semantic_chunking)`
  - Nhận danh sách documents (full text)
  - Semantic chunking hoặc sentence-based chunking
  - Tạo metadata cho mỗi chunk
  - Cập nhật BM25 và Dense indices

- `query(query, use_rewritten_query, n)`
  - Adaptive k selection
  - Hybrid retrieval (BM25 + Dense)
  - RRF fusion
  - Reranking
  - Trả về results với scores

#### 3.2.2. Adaptive Controller

**Location**: `backend/models/adaptive_controller.py`

**Chức năng:**
- Xác định số lượng documents cần retrieve (k) dựa trên entropy
- Sử dụng n iterations với các styles khác nhau
- Tính toán average entropy từ logprobs

**Methods:**

- `decide(query, language, n)`
  - Phase 1: Chạy n iterations với different styles
  - Mỗi iteration: LLM call không có context (non-hop)
  - Tính entropy từ logprobs
  - Tính average entropy
  - Chuyển đổi entropy thành k value
  - Trả về (k, metadata)

**Algorithm:**
1. Chọn n styles từ style_candidates (cycle nếu n > len(styles))
2. Mỗi style: Build style prompt = style + query
3. LLM generate với logprobs (temperature thấp: 0.01)
4. Tính entropy từ logprobs: `H = average(-logprob / ln(2))`
5. Average entropy từ n iterations
6. Map entropy -> k: `k = k_min + (entropy/entropy_max) * (k_max - k_min)`

**Metadata trả về:**
- `average_entropy`: Entropy trung bình
- `k_determined`: Giá trị k được xác định
- `iterations_detail`: Chi tiết từng iteration (style, output, entropy)
- `n`: Số iterations
- `k_min`, `k_max`: Giới hạn k

#### 3.2.3. Retrievers

**Location**: `backend/models/retrievers.py`

**BM25Retriever:**
- Implementation: Simple BM25 algorithm
- Parameters: k1=1.5, b=0.75
- Language support: Normalize text theo language trước khi tính toán
- Methods:
  - `add_documents(docs)`: Xây dựng index (doc_freqs, idf, doc_len)
  - `retrieve(query, k)`: Trả về top-k documents với BM25 scores

**DenseRetriever:**
- Implementation: SentenceTransformer embeddings
- Language-aware model selection
- Fallback: TF-IDF nếu SentenceTransformer không available
- Methods:
  - `add_documents(docs)`: Compute embeddings cho documents
  - `retrieve(query, k)`:
    - Embed query
    - Cosine similarity với document embeddings
    - Trả về top-k documents

#### 3.2.4. Reranker

**Location**: `backend/models/reranker.py`

**CrossEncoderReranker:**
- Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Language-aware: Tự động chọn model theo language
- Chức năng: Score query-document pairs cùng lúc
- Methods:
  - `rerank(query, docs)`: Trả về scores cho mỗi document

#### 3.2.5. Fusion

**Location**: `backend/models/fusion.py`

**Reciprocal Rank Fusion (RRF):**
- Kết hợp kết quả từ nhiều retrieval methods
- Formula: `score = sum(1 / (k_rrf + rank + 1))`
- Parameters: k_rrf = 60 (smoothing parameter)
- Methods:
  - `reciprocal_rank_fusion(ranked_lists, final_k, k_rrf)`:
    - Tính RRF score cho mỗi document
    - Sort theo score giảm dần
    - Trả về top final_k documents

### 3.3. Utilities

#### 3.3.1. Document Processing

**Location**: `backend/utils/document_processing.py`

**Chức năng:**
- Đọc text từ các định dạng file khác nhau
- Support: PDF, DOCX, TXT

**Functions:**
- `extract_text_from_pdf(file_data)`: Sử dụng pypdf
- `extract_text_from_docx(file_data)`: Sử dụng python-docx
- `read_file_to_text(file)`: Tự động detect format và extract

#### 3.3.2. Text Processing

**Location**: `backend/utils/text_processing.py`

**Functions:**

- `split_sentences(text, language)`:
  - Tách text thành sentences
  - Language-specific patterns cho EN/VI

- `semantic_chunk(text, embedding_model, similarity_threshold, min_chunk_size, max_chunk_size, language)`:
  - Tách text thành chunks dựa trên semantic similarity
  - Process:
    1. Split thành sentences
    2. Compute embeddings cho mỗi sentence
    3. Tính cosine similarity giữa consecutive sentences
    4. Group sentences khi similarity >= threshold
    5. Split khi similarity < threshold hoặc vượt max_chunk_size
    6. Merge small chunks với neighbors

#### 3.3.3. Normalization

**Location**: `backend/utils/normalization.py`

**Functions:**

- `normalize_text(text, language)`:
  - English: Lowercase only
  - Vietnamese: Remove diacritics + lowercase
  - Sử dụng unicodedata để remove accents

#### 3.3.4. Entropy Calculation

**Location**: `backend/utils/entropy.py`

**Functions:**

- `sequence_entropy_from_token_logprobs(token_logprobs)`:
  - Tính entropy từ logprobs (natural log)
  - Formula: `H = average(-logprob / ln(2))`
  - Unit: bits/token

- `entropy_to_k(average_entropy, k_min, k_max, entropy_max)`:
  - Chuyển đổi entropy thành k value
  - Formula: `k = k_min + (entropy/entropy_max) * (k_max - k_min)`
  - Clamp về [k_min, k_max]

### 3.4. Services

#### 3.4.1. LLM Service

**Location**: `backend/services/llm_service.py`

**Functions:**

- `rewrite_query(original_query, language, model)`:
  - GPT-4o, prompts riêng cho EN/VI
  - Temperature: 0.3, Max tokens: 200

- `generate_with_logprobs(query, style_prompt, language, model, temperature, max_tokens)`:
  - Non-hop call cho adaptive controller, lấy logprobs để tính entropy
  - Temperature: 0.01, Max tokens: 64

- `generate_answer(query, context_chunks, language, model, provider)`:
  - Answer generation với OpenAI GPT-4o hoặc Google Gemini 2.5 Flash Lite (mặc định Gemini Flash Lite)
  - Temperature: 0.7, Max tokens: 50000
  - Prompt được tối ưu: thân thiện, đúng trọng tâm, không icon/emoji, Markdown rõ ràng, hỗ trợ EN/VI, hiểu bối cảnh Adaptive RAG (có/không có context)
  - Logging: mỗi lần generate ghi `logs/generation_<timestamp>.json` (system/user prompt, answer, provider/model, error nếu có)

## 4. Giao diện Frontend

### 4.1. Cấu trúc giao diện

**Location**: `frontend/app.py`

**Layout:**
- Sidebar: Collection management
- Main area: Chat interface

### 4.2. Sidebar Components

#### 4.2.1. Collection Management

- **Create Collection:**
  - Input: Collection name
  - Select: Language (en/vi)
  - Button: Create
  - Hiển thị success/error message

- **Select Collection:**
  - Dropdown: Danh sách collections
  - Hiển thị: Language, Số lượng chunks
  - Adaptive Iterations selector (n): 1-10

- **Upload Documents:**
  - File uploader: PDF, DOCX, TXT (multiple files)
  - Button: Upload
  - Progress bar khi uploading
  - Hiển thị số files và chunks sau khi upload

- **Chunks Preview:**
  - Hiển thị danh sách chunks (tối đa 100)
  - Mỗi chunk: Index, số tokens, source, full text
  - Expandable sections

- **Delete Collection:**
  - Button: Delete Collection
  - Confirm dialog (click 2 lần)

### 4.3. Main Chat Interface

#### 4.3.1. Chat Messages

- Hiển thị lịch sử chat (user + assistant)
- User messages: Hiển thị query
- Assistant messages:
  - Streamed answer (word-by-word animation)
  - Total time
  - Expandable "Pipeline Steps" section

#### 4.3.2. Pipeline Steps Display

**Step 1: Query Rewriting**
- Original query
- Rewritten query
- Time taken
- Model used

**Step 2: Adaptive K Selection**
- K selected (metric)
- Average entropy (metric)
- Number of iterations
- Iterations detail (expandable):
  - Mỗi iteration: Run number, style, response, entropy

**Step 3: Retrieval**
- Number of results found
- Preview documents (expandable):
  - Mỗi document: Index, score, source, full text

**Step 4: Answer Generation**
- Model used
- Number of chunks used
- Time taken

### 4.4. User Interactions

- Chat input: "Ask a question..."
- Khi submit:
  - Hiển thị user message
  - Spinner: "Processing..."
  - Gọi API với adaptive_n từ sidebar
  - Hiển thị answer (streamed)
  - Hiển thị pipeline steps
  - Lưu vào session state

### 4.5. State Management

**Session state variables:**
- `messages`: List[Dict] - Chat history
- `selected_collection`: str - Current collection
- `chunks_loaded`: Dict - Cached chunks data
- `adaptive_n`: int - Number of iterations (1-10)

## 5. Phân tích Model Selection cho tiếng Anh và tiếng Việt

### 5.1. Chiến lược lựa chọn model

Hệ thống sử dụng language-aware model selection để tự động chọn model tối ưu cho từng ngôn ngữ. Mỗi bước trong pipeline có thể sử dụng model khác nhau tùy theo ngôn ngữ.

### 5.2. Bảng so sánh model selection

| Pipeline Step | Tiếng Anh (EN) | Tiếng Việt (VI) | Lý do lựa chọn |
|---------------|----------------|-----------------|---------------|
| **Query Rewriting** | GPT-4o | GPT-4o | GPT-4o hỗ trợ tốt cả hai ngôn ngữ |
| **Dense Embedding** | all-mpnet-base-v2 | vietnamese-sbert-v2 | Model chuyên biệt cho từng ngôn ngữ đảm bảo hiệu suất tối ưu |
| **Semantic Chunking** | all-mpnet-base-v2 | vietnamese-sbert-v2 | Dùng chung với embedding model để đảm bảo consistency |
| **Reranker** | ms-marco-L-12-v2 | ms-marco-L-12-v2 | Multilingual model với chất lượng cao cho cả hai ngôn ngữ |
| **Answer Generation** | GPT-4o | GPT-4o | GPT-4o sinh câu trả lời tự nhiên và chính xác cho cả hai ngôn ngữ |
| **Sparse Retrieval** | BM25 | BM25 | BM25 hoạt động tốt cho cả hai ngôn ngữ với normalization phù hợp |

### 5.3. Chi tiết model cho tiếng Anh

#### 5.3.1. Dense Embedding: all-mpnet-base-v2

- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Vector dimension**: 768
- **Training**: Large English corpus
- **Ưu điểm:**
  - Độ chính xác cao cho semantic retrieval
  - Hiểu tốt từ đồng nghĩa và ngữ cảnh
  - Phù hợp cho RAG systems
- **Nhược điểm:**
  - Tốc độ chậm hơn các model nhẹ hơn
  - Yêu cầu GPU để inference nhanh

#### 5.3.2. Reranker: ms-marco-MiniLM-L-12-v2

- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Vector dimension**: 384
- **Training**: MS MARCO dataset (multilingual)
- **Ưu điểm:**
  - Độ chính xác cao (cross-encoder)
  - Multilingual support
  - Cân bằng tốc độ/chất lượng

### 5.4. Chi tiết model cho tiếng Việt

#### 5.4.1. Dense Embedding: vietnamese-sbert-v2

- **Model**: `keepitreal/vietnamese-sbert-v2`
- **Training**: Fine-tuned chuyên biệt cho tiếng Việt
- **Ưu điểm:**
  - Tối ưu cho tiếng Việt: Train trên Vietnamese corpus
  - Xử lý dấu thanh tốt: Hiểu sự khác biệt giữa "má", "mà", "mả"
  - Từ ghép: Xử lý tốt cấu trúc từ ghép tiếng Việt
  - Chất lượng demo tốt: Cho kết quả retrieval chính xác
- **Nhược điểm:**
  - Chỉ hỗ trợ tiếng Việt, không dùng được cho ngôn ngữ khác
  - Có thể cần thời gian download model lần đầu

#### 5.4.2. Reranker: ms-marco-MiniLM-L-12-v2

- Cùng model như tiếng Anh
- Multilingual model hoạt động tốt cho cả hai ngôn ngữ

### 5.5. Lý do lựa chọn model chuyên biệt

#### 5.5.1. Embedding Models

- **Tiếng Anh**: `all-mpnet-base-v2` được train chủ yếu trên English corpus, cho hiệu suất tốt nhất cho tiếng Anh
- **Tiếng Việt**: `vietnamese-sbert-v2` được fine-tune trên Vietnamese data, hiểu rõ đặc thù của tiếng Việt (dấu thanh, từ ghép, cấu trúc câu)

**Kết quả:**
- **Tiếng Anh**: Độ chính xác retrieval cao, hiểu tốt ngữ nghĩa và từ đồng nghĩa
- **Tiếng Việt**: Demo cho kết quả tốt, xử lý chính xác các truy vấn tiếng Việt, không bị nhầm lẫn bởi dấu thanh

#### 5.5.2. Reranker

- Dùng chung model multilingual vì cross-encoder models thường được train trên multilingual data và hoạt động tốt cho cả hai ngôn ngữ
- `ms-marco-L-12-v2` là sự cân bằng tốt giữa chất lượng và tốc độ

#### 5.5.3. LLM

- GPT-4o hỗ trợ xuất sắc cả hai ngôn ngữ, không cần model riêng
- Temperature và max_tokens giống nhau cho cả hai ngôn ngữ vì GPT-4o xử lý tốt cả hai

### 5.6. Normalization strategies

#### 5.6.1. Tiếng Anh

- Chỉ lowercase: `text.lower()`
- Lý do: Tiếng Anh không có dấu thanh, case-insensitive matching là đủ

#### 5.6.2. Tiếng Việt

- Remove diacritics + lowercase
- Sử dụng unicodedata để remove accents
- Lý do:
  - Tiếng Việt có nhiều dấu thanh (á, à, ả, ã, ạ, ă)
  - Remove accents giúp tìm kiếm không phụ thuộc vào dấu thanh
  - Cải thiện recall cho retrieval

## 6. Flow chi tiết từng Phase

### 6.1. Phase 1: Document Ingestion

#### 6.1.1. Upload Documents

**Flow:**
1. User chọn files (PDF, DOCX, TXT) trong frontend
2. Frontend gửi POST `/collections/{name}/documents` với files
3. Backend đọc text từ mỗi file:
   - PDF: `extract_text_from_pdf()` sử dụng pypdf
   - DOCX: `extract_text_from_docx()` sử dụng python-docx
   - TXT: decode UTF-8 hoặc latin1
4. Lưu text và metadata (filename, file_type)

#### 6.1.2. Semantic Chunking

**Flow:**
1. Với mỗi document:
   a. Split thành sentences (language-specific):
      - English: Regex pattern `(?<=[.!?])\s+(?=[A-Z])`
      - Vietnamese: Regex pattern với Vietnamese uppercase characters
   b. Nếu `use_semantic_chunking = True`:
      - Load chunking model (lazy loading)
      - Compute embeddings cho mỗi sentence:
        * Batch size: 32
        * Normalize embeddings: True
      - Tính cosine similarity giữa consecutive sentences
      - Group sentences:
        * Nếu similarity >= threshold: Thêm vào current chunk
        * Nếu similarity < threshold AND chunk >= min_size: Split chunk
        * Nếu chunk >= max_size: Force split
      - Merge small chunks với neighbors
   c. Nếu `use_semantic_chunking = False`:
      - Chỉ split thành sentences (không group)
2. Tạo metadata cho mỗi chunk:
   - `source`: Filename
   - `file_type`: File type
   - `chunk_index`: Index trong document
   - `num_chunks`: Tổng số chunks trong document

#### 6.1.3. Index Building

**Flow:**

1. **BM25 Index:**
   - Normalize text theo language
   - Tính term frequency (TF) cho mỗi document
   - Tính inverse document frequency (IDF)
   - Lưu: doc_freqs, idf, doc_len, avg_doc_len

2. **Dense Index:**
   - Normalize text theo language
   - Compute embeddings cho chunks:
     * Sử dụng DenseRetriever model
     * Batch size: 16
     * Normalize embeddings
   - Lưu embeddings array

#### 6.1.4. Configuration

- **Similarity threshold:**
  - English: 0.4 (`config.chunking.similarity_threshold_en`)
  - Vietnamese: 0.5 (`config.chunking.similarity_threshold_vi`)
- **Chunk sizes:**
  - Min: 64 characters
  - Max: 1024 characters
- **Embedding batch size**: 32

### 6.2. Phase 2: Query Processing

#### 6.2.1. Query Rewriting

**Flow:**
1. Nhận original query từ user
2. Xác định language (từ request hoặc collection)
3. Gọi `rewrite_query()`:
   a. Tạo prompt theo language:
      - Vietnamese: "Bạn là một chuyên gia tìm kiếm thông tin..."
      - English: "You are an expert search information specialist..."
   b. LLM call (GPT-4o):
      - Model: `config.llm.query_rewrite_model`
      - Temperature: 0.3 (focused)
      - Max tokens: 200
      - System prompt: Instructions
      - User prompt: "Rewrite query: {original_query}"
   c. Trả về rewritten query
4. Lưu original và rewritten query

**Mục đích:**
- Mở rộng từ khóa quan trọng với từ đồng nghĩa
- Làm rõ ý định nếu truy vấn mơ hồ
- Cải thiện hiệu suất retrieval

#### 6.2.2. Adaptive K Selection

**Flow:**
1. Nhận query (original, không phải rewritten)
2. Xác định n (từ request hoặc config default: 3)
3. Chạy n iterations:
   a. Chọn style từ style_candidates (cycle nếu n > len(styles)):
      - 10 styles khác nhau (học thuật, ngắn gọn, chi tiết, etc.)
   b. Build style prompt:
      - Vietnamese: "Trả lời tập trung vào khái niệm cốt lõi:\n\n{query}"
      - English: "Answer briefly and concisely:\n\n{query}"
   c. LLM call với logprobs (non-hop):
      - Model: `config.adaptive.phase1_model` (gpt-4o)
      - Temperature: 0.01 (very deterministic)
      - Max tokens: 64
      - Logprobs: True, top_logprobs: 1
      - System prompt: Language-specific instructions
      - User prompt: style_prompt
   d. Tính entropy từ logprobs:
      - Formula: `H = average(-logprob / ln(2))`
      - Unit: bits/token
   e. Lưu iteration detail:
      - `run`: Iteration number
      - `style`: Style instruction (in correct language)
      - `output`: Generated text
      - `entropy`: Calculated entropy

4. Tính average entropy: `sum(entropies) / n`
5. Chuyển đổi entropy thành k:
   - Formula: `k = k_min + (entropy/entropy_max) * (k_max - k_min)`
   - Clamp: `max(k_min, min(k_max, k))`
   - Default: k_min=0, k_max=20, entropy_max=10
6. Trả về (k, metadata):
   - `k`: Số documents cần retrieve
   - `metadata`: average_entropy, k_determined, iterations_detail, n, k_min, k_max

**Lý do:**
- Entropy cao → Model không chắc chắn → Cần nhiều documents
- Entropy thấp → Model chắc chắn → Cần ít documents
- Adaptive approach: Tự động điều chỉnh k dựa trên độ phức tạp của query

#### 6.2.3. Hybrid Retrieval

**Flow:**

1. Sử dụng rewritten query cho retrieval

2. **BM25 Retrieval:**
   a. Normalize query theo language
   b. Tách query thành terms
   c. Với mỗi term:
      - Lấy IDF từ index
      - Với mỗi document:
        * Tính TF
        * Tính BM25 score:
          ```
          score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * doc_len/avg_doc_len))
          ```
        * Cộng vào document score
   d. Sort theo score giảm dần
   e. Trả về top-k: `List[(doc_index, score)]`

3. **Dense Retrieval:**
   a. Normalize query theo language
   b. Embed query:
      - Sử dụng DenseRetriever model
      - Normalize embedding
   c. Tính cosine similarity:
      - `similarity = dot(embeddings, query_embedding)`
   d. Sort theo similarity giảm dần
   e. Trả về top-k: `List[(doc_index, score)]`

4. **Reciprocal Rank Fusion:**
   a. Với mỗi document trong kết quả:
      - BM25 rank: r1
      - Dense rank: r2
      - RRF score = `1/(k_rrf + r1 + 1) + 1/(k_rrf + r2 + 1)`
      - k_rrf = 60 (smoothing parameter)
   b. Sort theo RRF score giảm dần
   c. Trả về top-k documents sau fusion

#### 6.2.4. Reranking

**Flow:**
1. Lấy candidate documents từ RRF (top-k)
2. Với mỗi candidate:
   a. Build pair: `[query, document_text]`
   b. Cross-encoder score:
      - Model: `cross-encoder/ms-marco-MiniLM-L-12-v2`
      - Input: Query-document pair
      - Output: Relevance score (float)
3. Combine RRF scores và reranker scores:
   a. Với mỗi document: `(doc_index, rrf_score, rerank_score)`
   b. Sort theo rerank_score giảm dần (ưu tiên reranker score)
4. Trả về top-k documents sau reranking

**Lý do:**
- Cross-encoder xử lý query-document pair cùng lúc
- Độ chính xác cao hơn cosine similarity đơn thuần
- Cải thiện precision của retrieval

### 6.3. Phase 3: Answer Generation

#### 6.3.1. Context Preparation

**Flow:**
1. Nhận retrieved documents (sau reranking)
2. Format context:
   - Với mỗi chunk:
     ```
     [Document {i+1} - Source: {source}]
     {text}
     
     ```
   - Join tất cả chunks
3. Nếu không có documents: `context = ""`

#### 6.3.2. LLM Answer Generation

**Flow:**
1. Xác định language (từ request hoặc collection)
2. Tạo prompt theo language:
   a. System prompt:
      - Vietnamese: "Bạn là một trợ lý AI thông minh hoạt động trong hệ thống Adaptive RAG..."
      - English: "You are an intelligent AI assistant working in an Adaptive RAG system..."
      - Instructions:
        * Nếu có context: Sử dụng khi hữu ích, kết hợp với kiến thức
        * Nếu không có context: Trả lời dựa trên kiến thức vốn có
        * Trích dẫn nguồn khi có thể
        * Trả lời tự nhiên và dễ hiểu
   b. User prompt:
      - Nếu có context:
        ```
        Question: {query}
        
        Reference documents:
        {context_text}
        ```
      - Nếu không có context:
        ```
        Question: {query}
        
        No reference documents provided.
        ```
3. LLM call (GPT-4o):
   - Model: `config.llm.answer_generation_model`
   - Temperature: 0.7 (balanced)
   - Max tokens: 1000
   - Messages: [system, user]
4. Extract answer từ response
5. Extract sources từ chunks:
   - `source`: Filename
   - `chunk_index`: Index trong document
   - `score`: Reranker score
6. Trả về: `{answer, sources, model, num_chunks}`

**Adaptive RAG Philosophy:**
- Hệ thống có thể trả lời không cần context nếu câu hỏi đơn giản
- Nếu context không liên quan, LLM có thể bỏ qua và trả lời dựa trên kiến thức
- Linh hoạt hơn traditional RAG (luôn cần context)

### 6.4. Phase 4: Response Assembly

#### 6.4.1. Pipeline Metadata

**Flow:**
1. Thu thập thông tin từ các steps:
   - Query rewriting: time, original, rewritten, model
   - Adaptive k selection: time, k, entropy, iterations detail
   - Retrieval: time, num_results, preview_documents
   - Answer generation: time, model, num_chunks_used
2. Tính total_time: Sum của tất cả step times
3. Assemble pipeline_steps dict

#### 6.4.2. Response Format

```json
{
  "original_query": "string",
  "rewritten_query": "string",
  "k": 0,
  "entropy": 0.0,
  "results": [
    {
      "text": "string",
      "metadata": {},
      "rrf_score": 0.0,
      "rerank_score": 0.0,
      "score": 0.0
    }
  ],
  "answer": "string",
  "sources": [
    {
      "source": "string",
      "chunk_index": 0,
      "score": 0.0
    }
  ],
  "pipeline_steps": {
    "query_rewriting": {},
    "adaptive_k_selection": {},
    "retrieval": {},
    "answer_generation": {},
    "total_time": 0.0
  }
}
```

## 7. Configuration System

### 7.1. Cấu trúc configuration

**Location**: `backend/config/config.py`

**Các config classes:**
- `LLMConfig`: Settings cho OpenAI LLM
- `RetrievalConfig`: Settings cho retrieval models
- `ChunkingConfig`: Settings cho semantic chunking
- `AdaptiveConfig`: Settings cho adaptive k selection
- `RerankerConfig`: Settings cho reranker
- `FusionConfig`: Settings cho result fusion
- `APIConfig`: Default API settings
- `RAGConfig`: Main config class chứa tất cả

### 7.2. Chi tiết configuration

#### 7.2.1. LLMConfig

- `query_rewrite_model`: "gpt-4o"
- `answer_generation_model`: "gpt-4o" (OpenAI) hoặc "gemini-2.5-flash-lite" (Gemini)
- `answer_generation_provider`: "gemini" (mặc định, có thể đổi "openai")
- `query_rewrite_temperature`: 0.3, `query_rewrite_max_tokens`: 200
- `answer_generation_temperature`: 0.7, `answer_generation_max_tokens`: 50000

#### 7.2.2. RetrievalConfig

- `dense_model_name_en`: "sentence-transformers/all-mpnet-base-v2"
- `dense_model_name_vi`: "keepitreal/vietnamese-sbert-v2"
- `chunking_model_name`: None (sử dụng dense model)
- `default_similarity_threshold`: 0.6

#### 7.2.3. ChunkingConfig

- `similarity_threshold_en`: 0.4
- `similarity_threshold_vi`: 0.5
- `min_chunk_size`: 64
- `max_chunk_size`: 1024
- `embedding_batch_size`: 32

#### 7.2.4. AdaptiveConfig

- `k_min`: 0
- `k_max`: 10
- `default_n`: 5 (có thể chỉnh 1-10 từ sidebar)
- `style_candidates`: 10 styles (EN/VI mapping)
- `phase1_model`: "gpt-4o"
- `phase1_temperature`: 0.01
- `phase1_max_tokens`: 64
- `entropy_max`: 0.5 (normalize entropy trước khi map k)

#### 7.2.5. RerankerConfig

- `model_name_en`: "cross-encoder/ms-marco-MiniLM-L-12-v2"
- `model_name_vi`: "cross-encoder/ms-marco-MiniLM-L-12-v2"

#### 7.2.6. FusionConfig

- `k_rrf`: 60

#### 7.2.7. APIConfig

- `default_language`: "en"
- `default_similarity_threshold`: 0.5
- `default_dense_model_name`: None
- `default_chunking_model_name`: None

### 7.3. Cách sử dụng configuration

```python
from backend.config.config import get_config

config = get_config()
model_name = config.llm.query_rewrite_model
k_min = config.adaptive.k_min
# ...
```

- **Reload**: `reload_config()` (tạo instance mới)

## 8. Dependencies và Technical Stack

### 8.1. Core Dependencies

- `fastapi>=0.104.0`: Web framework
- `uvicorn[standard]>=0.24.0`: ASGI server
- `pydantic>=2.0.0`: Data validation
- `python-multipart>=0.0.6`: File upload support

### 8.2. ML and NLP Libraries

- `sentence-transformers>=2.2.0`: Embedding models
- `numpy>=1.24.0`: Numerical computing
- `scikit-learn>=1.3.0`: ML utilities (TF-IDF fallback)

### 8.3. Document Processing

- `pypdf>=3.17.0`: PDF extraction
- `python-docx>=1.1.0`: DOCX extraction

### 8.4. LLM Integration

- `openai>=1.0.0`: OpenAI API client
- `google-generativeai>=0.3.0`: Gemini API client

### 8.5. Frontend

- `streamlit>=1.28.0`: Web UI framework
- `requests>=2.31.0`: HTTP client

### 8.6. Environment Variables

- `OPENAI_API_KEY`: Required cho LLM calls

## 9. Hyperparameter Tuning (Entropy) - Tóm tắt

- **Mục tiêu**: Tìm phương pháp tính entropy ổn định, tương quan với độ khó câu hỏi, và tối ưu n (iterations) cho adaptive k mapping.
- **Dataset**: 101 queries/ngôn ngữ (EN/VI), phân loại easy/medium/hard, bao phủ factual/analytical/reasoning.
- **Phương pháp so sánh**:
  - Full Token n=1: Baseline; EN ổn, VI nhiễu; stability thấp.
  - First N Tokens n=1: Không cải thiện, loại bỏ.
  - Full Token n=5: Stability cao, entropy std ~0, phân tách độ khó rõ cho EN/VI → **chọn cho production**.
- **Kết luận chính**:
  - Entropy tăng tuyến tính với độ khó (easy ~0.19-0.25, medium ~0.25-0.30, hard ~0.30-0.40).
  - Tăng n (5) giảm noise/outliers, curve mượt, đáng tin cậy hơn.
  - Trade-off chi phí API 5x nhưng stability và chất lượng adaptive k cải thiện rõ.
- **Tài liệu chi tiết**: `hyperparam_tuning/` (README, notebooks, CSV kết quả cho EN/VI và các biến thể).

## 10. Kết luận

Adaptive RAG System là một hệ thống RAG hoàn chỉnh với các tính năng:

1. **Language-aware model selection**: Tự động chọn model tối ưu cho tiếng Anh và tiếng Việt

2. **Adaptive retrieval**: Tự động điều chỉnh số lượng documents cần retrieve dựa trên entropy

3. **Hybrid retrieval**: Kết hợp BM25 và Dense retrieval để đạt độ chính xác cao

4. **Semantic chunking**: Chia documents thành chunks có ý nghĩa hơn fixed-size chunking

5. **Reranking**: Sử dụng cross-encoder để đánh giá lại độ liên quan

6. **Flexible answer generation**: Có thể trả lời với hoặc không cần context, phù hợp với Adaptive RAG philosophy

7. **Professional UI**: Streamlit interface với đầy đủ thông tin về pipeline steps

Hệ thống được thiết kế để hỗ trợ cả tiếng Anh và tiếng Việt một cách tối ưu, với model selection thông minh và adaptive approach để cải thiện hiệu suất retrieval và answer quality.

---

**END OF DOCUMENTATION**
