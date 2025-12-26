# Frontend Web - Adaptive RAG System

Frontend web thuần (HTML/CSS/JavaScript) thay thế cho Streamlit interface.

## 📁 Cấu trúc Files

```
frontend/
├── index.html          # File HTML chính
├── css/
│   └── style.css      # Styling cho toàn bộ ứng dụng
├── js/
│   ├── api.js         # API client để gọi backend FastAPI
│   └── app.js         # Logic chính của ứng dụng (state management, UI)
├── assets/
│   ├── page_icon.jpg  # Icon trang
│   ├── user_icon.jpg  # Avatar người dùng
│   └── bot_icon.jpg   # Avatar bot
└── README.md          # File này
```

## 🚀 Hướng dẫn Chạy

### Yêu cầu

1. **Backend FastAPI đã chạy** trên `http://localhost:2022`
   ```bash
   # Từ thư mục gốc dự án
   python run_backend.py
   ```

2. **Web server** để serve static files (do CORS policy)

### Cách 1: Dùng Python HTTP Server

```bash
# Từ thư mục frontend/
cd frontend
python -m http.server 8000
```

Sau đó mở trình duyệt: `http://localhost:8000`

### Cách 2: Dùng Node.js http-server

```bash
# Cài đặt (nếu chưa có) 
npm install -g http-server

# Chạy từ thư mục frontend/
cd frontend
http-server -p 8000
```

Sau đó mở trình duyệt: `http://localhost:8000`

### Cách 3: Dùng VS Code Live Server

1. Cài extension "Live Server" trong VS Code
2. Click chuột phải vào `index.html` → "Open with Live Server"

## 🔄 Mapping Streamlit → Web

### 1. **State Management**

| Streamlit | Web |
|-----------|-----|
| `st.session_state` | `AppState` object trong `app.js` + `localStorage` |
| Session persistence | `localStorage` để lưu selected collection, model, n |
| Message history | `AppState.messages` array |

### 2. **UI Components**

| Streamlit Component | Web Equivalent |
|---------------------|----------------|
| `st.sidebar` | `<aside class="sidebar">` (left sidebar) |
| `st.title()`, `st.header()` | `<h1>`, `<h2>`, `<h3>` tags |
| `st.text_input()` | `<input type="text">` |
| `st.selectbox()` | `<select>` dropdown |
| `st.button()` | `<button>` với onclick handlers |
| `st.file_uploader()` | `<input type="file" multiple>` |
| `st.chat_message()` | `<div class="message">` với avatar |
| `st.chat_input()` | `<input class="chat-input">` |
| `st.expander()` | `<div class="pipeline-step">` với toggle |
| `st.metric()` | `<div class="metric">` |
| `st.progress()` | `<div class="progress-bar">` |

### 3. **API Calls**

| Streamlit Function | Web API Call |
|-------------------|--------------|
| `requests.get("/collections")` | `CollectionsAPI.list()` |
| `requests.post("/collections", json=...)` | `CollectionsAPI.create(name, language)` |
| `requests.delete("/collections/{name}")` | `CollectionsAPI.delete(name)` |
| `requests.get("/collections/{name}/info")` | `CollectionsAPI.getInfo(name)` |
| `requests.post("/collections/{name}/documents", files=...)` | `CollectionsAPI.uploadDocuments(name, files)` |
| `requests.post("/collections/{name}/query", json=...)` | `QueryAPI.query(collectionName, query, options)` |
| `requests.get("/models")` | `ModelsAPI.list()` |

### 4. **Layout Structure**

**Streamlit:**
```python
with st.sidebar:
    # Collections management
    
col_main, col_right = st.columns([2.5, 1])
with col_main:
    # Chat interface
with col_right:
    # Pipeline steps
```

**Web:**
```html
<div class="app-container">
    <aside class="sidebar">...</aside>
    <main class="main-content">...</main>
    <aside class="pipeline-sidebar">...</aside>
</div>
```

### 5. **Event Handling**

| Streamlit | Web |
|-----------|-----|
| `st.button()` → auto-trigger | `onclick="functionName()"` hoặc `addEventListener()` |
| `st.selectbox()` → auto-update | `onchange="onCollectionChange()"` |
| `st.chat_input()` → auto-submit | `onkeypress="handleChatInputKeyPress(event)"` |
| `st.rerun()` | Manual DOM updates với `innerHTML` hoặc `appendChild()` |

### 6. **Pipeline Steps Display**

**Streamlit:**
```python
def render_pipeline_step(step_name, step_data, step_number):
    with st.expander(f"Step {step_number}: {step_name}"):
        # Display step details
```

**Web:**
```javascript
function renderPipelineSteps(steps) {
    // Tạo HTML với expandable sections
    // Sử dụng togglePipelineStep() để expand/collapse
}
```

## 🎨 Tính năng Chính

### ✅ Đã Implement

1. **Collection Management**
   - Tạo collection mới
   - Chọn collection
   - Xóa collection
   - Upload documents (PDF, DOCX, TXT)
   - Hiển thị thông tin collection (language, chunks, documents)

2. **Chat Interface**
   - Nhập câu hỏi
   - Hiển thị lịch sử chat (user + assistant messages)
   - Avatar cho user và bot
   - Timestamp cho mỗi message
   - Markdown formatting cơ bản

3. **Pipeline Steps Display**
   - Query Rewriting
   - Adaptive K Selection
   - Retrieval (Dense, Sparse, Hybrid, RRF, Cross-Encoder)
   - Answer Generation
   - Expandable/collapsible sections
   - Hiển thị scores, metadata, candidates

4. **Model Selection**
   - Dropdown để chọn model
   - Lưu selection vào localStorage

5. **Adaptive Iterations**
   - Selector cho n (1-10)
   - Lưu vào localStorage

### 🔄 State Management

- **AppState object**: Quản lý state trong memory
- **localStorage**: Lưu persistent state (selected collection, model, n)
- **Session ID**: Tự động tạo và quản lý bởi backend

## 🔧 Cấu hình

### Thay đổi Backend URL

Nếu backend chạy trên port khác, sửa trong `js/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:2022'; // Thay đổi port nếu cần
```

## 📝 Notes

1. **CORS**: Backend đã enable CORS với `allow_origins=["*"]`, nên frontend có thể gọi API từ bất kỳ origin nào.

2. **Session Management**: 
   - Frontend tự động tạo session khi chọn collection
   - Session ID được lưu trong `AppState.sessionId`
   - Backend quản lý conversation history và user memory theo session

3. **Error Handling**: 
   - Hiện tại dùng `alert()` để hiển thị errors
   - Có thể nâng cấp thành toast notifications

4. **Markdown Rendering**: 
   - Hiện tại chỉ support basic markdown (bold, italic, code, line breaks)
   - Có thể tích hợp thư viện như `marked.js` hoặc `markdown-it` để full support

## 🚀 Mở rộng trong tương lai

- [ ] Toast notifications thay cho `alert()`
- [ ] Full markdown rendering với syntax highlighting
- [ ] Dark mode
- [ ] Export chat history
- [ ] Real-time streaming response (Server-Sent Events hoặc WebSocket)
- [ ] Drag & drop file upload
- [ ] Chunks preview trong sidebar (như Streamlit)
- [ ] Responsive design cho mobile

## 🐛 Troubleshooting

### CORS Error

Nếu gặp CORS error, đảm bảo backend đã enable CORS:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ...
)
```

### Backend không kết nối được

1. Kiểm tra backend đã chạy: `http://localhost:2022/docs`
2. Kiểm tra `API_BASE_URL` trong `js/api.js`
3. Kiểm tra console browser (F12) để xem error chi tiết

### File upload không hoạt động

1. Kiểm tra backend endpoint `/collections/{name}/documents` hoạt động
2. Kiểm tra file types được accept: `.pdf`, `.docx`, `.txt`
3. Kiểm tra console browser để xem error

## 📚 Tài liệu tham khảo

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MDN Web Docs](https://developer.mozilla.org/)
- [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
