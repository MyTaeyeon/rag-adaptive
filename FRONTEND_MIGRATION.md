# Migration Guide: Streamlit → Web Frontend

## 📋 Tổng quan

Dự án đã được chuyển đổi từ **Streamlit frontend** sang **Web frontend thuần** (HTML/CSS/JavaScript), tách biệt hoàn toàn với backend FastAPI.

## 🏗️ Kiến trúc mới

```
┌─────────────────────────────────────────┐
│         Frontend (Web)                   │
│  - HTML/CSS/JavaScript                   │
│  - Static files                         │
│  - Gọi API qua fetch()                  │
└──────────────┬──────────────────────────┘
               │ HTTP/REST API
               │
┌──────────────▼──────────────────────────┐
│         Backend (FastAPI)                │
│  - FastAPI server                       │
│  - RAG pipeline logic                   │
│  - LLM integration                      │
│  - Không thay đổi                       │
└─────────────────────────────────────────┘
```

## 🚀 Cách chạy

### 1. Chạy Backend

```bash
# Từ thư mục gốc
python run_backend.py
```

Backend sẽ chạy tại: `http://localhost:2022`

### 2. Chạy Frontend

```bash
# Từ thư mục frontend/
cd frontend
python -m http.server 8000
```

Frontend sẽ chạy tại: `http://localhost:8000`

Mở trình duyệt và truy cập: `http://localhost:8000`

## 📁 Cấu trúc Files

```
rag-adaptive-feature-quang-first_push/
├── backend/                    # Backend FastAPI (KHÔNG ĐỔI)
│   └── api/
│       └── main.py            # API endpoints
├── frontend/                   # Frontend Web mới
│   ├── index.html             # HTML chính
│   ├── css/
│   │   └── style.css          # Styling
│   ├── js/
│   │   ├── api.js             # API client
│   │   └── app.js             # App logic
│   ├── assets/                # Images
│   └── README.md              # Chi tiết frontend
├── run_backend.py             # Script chạy backend
└── FRONTEND_MIGRATION.md      # File này
```

## 🔄 So sánh Streamlit vs Web

### State Management

| Streamlit | Web |
|-----------|-----|
| `st.session_state` | `AppState` object + `localStorage` |
| Auto-persist | Manual save/load từ localStorage |

### UI Components

| Streamlit | Web |
|-----------|-----|
| `st.sidebar` | `<aside class="sidebar">` |
| `st.chat_message()` | `<div class="message">` |
| `st.expander()` | `<div class="pipeline-step">` với toggle |
| `st.button()` | `<button onclick="...">` |

### API Calls

| Streamlit | Web |
|-----------|-----|
| `requests.get/post()` | `CollectionsAPI.list()`, `QueryAPI.query()`, etc. |
| Synchronous | Asynchronous với `async/await` |

## ✅ Tính năng đã implement

- ✅ Collection management (create, select, delete)
- ✅ File upload (PDF, DOCX, TXT)
- ✅ Chat interface với message history
- ✅ Pipeline steps display (expandable)
- ✅ Model selection
- ✅ Adaptive iterations (n) selector
- ✅ Session management (tự động)

## 📝 Notes

1. **Backend không thay đổi**: Tất cả logic RAG/LLM giữ nguyên
2. **API endpoints giữ nguyên**: Frontend gọi cùng các endpoints như Streamlit
3. **Session management**: Backend tự động quản lý conversation history
4. **CORS**: Backend đã enable CORS để frontend có thể gọi API

## 🔧 Troubleshooting

### CORS Error
- Đảm bảo backend đã enable CORS (đã có sẵn trong code)

### Backend không kết nối
- Kiểm tra backend chạy tại `http://localhost:2022`
- Kiểm tra `API_BASE_URL` trong `frontend/js/api.js`

### File upload lỗi
- Kiểm tra file types: `.pdf`, `.docx`, `.txt`
- Kiểm tra console browser (F12) để xem error

## 📚 Tài liệu chi tiết

Xem `frontend/README.md` để biết chi tiết về:
- Cấu trúc code
- Mapping chi tiết Streamlit → Web
- Hướng dẫn mở rộng
- Troubleshooting

## 🎯 Next Steps

1. Test toàn bộ tính năng
2. Customize styling nếu cần
3. Thêm tính năng mới (xem `frontend/README.md` phần "Mở rộng")

