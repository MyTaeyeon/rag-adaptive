# Debugging Guide

## 🔍 Kiểm tra Frontend đang chạy

### 1. Kiểm tra Server đang chạy

```bash
# Frontend server (port 8000)
cd frontend
python -m http.server 8000

# Backend server (port 2022)
python run_backend.py
```

### 2. Kiểm tra Browser Console

Mở Developer Tools (F12) và kiểm tra:
- **Console tab**: Xem có lỗi JavaScript không
- **Network tab**: Xem các API calls có thành công không

### 3. Các lỗi thường gặp

#### ❌ "Cannot connect to backend"

**Nguyên nhân**: Backend chưa chạy hoặc chạy sai port

**Giải pháp**:
1. Kiểm tra backend đang chạy: `http://localhost:2022/docs`
2. Kiểm tra `API_BASE_URL` trong `frontend/js/api.js`
3. Đảm bảo không có firewall chặn

#### ❌ CORS Error

**Nguyên nhân**: Backend chưa enable CORS

**Giải pháp**: Backend đã có CORS middleware, nhưng nếu vẫn lỗi:
```python
# backend/api/main.py đã có:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    ...
)
```

#### ❌ "Failed to load collections"

**Nguyên nhân**: 
- Backend chưa chạy
- API endpoint không đúng
- Network issue

**Giải pháp**:
1. Kiểm tra backend: `curl http://localhost:2022/collections`
2. Kiểm tra browser console để xem error chi tiết
3. Kiểm tra Network tab trong DevTools

#### ❌ File upload không hoạt động

**Nguyên nhân**:
- File quá lớn
- File type không được support
- Backend error

**Giải pháp**:
1. Kiểm tra file types: `.pdf`, `.docx`, `.txt`
2. Kiểm tra console để xem error
3. Kiểm tra backend logs

#### ❌ Chat không gửi được message

**Nguyên nhân**:
- Collection chưa được chọn
- Backend error
- Session issue

**Giải pháp**:
1. Đảm bảo đã chọn collection
2. Kiểm tra console để xem error
3. Kiểm tra Network tab để xem API response

## 🧪 Test các tính năng

### Test 1: Load Collections
1. Mở browser console (F12)
2. Refresh page
3. Kiểm tra có log "Error loading collections" không
4. Kiểm tra dropdown có collections không

### Test 2: Create Collection
1. Click "Create Collection"
2. Nhập tên collection
3. Click "Create"
4. Kiểm tra console có error không
5. Kiểm tra collection có xuất hiện trong dropdown không

### Test 3: Upload Files
1. Chọn collection
2. Chọn files (PDF/DOCX/TXT)
3. Click "Upload"
4. Kiểm tra progress bar
5. Kiểm tra console có error không
6. Kiểm tra collection info có update không

### Test 4: Send Message
1. Chọn collection
2. Nhập câu hỏi
3. Click "Send" hoặc Enter
4. Kiểm tra message có hiển thị không
5. Kiểm tra pipeline steps có hiển thị không
6. Kiểm tra Network tab để xem API call

## 🔧 Debug Commands

### Test API từ Browser Console

```javascript
// Test load collections
CollectionsAPI.list().then(console.log).catch(console.error);

// Test create collection
CollectionsAPI.create('test-collection', 'en').then(console.log).catch(console.error);

// Test query
QueryAPI.query('test-collection', 'What is RAG?').then(console.log).catch(console.error);
```

### Test Backend từ Terminal

```bash
# Test collections endpoint
curl http://localhost:2022/collections

# Test models endpoint
curl http://localhost:2022/models

# Test create collection
curl -X POST http://localhost:2022/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "language": "en"}'
```

## 📊 Kiểm tra State

### Xem AppState trong Console

```javascript
// Trong browser console
console.log(AppState);

// Xem messages
console.log(AppState.messages);

// Xem pipeline steps
console.log(AppState.pipelineSteps);
```

### Xem localStorage

```javascript
// Trong browser console
console.log(localStorage.getItem('ragAppState'));
```

## 🐛 Common Issues

### Issue: Pipeline steps không hiển thị

**Kiểm tra**:
1. `AppState.pipelineSteps` có data không
2. `renderPipelineSteps()` có được gọi không
3. Console có error không

**Fix**: Kiểm tra response từ API có `pipeline_steps` không

### Issue: Messages không lưu

**Kiểm tra**:
1. `AppState.messages` có được update không
2. DOM có được update không

**Fix**: Kiểm tra `addMessage()` function

### Issue: Session không persist

**Kiểm tra**:
1. `AppState.sessionId` có giá trị không
2. Backend có trả về `session_id` không

**Fix**: Kiểm tra API response có `session_id` field

## 📝 Logging

Để enable detailed logging, thêm vào `app.js`:

```javascript
// Enable debug mode
const DEBUG = true;

function debugLog(...args) {
    if (DEBUG) {
        console.log('[DEBUG]', ...args);
    }
}
```

Sau đó thêm `debugLog()` vào các function quan trọng.

## 🔗 Useful Links

- Backend API Docs: `http://localhost:2022/docs`
- Backend OpenAPI: `http://localhost:2022/openapi.json`

