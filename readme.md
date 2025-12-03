1. Tải lib trong lb_install
2. chạy be: uvicorn backend:app --reload
3. Chạy fe:streamlit run frontend.py




# 1. Flow Demo
+ Tạo collection -> Up load (nhiều) tài liệu  -> Chunking -> Embedding 
  - Cần sửa lại flow cho tiếng việt, hiện chỉ dùng được tiếng anh
  - Cần animation đẹp hơn, từ load tài liệu, chờ chunking, chờ embedding, thông báo embedding hoàn tất -> cần animation
+ Nhận query người dùng -> call api rewiting query -> đưa vào adaptive k tìm k -> đưa k + query mới vào hybrid retrieval -> rerank ->  