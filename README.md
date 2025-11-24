cấu trúc schema lõi:

    {
      "id": "unique_id",
      "question": "string",
      "context": "string",
      "answers": ["answer_1", "answer_2"],
      "dataset": "viquad | vimed | hotpotqa"
    }


---


### HotpotQA mapping:


| HotpotQA field     | Unified field         | Mapping                                    |
| ------------------ | --------------------- | ------------------------------------------ |
| `id`               | `id`                  | giữ nguyên                                 | 
| `question`         | `question`            | giữ nguyên                                 |
| `answer`           | `answers`             | biến từ string → list: `[answer]`          | 
| `context`          | `context_docs`        | GIỮ nguyên dạng structured (list document) | 
| `context`          | `context` (flattened) | GỘP nhiều document vào 1 string            | 
| `type`             | `type`                | giữ nguyên                                 | 
| `level`            | `level`               | giữ nguyên                                 | 
| `supporting_facts` | `supporting_facts`    | giữ nguyên nhưng enrich thêm câu văn       | 
| *(mới thêm)*       | `dataset`             | `"hotpotqa"`                               | 

---
