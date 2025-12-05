"""
Logging Module

Handle logging kết quả training vào JSONL file với resume capability.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))
import config


class TrainingLogger:
    """
    Logger cho training pipeline với resume capability.
    """
    
    def __init__(self, log_file_path: str = None):
        if log_file_path is None:
            log_file_path = config.LOG_FILE_PATH
        self.log_file_path = Path(log_file_path)
    
    def log_sample(self, sample_data: Dict[str, Any]) -> None:
        """
        Log một sample thành công vào file (append mode).
        
        Args:
            sample_data: Dict chứa toàn bộ data của sample
        """
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
    
    def log_failed_sample(self, sample_id: int) -> None:
        """
        Log sample bị lỗi vào cuối file và dừng.
        
        Args:
            sample_id: ID của sample bị lỗi (đã chuẩn hóa)
        """
        failed_entry = {"failed_sample_id": sample_id}
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failed_entry, ensure_ascii=False) + '\n')
    
    def get_resume_point(self) -> Optional[int]:
        """
        Đọc log file để tìm điểm resume.
        
        Returns:
            Sample ID để resume (nếu có), None nếu không có lỗi
        """
        if not self.log_file_path.exists():
            return None
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return None
        
        last_line = lines[-1].strip()
        if not last_line:
            return None
        
        try:
            last_entry = json.loads(last_line)
            if "failed_sample_id" in last_entry:
                return last_entry["failed_sample_id"]
            
            last_sample = json.loads(lines[-1])
            if "sample_id" in last_sample:
                return last_sample["sample_id"] + 1
            
        except json.JSONDecodeError:
            pass
        
        return None
    
    def get_completed_samples(self) -> set:
        """
        Lấy set các sample IDs đã hoàn thành.
        
        Returns:
            Set các sample IDs đã hoàn thành
        """
        completed = set()
        
        if not self.log_file_path.exists():
            return completed
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    if "sample_id" in entry:
                        completed.add(entry["sample_id"])
                    elif "failed_sample_id" in entry:
                        break
                except json.JSONDecodeError:
                    continue
        
        return completed

