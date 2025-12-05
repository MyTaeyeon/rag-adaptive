"""
Phase 2: Evaluation Pipeline

Đọc kết quả từ training_log.jsonl, gọi LLM judge để đánh giá và lưu vào score.jsonl.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))
import config
from judge import LLMJudge


def load_training_log(log_file_path: str = None) -> List[Dict[str, Any]]:
    """
    Đọc file training_log.jsonl và trả về list các samples.
    
    Args:
        log_file_path: Đường dẫn đến log file
    
    Returns:
        List các samples từ log file
    """
    if log_file_path is None:
        log_file_path = config.LOG_FILE_PATH
    
    log_file = Path(log_file_path)
    
    if not log_file.exists():
        print(f"Log file not found: {log_file_path}")
        return []
    
    samples = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                if "sample_id" in entry and "failed_sample_id" not in entry:
                    samples.append(entry)
            except json.JSONDecodeError:
                continue
    
    return samples


def extract_answers_and_context(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    Trích xuất answers từ 3 methods và context từ sample.
    
    Args:
        sample: Sample data từ training_log.jsonl
    
    Returns:
        Dict với keys: answer_1, answer_2, answer_3, context
    """
    stage1_results = sample.get("stage_1_results", {})
    
    answer_1 = stage1_results.get("method_1_baseline", {}).get("final_output", "")
    answer_2 = stage1_results.get("method_2_adaptive_mini", {}).get("final_output", "")
    answer_3 = stage1_results.get("method_3_adaptive_n5", {}).get("final_output", "")
    
    context = sample.get("input_data", {}).get("context", "")
    
    return {
        "answer_1": answer_1,
        "answer_2": answer_2,
        "answer_3": answer_3,
        "context": context
    }


def evaluate_sample(sample: Dict[str, Any], judge: LLMJudge) -> Dict[str, Any]:
    """
    Đánh giá một sample với LLM judge.
    
    Args:
        sample: Sample data từ training_log.jsonl
        judge: LLMJudge instance
    
    Returns:
        Dict chứa kết quả evaluation
    """
    input_data = sample.get("input_data", {})
    query = input_data.get("query", "")
    ground_truth = input_data.get("ground_truth", "")
    
    answers_context = extract_answers_and_context(sample)
    
    evaluation_result = judge.evaluate(
        query=query,
        ground_truth=ground_truth,
        context=answers_context["context"],
        answer_1=answers_context["answer_1"],
        answer_2=answers_context["answer_2"],
        answer_3=answers_context["answer_3"]
    )
    
    return {
        "sample_id": sample.get("sample_id"),
        "original_id": sample.get("original_id"),
        "dataset_source": sample.get("dataset_source"),
        "evaluation": evaluation_result,
        "timestamp": datetime.now().isoformat()
    }


def save_score(score_data: Dict[str, Any], score_file_path: str = None) -> None:
    """
    Lưu score vào file JSONL.
    
    Args:
        score_data: Data cần lưu
        score_file_path: Đường dẫn đến file score
    """
    if score_file_path is None:
        score_file_path = config.SCORE_FILE_PATH
    
    score_file = Path(score_file_path)
    
    with open(score_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(score_data, ensure_ascii=False) + '\n')


def main():
    """
    Main function để chạy phase 2 evaluation.
    """
    print("=" * 60)
    print("Phase 2: Evaluation Pipeline")
    print("=" * 60)
    
    print(f"\nLoading training log from: {config.LOG_FILE_PATH}")
    samples = load_training_log()
    
    if not samples:
        print("No samples found in training log.")
        return
    
    print(f"Found {len(samples)} samples to evaluate")
    
    print("\nInitializing LLM Judge...")
    judge = LLMJudge()
    print(f"Judge model: {judge.model}")
    
    print(f"\nSaving scores to: {config.SCORE_FILE_PATH}")
    print("Starting evaluation...\n")
    
    for idx, sample in enumerate(samples, 1):
        sample_id = sample.get("sample_id", "N/A")
        print(f"[{idx}/{len(samples)}] Evaluating sample ID: {sample_id}")
        
        try:
            score_data = evaluate_sample(sample, judge)
            save_score(score_data)
            print(f"  Success - Score saved")
            
            scores = score_data["evaluation"]["scores"]
            if "method_1" in scores:
                print(f"  Method 1 score: {scores['method_1'].get('overall_score', 'N/A')}")
            if "method_2" in scores:
                print(f"  Method 2 score: {scores['method_2'].get('overall_score', 'N/A')}")
            if "method_3" in scores:
                print(f"  Method 3 score: {scores['method_3'].get('overall_score', 'N/A')}")
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("Phase 2 evaluation completed!")
    print(f"Results saved to: {config.SCORE_FILE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

