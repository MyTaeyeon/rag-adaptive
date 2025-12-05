"""
Phase 3: Convert to CSV

Đọc training_log.jsonl và score.jsonl, merge và chuyển đổi thành result.csv
"""

import json
import csv
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))
import config


def load_training_log(log_file_path: str = None) -> Dict[int, Dict]:
    """
    Đọc training_log.jsonl và trả về dict với key là sample_id.
    
    Args:
        log_file_path: Đường dẫn đến log file
    
    Returns:
        Dict với key là sample_id
    """
    if log_file_path is None:
        log_file_path = config.LOG_FILE_PATH
    
    log_file = Path(log_file_path)
    
    if not log_file.exists():
        print(f"Training log file not found: {log_file_path}")
        return {}
    
    samples = {}
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                if "sample_id" in entry and "failed_sample_id" not in entry:
                    samples[entry["sample_id"]] = entry
            except json.JSONDecodeError:
                continue
    
    return samples


def load_scores(score_file_path: str = None) -> Dict[int, Dict]:
    """
    Đọc score.jsonl và trả về dict với key là sample_id.
    
    Args:
        score_file_path: Đường dẫn đến score file
    
    Returns:
        Dict với key là sample_id
    """
    if score_file_path is None:
        score_file_path = config.SCORE_FILE_PATH
    
    score_file = Path(score_file_path)
    
    if not score_file.exists():
        print(f"Score file not found: {score_file_path}")
        return {}
    
    scores = {}
    with open(score_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                if "sample_id" in entry:
                    scores[entry["sample_id"]] = entry
            except json.JSONDecodeError:
                continue
    
    return scores


def extract_metrics(training_sample: Dict, score_sample: Dict) -> Dict[str, Any]:
    """
    Extract metrics từ training sample và score sample.
    
    Args:
        training_sample: Sample từ training_log.jsonl
        score_sample: Sample từ score.jsonl
    
    Returns:
        Dict chứa các metrics
    """
    stage1_results = training_sample.get("stage_1_results", {})
    evaluation = score_sample.get("evaluation", {})
    scores = evaluation.get("scores", {})
    
    method1 = stage1_results.get("method_1_baseline", {})
    method2 = stage1_results.get("method_2_adaptive_mini", {})
    method3 = stage1_results.get("method_3_adaptive_n5", {})
    
    method1_metrics = method1.get("execution_metrics", {})
    method2_metrics = method2.get("execution_metrics", {})
    method3_metrics = method3.get("execution_metrics", {})
    
    method2_phase1 = method2.get("phase_1_analysis", {})
    method3_phase1 = method3.get("phase_1_analysis", {})
    
    method1_score = scores.get("method_1", {}).get("overall_score", 0.0)
    method2_score = scores.get("method_2", {}).get("overall_score", 0.0)
    method3_score = scores.get("method_3", {}).get("overall_score", 0.0)
    
    return {
        "sample_id": training_sample.get("sample_id", 0),
        "m1_latency": method1_metrics.get("latency_ms", 0.0),
        "m2_latency": method2_metrics.get("latency_ms", 0.0),
        "m3_latency": method3_metrics.get("latency_ms", 0.0),
        "m1_total_input_token": method1_metrics.get("total_input_tokens", 0),
        "m2_total_input_token": method2_metrics.get("total_input_tokens", 0),
        "m3_total_input_token": method3_metrics.get("total_input_tokens", 0),
        "m1_total_output_token": method1_metrics.get("total_output_tokens", 0),
        "m2_total_output_token": method2_metrics.get("total_output_tokens", 0),
        "m3_total_output_token": method3_metrics.get("total_output_tokens", 0),
        "m1_k": config.BASELINE_K,
        "m2_k": method2_phase1.get("k_determined", 0),
        "m3_k": method3_phase1.get("k_determined", 0),
        "m1_judge_score": method1_score,
        "m2_judge_score": method2_score,
        "m3_judge_score": method3_score,
    }


def convert_to_csv(output_file: str = None):
    """
    Convert training_log.jsonl và score.jsonl thành CSV file.
    
    Args:
        output_file: Tên file output CSV (mặc định từ config)
    """
    if output_file is None:
        output_file = config.RESULT_CSV_PATH
    print("=" * 60)
    print("Phase 3: Convert to CSV")
    print("=" * 60)
    
    print(f"\nLoading training log from: {config.LOG_FILE_PATH}")
    training_samples = load_training_log()
    
    print(f"Loading scores from: {config.SCORE_FILE_PATH}")
    score_samples = load_scores()
    
    if not training_samples:
        print("No training samples found.")
        return
    
    if not score_samples:
        print("No score samples found.")
        return
    
    print(f"\nFound {len(training_samples)} training samples")
    print(f"Found {len(score_samples)} score samples")
    
    all_sample_ids = sorted(set(training_samples.keys()) & set(score_samples.keys()))
    
    if not all_sample_ids:
        print("No matching samples found between training log and scores.")
        return
    
    print(f"Processing {len(all_sample_ids)} samples...\n")
    
    csv_data = []
    for sample_id in all_sample_ids:
        try:
            metrics = extract_metrics(training_samples[sample_id], score_samples[sample_id])
            csv_data.append(metrics)
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue
    
    if not csv_data:
        print("No data to write.")
        return
    
    csv_file = Path(output_file)
    
    fieldnames = [
        "sample_id",
        "m1_latency", "m2_latency", "m3_latency",
        "m1_total_input_token", "m2_total_input_token", "m3_total_input_token",
        "m1_total_output_token", "m2_total_output_token", "m3_total_output_token",
        "m1_k", "m2_k", "m3_k",
        "m1_judge_score", "m2_judge_score", "m3_judge_score"
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Successfully converted {len(csv_data)} samples to {output_file}")
    print("=" * 60)


def main():
    """
    Main function.
    """
    convert_to_csv()


if __name__ == "__main__":
    main()

