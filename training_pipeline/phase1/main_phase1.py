"""
Main Training Pipeline

Chạy training pipeline với 3 phương pháp RAG, có resume capability.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(str(Path(__file__).parent.parent))
import config
from logger import TrainingLogger
from retrieval import InMemoryRetrieval
from generation import LLMGenerator
from methods import BaselineRAG, AdaptiveMiniRAG, AdaptiveN5RAG


def normalize_sample_ids(samples: List[Dict]) -> List[Dict]:
    """
    Chuẩn hóa ID của samples từ 0 đến len(samples)-1.
    
    Args:
        samples: List các samples từ dataset
    
    Returns:
        List samples với ID đã chuẩn hóa
    """
    normalized = []
    for idx, sample in enumerate(samples):
        normalized_sample = sample.copy()
        normalized_sample["normalized_id"] = idx
        normalized.append(normalized_sample)
    return normalized


def process_sample(
    sample: Dict,
    retrieval: InMemoryRetrieval,
    baseline: BaselineRAG,
    adaptive_mini: AdaptiveMiniRAG,
    adaptive_n5: AdaptiveN5RAG
) -> Dict[str, Any]:
    """
    Xử lý một sample với cả 3 methods.
    
    Args:
        sample: Sample data
        retrieval: Retrieval system instance
        baseline: Baseline RAG method
        adaptive_mini: Adaptive Mini RAG method
        adaptive_n5: Adaptive N5 RAG method
    
    Returns:
        Dict chứa kết quả của cả 3 methods
    """
    query = sample["question"]
    context = sample["context"]
    
    retrieval.index_context(context)
    
    try:
        result = {
            "sample_id": sample["normalized_id"],
            "original_id": sample.get("id", ""),
            "dataset_source": sample.get("source", "unknown"),
            "input_data": {
                "query": query,
                "ground_truth": sample.get("ground_truth", ""),
                "context": context
            },
            "stage_1_results": {}
        }
        
        result["stage_1_results"]["method_1_baseline"] = baseline.run(query)
        result["stage_1_results"]["method_2_adaptive_mini"] = adaptive_mini.run(query)
        result["stage_1_results"]["method_3_adaptive_n5"] = adaptive_n5.run(query)
        
        result["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "processing_status": "success"
        }
        
        return result
        
    except Exception as e:
        return {
            "sample_id": sample["normalized_id"],
            "error": str(e),
            "status": "failed"
        }
    finally:
        retrieval.clear()


def main():
    """
    Main function để chạy training pipeline.
    """
    print("=" * 60)
    print("RAG Training Pipeline - Stage 1: Generation & Retrieval")
    print("=" * 60)
    
    dataset_path = Path(config.DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    print(f"\nLoading dataset from: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Total samples: {len(samples)}")
    
    normalized_samples = normalize_sample_ids(samples)
    print("Sample IDs normalized: 0 to", len(normalized_samples) - 1)
    
    logger = TrainingLogger()
    resume_point = logger.get_resume_point()
    
    if resume_point is not None:
        print(f"\nResuming from sample ID: {resume_point}")
        start_idx = resume_point
    else:
        print("\nStarting from the beginning")
        start_idx = 0
    
    print("\nInitializing systems...")
    retrieval = InMemoryRetrieval()
    generator = LLMGenerator()
    baseline = BaselineRAG(retrieval, generator)
    adaptive_mini = AdaptiveMiniRAG(retrieval, generator)
    adaptive_n5 = AdaptiveN5RAG(retrieval, generator)
    
    print("Systems initialized. Starting processing...\n")
    
    for idx in range(start_idx, len(normalized_samples)):
        sample = normalized_samples[idx]
        print(f"[{idx + 1}/{len(normalized_samples)}] Processing sample ID: {sample['normalized_id']} (original: {sample.get('id', 'N/A')})")
        
        try:
            result = process_sample(
                sample,
                retrieval,
                baseline,
                adaptive_mini,
                adaptive_n5
            )
            
            if result.get("status") == "failed":
                print(f"  ERROR: {result.get('error', 'Unknown error')}")
                logger.log_failed_sample(sample["normalized_id"])
                print(f"\nTraining stopped at sample ID: {sample['normalized_id']}")
                print("To resume, run the pipeline again.")
                break
            
            logger.log_sample(result)
            print(f"  Success - Logged to {config.LOG_FILE_PATH}")
            
        except Exception as e:
            print(f"  FATAL ERROR: {str(e)}")
            logger.log_failed_sample(sample["normalized_id"])
            print(f"\nTraining stopped at sample ID: {sample['normalized_id']}")
            print("To resume, run the pipeline again.")
            break
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

