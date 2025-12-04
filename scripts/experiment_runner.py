"""
experiment_runner.py
Orchestrates evaluation of a dataset (samples/dataset.json) using the pipeline templates.

Outputs CSV with columns:
  id, category, controller_mode, k, latency_seconds, token_count_est, system_answer, ground_truth, llm_score

Note: This is a template that simulates LLM answers. Replace `simulate_answer` with real LLM call in production.
"""
import time
import json
import csv
from pathlib import Path
from scripts.adaptive_controller import heuristic_k
from scripts.retrieve_hybrid import hybrid_retrieve

SAMPLES = "samples/dataset.json"
CORPUS = "samples/corpus.tsv"
OUT_CSV = "experiments/results_simulated.csv"

def simulate_llm_answer(query, retrieved_docs):
    # Simple heuristic answer generator (for debugging)
    return f"Simulated answer for: {query} (docs: {','.join(retrieved_docs[:3])})"

def estimate_token_count(text):
    # rough estimate: 1 token ~= 0.75 words
    return int(len(text.split()) / 0.75)

def run():
    Path("experiments").mkdir(exist_ok=True)
    with open(SAMPLES, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        qid = item["id"]
        q = item["question"]
        gt = item.get("ground_truth","")
        t0 = time.time()
        ctrl = heuristic_k(q)
        k = ctrl["k"]
        ret = hybrid_retrieve(q, CORPUS, topk=k)
        answer = simulate_llm_answer(q, ret["merged"])
        latency = time.time() - t0
        tokens = estimate_token_count(q) + estimate_token_count(answer)
        # simulated scoring: 1 if ground-truth tokens appear in answer
        score = 1.0 if any(w.lower() in answer.lower() for w in gt.split()[:3]) else 0.0
        rows.append([qid, item.get("category",""), ctrl["mode"], k, round(latency,3), tokens, answer, gt, score])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","category","controller_mode","k","latency_seconds","token_count_est","system_answer","ground_truth","llm_score"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    run()
