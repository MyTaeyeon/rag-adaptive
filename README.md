# How to run this project

   # prepare data
   python scripts/download_dataset.py
   python scripts/prepare_dataset.py --raw_dir raw_data --out samples/dataset_normalized.json --sample_size 100

# Adaptive RAG — Repo Scaffold

This repository is a scaffold for the **Adaptive RAG** project:
- Adaptive controller: dynamic `k` selection (heuristics + LLM entropy)
- Hybrid retrieval: BM25 + ColBERTv2 (dense) + RRF merging
- Reranker (optional): cross-encoder
- Experiment runner + LLM-as-judge evaluation
- Demo: FastAPI backend + simple React frontend

## What is included
- `scripts/` : Python scripts for indexing, retrieval, adaptive controller, experiment runner, and LLM judge (templates).
- `backend/fastapi_app.py` : Minimal FastAPI orchestrator for the demo.
- `frontend/` : Single-file React demo (App.jsx) and instructions.
- `samples/` : Small sample dataset (`dataset.json`) and a tiny corpus for indexing.
- `requirements.txt` and `setup.sh` : environment hints.

## How to use
1. Create a Python environment (recommended: Python 3.10+)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Additional optional components: Pyserini, ColBERTv2, FAISS (GPU/CPU) depending on your setup.

3. Build indexes (see scripts/index_bm25.py and scripts/index_colbert.py for details).
4. Run backend demo:
   ```bash
   uvicorn backend.fastapi_app:app --reload --port 8000
   ```
5. Open `frontend/README_FRONTEND.md` for starting the React demo.

## Notes
- Scripts are provided as runnable templates. You will likely need to adapt paths and install heavy dependencies (ColBERTv2, Pyserini, FAISS) for full functionality.
- See `scripts/README_SCRIPTS.md` for quick notes on each script.

If you want, I can:
- expand sample dataset to 100+ items,
- implement a runnable TF-IDF-based retrieval baseline for quick local testing,
- or prepare Dockerfiles / k8s manifests for deployment.