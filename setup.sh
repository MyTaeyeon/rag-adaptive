#!/bin/bash
# Setup script (Linux / WSL). Use python -m venv env and activate before running.
python -m venv rag-env
source rag-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Optional (heavy) components:
# pip install pyserini
# pip install faiss-cpu   # or faiss-gpu
# For ColBERTv2, follow the official repo instructions.
