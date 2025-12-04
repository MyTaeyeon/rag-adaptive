# Scripts overview

- index_bm25.py : build a BM25 index (Pyserini/Lucene) from a TSV/JSON corpus.
- index_colbert.py : prepare ColBERTv2 embeddings & index (template).
- adaptive_controller.py : heuristics + LLM-entropy based controller to choose k.
- retrieve_hybrid.py : run BM25 and ColBERT retrieval, merge with RRF.
- rerank_crossencoder.py : optional cross-encoder reranker (template).
- experiment_runner.py : orchestrate running a dataset through the pipeline and log metrics.
- llm_judge.py : call LLM to score system answer vs ground truth (template).

Many scripts are templates and include TODOs for adding your API keys and paths.

