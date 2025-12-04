"""
index_bm25.py

Template to build a BM25 index using Pyserini (Lucene) or ElasticSearch.
For small experiments, you can skip building a Lucene index and use the `samples/corpus.tsv` directly with TF-IDF retriever.
"""
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="samples/corpus.tsv")
    parser.add_argument("--index_dir", default="indexes/pyserini_wiki")
    args = parser.parse_args()
    print("This is a placeholder script. For production, use Pyserini to build a Lucene index:")
    print("pip install pyserini")
    print("python -m pyserini.index.lucene --collection JsonCollection ...")
if __name__ == "__main__":
    main()
