cấu trúc schema lõi:

    {
      "id": "unique_id",
      "question": "string",
      "context": "string",
      "answers": ["answer_1", "answer_2"],
      "dataset": "viquad | vimed | hotpotqa"
    }


---

Data preparation's script:

    cd datasets
    python ./scripts/normalize_and_write.py data/viquad/train.jsonl --config config/normalization.yaml

merged

	python scripts/normalize_and_split.py --input vimed/raw/train.jsonl --out_dir vimed/process/ --config config/normalization.yaml

	python scripts/norm_merge.py --inputs vimed/raw/train.jsonl --out_dir vimed/process/ --config config/normalization.yaml