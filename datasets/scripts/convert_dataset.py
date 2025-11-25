# convert_dataset.py
import argparse
from pathlib import Path
from datasets import load_dataset
from utils.convert import (
    convert_hotpot_dataset,
    convert_vimed_dataset,
    convert_viquad_dataset
)
import os
import traceback

DATASET_REGISTRY = {
    "hotpotqa": {
        "hf_name": "hotpotqa/hotpot_qa",
        "hf_config": "distractor",
        "splits": ["train", "validation"],
        "convert_fn": convert_hotpot_dataset
    },
    "vimed": {
        "hf_name": "tmnam20/ViMedAQA",
        "hf_config": "all",
        "splits": ["train", "test", "validation"],
        "convert_fn": convert_vimed_dataset
    },
    "viquad": {
        "hf_name": "taidng/UIT-ViQuAD2.0",
        "hf_config": None,
        "splits": ["train", "test", "validation"],
        "convert_fn": convert_viquad_dataset
    }
}

def try_load(hf_name, hf_config, cache_dir=None, force=False, use_auth_token=False):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if force:
        kwargs["download_mode"] = "force_redownload"
    if use_auth_token:
        kwargs["use_auth_token"] = True

    print(f"  Loading dataset: {hf_name} config={hf_config} cache_dir={cache_dir} force={force} auth={use_auth_token}")
    try:
        if hf_config is None:
            ds = load_dataset(hf_name, **kwargs)
        else:
            ds = load_dataset(hf_name, hf_config, **kwargs)
        print(f"  -> Loaded dataset keys: {list(ds.keys())}")
        return ds
    except Exception as e:
        print("  !!! Error while loading dataset:")
        traceback.print_exc()
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Thư mục chứa các dataset con (vd: datasets/)")
    ap.add_argument("--cache_dir", default=None, help="Optional HF cache_dir (ví dụ: /data/hf_cache)")
    ap.add_argument("--force", action="store_true", help="Force redownload datasets (download_mode=force_redownload)")
    ap.add_argument("--use_auth_token", action="store_true", help="Set use_auth_token=True for private datasets (requires huggingface-cli login)")
    args = ap.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input dir not found: {root}")

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue

        name = folder.name
        if name not in DATASET_REGISTRY:
            print(f"⚠️  Bỏ qua thư mục '{name}' (không có trong DATASET_REGISTRY)")
            continue

        cfg = DATASET_REGISTRY[name]
        print(f"\n=== Processing dataset: {name} ===")

        # load dataset từ HuggingFace
        ds = try_load(cfg["hf_name"], cfg["hf_config"], cache_dir=args.cache_dir, force=args.force, use_auth_token=args.use_auth_token)
        if ds is None:
            print(f"  -> Skip {name} because load_dataset failed.")
            continue

        # tạo thư mục output
        raw_dir = folder / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # convert từng split
        for split in cfg["splits"]:
            if split not in ds:
                print(f"⚠️  Split '{split}' không tồn tại trong dataset '{name}'")
                continue

            out_file = raw_dir / f"{split}.jsonl"
            print(f"  → Converting Split: {split} → {out_file}")

            try:
                convert_fn = cfg["convert_fn"]
                convert_fn(ds[split], str(out_file))
            except Exception:
                print(f"  !!! Error converting {name}:{split}")
                traceback.print_exc()

    print("\n✨ Done converting all datasets!")

if __name__ == "__main__":
    main()
