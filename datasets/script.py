from utils.convert import *
from datasets import load_dataset

# hotpot qa
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
convert_hotpot_dataset(ds["train"], "./hotpotqa/hotpot_train_unified.jsonl")
convert_hotpot_dataset(ds["validation"], "./hotpotqa/hotpot_valid_unified.jsonl")

# vimed
ds = load_dataset("tmnam20/ViMedAQA", "all")
convert_vimed_dataset(ds['train'], "./vimed/vimed_train_unified.jsol")
convert_vimed_dataset(ds['test'], "./vimed/vimed_test_unified.jsol")
convert_vimed_dataset(ds['validation'], "./vimed/vimed_valid_unified.jsol")

# viquad
ds = load_dataset("taidng/UIT-ViQuAD2.0")
convert_viquad_dataset(ds['train'], "./viquad/viquad_train_unified.jsol")
convert_viquad_dataset(ds['test'], "./viquad/viquad_test_unified.jsol")
convert_viquad_dataset(ds['validation'], "./viquad/viquad_valid_unified.jsol")