from datasets import load_dataset

ds = load_dataset("taidng/UIT-ViQuAD2.0")

# ds['train'].to_csv("viquad_train.csv")
# ds['validation'].to_csv("viquad_validation.csv")
# ds['test'].to_csv("viquad_test.csv")

ds['train'].to_json("viquad_train.jsonl", lines=True)
ds['validation'].to_json("viquad_validation.jsonl", lines=True)
ds['test'].to_json("viquad_test.jsonl", lines=True)
