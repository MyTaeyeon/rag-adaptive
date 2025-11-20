from datasets import load_dataset

ds = load_dataset("tmnam20/ViMedAQA", "all")

ds['train'].to_json("viquad_train.jsonl", lines=True)
ds['validation'].to_json("viquad_valid.jsonl", lines=True)
ds['test'].to_json("viquad_test.jsonl", lines=True)

ds['train'].to_json("viquad_train.json")
ds['validation'].to_json("viquad_validation.json")
ds['test'].to_json("viquad_test.json")

ds['train'].to_csv("viquad_train.csv")
ds['validation'].to_csv("viquad_validation.csv")
ds['test'].to_csv("viquad_test.csv")
