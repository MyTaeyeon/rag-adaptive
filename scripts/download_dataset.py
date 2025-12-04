# create raw_data directory if it doesn't exist
import os
if not os.path.exists("./raw_data"):
    os.makedirs("./raw_data")

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
# save ds['train'] to a json file in ./raw_data/hotpot_train_v1.json
ds['train'].to_json("./raw_data/hotpot_train_v1.json")

ds = load_dataset("sentence-transformers/natural-questions")
# save ds['train'] to a json file in ./raw_data/naturalquestions.jsonl
ds['train'].to_json("./raw_data/naturalquestions.jsonl", lines=True)

ds = load_dataset("microsoft/ms_marco", "v2.1")
# save ds['train'] to a json file in ./raw_data/msmarco.tsv
ds['train'].to_csv("./raw_data/msmarco.tsv", sep="\t", index=False)

ds = load_dataset("mandarjoshi/trivia_qa", "rc")
# save ds['train'] to a json file in ./raw_data/triviaqa.json
ds['train'].to_json("./raw_data/triviaqa.json")

ds = load_dataset("cais/mmlu", "abstract_algebra")
# save ds['test'] to a json file in ./raw_data/mmlu.csv
ds['test'].to_csv("./raw_data/mmlu.csv", index=False)