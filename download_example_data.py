from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
ds.to_json("./example_data/train.jsonl", orient="records", lines=True)
