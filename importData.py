from datasets import load_dataset

dataset = load_dataset("matthieulel/galaxy10_decals")

dataset.save_to_disk("data")

print(dataset)