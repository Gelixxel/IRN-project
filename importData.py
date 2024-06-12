from datasets import load_dataset

def load_galaxy_data():
    dataset = load_dataset("matthieulel/galaxy10_decals")
    return dataset

#dataset.save_to_disk("data")
#print(dataset)