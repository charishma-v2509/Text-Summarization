from training.dataset import CNNDailyMailExtractiveDataset

ds = CNNDailyMailExtractiveDataset(split="train", max_samples=100)

print("Dataset length:", len(ds))
print("First sample:", ds[0])
