from training.dataset import CNNDailyMailDataset

print("Loading train dataset...")
train_dataset = CNNDailyMailDataset(
    split="train",
    build_vocab=True
)

print("Dataset size:", len(train_dataset))
print("Vocabulary size:", len(train_dataset.vocab.word2idx))
