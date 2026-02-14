import sentencepiece as spm
from datasets import load_dataset

# Load raw dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")

# Write raw text to file
with open("tokenizer/corpus.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(item["article"] + "\n")
        f.write(item["highlights"] + "\n")

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input="tokenizer/corpus.txt",
    model_prefix="tokenizer/spm",
    vocab_size=16000,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    model_type="bpe"
)

print("Tokenizer trained successfully.")
