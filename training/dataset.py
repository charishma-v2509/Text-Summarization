import re
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import sentencepiece as spm
from training.config import MAX_INPUT_LENGTH


def split_into_sentences(text):
    # Simple sentence splitter (good enough, no dependency)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def tokenize_words(text):
    return set(re.findall(r"\w+", text.lower()))


class CNNDailyMailExtractiveDataset(Dataset):
    def __init__(self, split="train", max_samples=10000):
        self.dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split=f"{split}[:{max_samples}]"
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load("tokenizer/spm.model")

        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for item in self.dataset:
            article = item["article"]
            summary = item["highlights"]

            article_sents = split_into_sentences(article)
            summary_sents = split_into_sentences(summary)

            summary_words = set()
            for s in summary_sents:
                summary_words |= tokenize_words(s)

            for sent in article_sents:
                sent_words = tokenize_words(sent)

                # Overlap-based label
                overlap = len(sent_words & summary_words)
                label = 1 if overlap >= 3 else 0

                token_ids = self.sp.encode(sent, out_type=int)

# Manual truncation
                if len(token_ids) > MAX_INPUT_LENGTH:
                    token_ids = token_ids[:MAX_INPUT_LENGTH]


                self.data.append((
                    torch.tensor(token_ids, dtype=torch.long),
                    torch.tensor(label, dtype=torch.float)
                ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
