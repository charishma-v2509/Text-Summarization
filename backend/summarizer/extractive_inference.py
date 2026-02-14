import os
import sys
model = None
sp = None


# Add backend root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sys
import json
import torch
import re
import sentencepiece as spm

from model.extractive_model import ExtractiveSummarizer

DEVICE = torch.device("cpu")

VOCAB_SIZE = 16000
MAX_INPUT_LENGTH = 128


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def load_model():
    global model, sp
    if model is None:
        model = ExtractiveSummarizer(vocab_size=VOCAB_SIZE)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        checkpoint_path = os.path.join(project_root, "checkpoints", "extractive_epoch5.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()

        sp = spm.SentencePieceProcessor()
        sp.load(os.path.join(project_root, "tokenizer", "spm.model"))

    return model, sp



def summarize(article, summary_length):
    model, sp = load_model()

    sp = spm.SentencePieceProcessor()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    checkpoint_path = os.path.join(project_root, "tokenizer", "spm.model")
    sp.load(checkpoint_path)

    sentences = split_into_sentences(article)

    if len(sentences) == 0:
        return "No valid sentences found."

    scores = []

    with torch.no_grad():
        for sent in sentences:
            token_ids = sp.encode(sent, out_type=int)

            if len(token_ids) > MAX_INPUT_LENGTH:
                token_ids = token_ids[:MAX_INPUT_LENGTH]

            tokens = torch.tensor(token_ids).unsqueeze(0)
            score = model(tokens)
            scores.append(score.item())

    if summary_length == "short":
        k = 2
    elif summary_length == "long":
        k = 6
    else:
        k = 4

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    selected_indices = sorted(ranked_indices[:k])

    summary = " ".join([sentences[i] for i in selected_indices])

    return summary


if __name__ == "__main__":
    input_json = json.loads(sys.stdin.read())

    article = input_json["text"]
    length = input_json.get("mode", "medium")

    summary = summarize(article, length)

    print(json.dumps({"summary": summary}))
