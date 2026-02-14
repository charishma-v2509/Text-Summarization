import torch
import re
import sentencepiece as spm

from model.extractive_model import ExtractiveSummarizer
from training.config import VOCAB_SIZE, PAD_IDX, MAX_INPUT_LENGTH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def load_model(checkpoint_path):
    model = ExtractiveSummarizer(vocab_size=VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def summarize(article, model, sp, summary_length="medium"):

    sentences = split_into_sentences(article)

    if len(sentences) == 0:
        return "No valid sentences found."

    tokenized_sentences = []

    for sent in sentences:
        token_ids = sp.encode(sent, out_type=int)

        if len(token_ids) > MAX_INPUT_LENGTH:
            token_ids = token_ids[:MAX_INPUT_LENGTH]

        tokenized_sentences.append(torch.tensor(token_ids))

    scores = []

    with torch.no_grad():
        for i, tokens in enumerate(tokenized_sentences):
            tokens = tokens.unsqueeze(0).to(DEVICE)
            score = model(tokens)
            position_bonus = 1 / (i + 1)
            final_score = score.item() + 0.1 * position_bonus
            scores.append(final_score)


    # Decide number of sentences based on length choice
    if summary_length == "short":
        k = 3
    elif summary_length == "long":
        k = 8
    else:
        k = 5  # medium

    # Get top-k sentence indices
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    selected_indices = sorted(ranked_indices[:k])

    summary = " ".join([sentences[i] for i in selected_indices])

    return summary


if __name__ == "__main__":

    print("Loading model...")
    model = load_model("checkpoints/extractive_epoch5.pth")

    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/spm.model")

    article = """
    LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access 
    to a reported £20 million fortune as he turns 18 on Monday. 
    The young actor insists the money will not change him. 
    Radcliffe has earned the fortune from the Harry Potter films. 
    He says he plans to invest wisely and continue acting.
    """

    summary = summarize(article, model, sp, summary_length="medium")

    print("\n===== GENERATED SUMMARY =====\n")
    print(summary)
