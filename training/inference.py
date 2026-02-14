import torch
import sentencepiece as spm

from training.config import *
from training.train import create_padding_mask
from model.transformer import Transformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search_decode(
    model,
    src,
    beam_size=3,
    max_len=MAX_OUTPUT_LENGTH,
    length_penalty=0.7
):
    model.eval()
    src_mask = create_padding_mask(src).to(device)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)

        # Each beam: (token_ids, log_prob)
        beams = [([SOS_IDX], 0.0)]

        for _ in range(max_len):
            new_beams = []

            for tokens, score in beams:
                if tokens[-1] == EOS_IDX:
                    new_beams.append((tokens, score))
                    continue

                tgt = torch.tensor([tokens], device=device)
                tgt_mask = create_padding_mask(tgt).to(device)

                dec_output = model.decoder(tgt, enc_output, tgt_mask, src_mask)
                logits = model.output_projection(dec_output)

                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                topk_log_probs, topk_ids = torch.topk(
                    log_probs, beam_size, dim=-1
                )

                for i in range(beam_size):
                    new_tokens = tokens + [topk_ids[0, i].item()]
                    new_score = score + topk_log_probs[0, i].item()
                    new_beams.append((new_tokens, new_score))

            # Length penalty + keep best beams
            beams = sorted(
                new_beams,
                key=lambda x: x[1] / (len(x[0]) ** length_penalty),
                reverse=True
            )[:beam_size]

        best_tokens = beams[0][0]

    return best_tokens



if __name__ == "__main__":
    print("Loading trained model...")

    checkpoint = torch.load(
        "checkpoints/transformer_epoch5.pth",
        map_location=device
    )


    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_encoder_layers=2,
        num_decoder_layers=2
    ).to(device)


    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")

    # Load ONE sample from dataset (no vocab rebuild)
    from training.dataset import CNNDailyMailDataset

    dataset = CNNDailyMailDataset(
        split="train"
    )

    sample = dataset[0]
    src = sample["input_ids"].unsqueeze(0).to(device)

    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/spm.model")

    print("\nINPUT (article snippet):")
    # decode only first 40 tokens for display
    input_text = sp.decode(src[0][:40].tolist())
    print(input_text)

    output_ids = beam_search_decode(
        model,
        src,
        beam_size=3,       # you can try 5 later
        max_len=50
    )


    print("\nGENERATED SUMMARY:")
    decoded_text = sp.decode(output_ids)
    print(decoded_text)
