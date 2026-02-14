import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_encoder_layers,
        num_decoder_layers,
        max_src_len=512,
        max_tgt_len=128,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_len=max_src_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_len=max_tgt_len,
            dropout=dropout
        )

        # Final linear layer to vocabulary
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):
        """
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        """

        # Encode source sequence
        enc_output = self.encoder(src, src_mask)

        # Decode target sequence
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        # Project to vocabulary
        logits = self.output_projection(dec_output)

        return logits
