import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.positional_encoding import PositionalEncoding
from model.encoder import PositionwiseFeedForward


# -----------------------------
# Decoder Layer
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # 1. Masked self-attention
        attn1 = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))

        # 2. Encoder–decoder attention
        attn2 = self.enc_dec_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # 3. Feed-forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))

        return x


# -----------------------------
# Decoder Stack
# -----------------------------
class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=128,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        x: (batch_size, tgt_seq_len)
        enc_output: (batch_size, src_seq_len, d_model)
        """

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        return x
