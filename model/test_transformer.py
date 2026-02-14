import torch
from transformer import Transformer

batch_size = 2
src_len = 12
tgt_len = 6
src_vocab_size = 20000
tgt_vocab_size = 20000

d_model = 16
num_heads = 4
d_ff = 64
num_layers = 3

src = torch.randint(0, src_vocab_size, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_encoder_layers=num_layers,
    num_decoder_layers=num_layers
)

out = model(src, tgt)

print("Transformer output shape:", out.shape)
