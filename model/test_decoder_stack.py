import torch
from decoder import Decoder

batch_size = 2
tgt_len = 7
src_len = 10
vocab_size = 20000
d_model = 16
num_heads = 4
d_ff = 64
num_layers = 3

x = torch.randint(0, vocab_size, (batch_size, tgt_len))
enc_output = torch.rand(batch_size, src_len, d_model)

decoder = Decoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

out = decoder(x, enc_output)

print("Decoder stack output shape:", out.shape)
