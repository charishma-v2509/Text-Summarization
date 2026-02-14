import torch
from decoder import DecoderLayer

batch_size = 2
tgt_len = 6
src_len = 10
d_model = 16
num_heads = 4
d_ff = 64

x = torch.rand(batch_size, tgt_len, d_model)
enc_output = torch.rand(batch_size, src_len, d_model)

decoder_layer = DecoderLayer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff
)

out = decoder_layer(x, enc_output)

print("Decoder output shape:", out.shape)
