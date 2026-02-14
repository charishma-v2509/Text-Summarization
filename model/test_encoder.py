import torch
from encoder import EncoderLayer

batch_size = 2
seq_len = 6
d_model = 16
num_heads = 4
d_ff = 64

x = torch.rand(batch_size, seq_len, d_model)

encoder_layer = EncoderLayer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff
)

out = encoder_layer(x)

print("Encoder output shape:", out.shape)
