import torch
from positional_encoding import PositionalEncoding

batch_size = 2
seq_len = 5
d_model = 16

x = torch.zeros(batch_size, seq_len, d_model)
pe = PositionalEncoding(d_model)

out = pe(x)

print("Output shape:", out.shape)
print("First position vector:\n", out[0, 0])
