import torch
from attention import MultiHeadAttention

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

x = torch.rand(batch_size, seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
out = mha(x, x, x)

print("Output shape:", out.shape)
