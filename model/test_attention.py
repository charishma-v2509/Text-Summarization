import torch
from attention import ScaledDotProductAttention

batch_size = 2
seq_len = 4
d_k = 8

Q = torch.rand(batch_size, seq_len, d_k)
K = torch.rand(batch_size, seq_len, d_k)
V = torch.rand(batch_size, seq_len, d_k)

attention = ScaledDotProductAttention()
output, weights = attention(Q, K, V)

print("Output shape:", output.shape)
print("Attention weights shape:", weights.shape)
print("Row sum (should be 1):", weights[0, 0].sum())
