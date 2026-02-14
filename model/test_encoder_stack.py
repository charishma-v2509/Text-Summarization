import torch
from encoder import Encoder

batch_size = 2
seq_len = 10
vocab_size = 20000
d_model = 16
num_heads = 4
d_ff = 64
num_layers = 3

x = torch.randint(0, vocab_size, (batch_size, seq_len))

encoder = Encoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers
)

out = encoder(x)

print("Encoder stack output shape:", out.shape)
