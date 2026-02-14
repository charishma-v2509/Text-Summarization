import torch
import torch.nn as nn
from model.encoder import Encoder


class ExtractiveSummarizer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2
    ):
        super().__init__()

        # Use your existing Encoder (which already has embedding + positional encoding)
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, input_ids, src_mask=None):

        enc_output = self.encoder(input_ids, src_mask)

        # Mean pooling over sequence dimension
        sentence_embedding = enc_output.mean(dim=1)

        logits = self.classifier(sentence_embedding)

        return torch.sigmoid(logits).squeeze(-1)
