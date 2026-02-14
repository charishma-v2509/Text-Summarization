import torch
from model.extractive_model import ExtractiveSummarizer

model = ExtractiveSummarizer(vocab_size=16000)

dummy_input = torch.randint(0, 16000, (4, 50))  # batch of 4 sentences

output = model(dummy_input)

print("Output shape:", output.shape)
