import torch

from src.feed_forward import FeedForward


batch_size = 1
sequence_length = 8
embedding_dim = 32
hidden_dim = 128


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


feed_forward = FeedForward(
    embedding_dim,
    hidden_dim
)


output = feed_forward(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)