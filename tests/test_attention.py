import torch

from src.attention import MultiHeadAttention


batch_size = 1
sequence_length = 8
embedding_dim = 32
num_heads = 4


x = torch.randn(
    batch_size,
    sequence_length,
    embedding_dim
)


attention = MultiHeadAttention(
    embedding_dim,
    num_heads
)


output = attention(x)


print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)