import torch

from src.language_model import FourmulaMindLM


batch_size = 1
sequence_length = 8

vocab_size = 13
embedding_dim = 32
num_heads = 4
hidden_dim = 128


x = torch.randint(
    0,
    vocab_size,
    (
        batch_size,
        sequence_length
    )
)


model = FourmulaMindLM(
    vocab_size=vocab_size,
    block_size=sequence_length,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)


logits = model(x)


print("Input shape:")
print(x.shape)

print("\nLogits shape:")
print(logits.shape)