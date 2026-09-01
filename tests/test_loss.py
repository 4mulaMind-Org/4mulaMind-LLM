import torch

from src.loss import LanguageModelLoss


batch_size = 1
sequence_length = 8
vocab_size = 13


logits = torch.randn(
    batch_size,
    sequence_length,
    vocab_size
)

targets = torch.randint(
    0,
    vocab_size,
    (
        batch_size,
        sequence_length
    )
)


loss_function = LanguageModelLoss()

loss = loss_function(
    logits,
    targets
)


print("Logits shape:")
print(logits.shape)

print("\nTargets shape:")
print(targets.shape)

print("\nLoss:")
print(loss.item())