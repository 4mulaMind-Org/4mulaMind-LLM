import torch

from src.language_model import FourmulaMindLM
from src.loss import LanguageModelLoss


# -----------------------------
# Configuration
# -----------------------------

vocab_size = 13
block_size = 8

embedding_dim = 32
num_heads = 4
hidden_dim = 128

learning_rate = 0.001


# -----------------------------
# Model
# -----------------------------

model = FourmulaMindLM(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)


# -----------------------------
# Loss
# -----------------------------

loss_function = LanguageModelLoss()


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -----------------------------
# Fake training batch
# -----------------------------

x = torch.randint(
    0,
    vocab_size,
    (4, block_size)
)

targets = torch.randint(
    0,
    vocab_size,
    (4, block_size)
)


# -----------------------------
# Forward pass
# -----------------------------

logits = model(x)


# -----------------------------
# Calculate loss
# -----------------------------

loss = loss_function(
    logits,
    targets
)

print("Loss before backward:")
print(loss.item())


# -----------------------------
# Backpropagation
# -----------------------------

optimizer.zero_grad()

loss.backward()


# -----------------------------
# Check gradients
# -----------------------------

gradient_found = False

for parameter in model.parameters():

    if parameter.grad is not None:

        gradient_found = True

        break


print("\nGradients found:")
print(gradient_found)


# -----------------------------
# Update parameters
# -----------------------------

optimizer.step()

print("\nOptimizer step completed.")