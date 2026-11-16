import os
import random

import torch

from src.data.packet_dataset import PacketDataset
from src.model.language_model import FourmulaMindLM
from src.training.loss import LanguageModelLoss
from src.training.trainer import Trainer


# ============================================================
# Reproducibility
# ============================================================

random.seed(42)

torch.manual_seed(42)


# ============================================================
# Paths
# ============================================================

TRAIN_PACKET_DIR = "data/packets/run4_train"

VALIDATION_PACKET_DIR = (
    "data/packets/validation"
)

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = "checkpoints/4mulamind_run4.pt"


# ============================================================
# Create Directories
# ============================================================

os.makedirs(
    "checkpoints",
    exist_ok=True
)


# ============================================================
# Configuration
# ============================================================

block_size = 128

embedding_dim = 256
num_heads = 8
hidden_dim = 1024
num_layers = 6

learning_rate = 0.001
batch_size = 8

training_steps = 20000
warmup_steps = 1000

patience = 5


# ============================================================
# Header
# ============================================================

print()
print("================================")
print("4mulaMind LLM")
print("Packet-Based Training")
print("================================")


# ============================================================
# Load Tokenizer Metadata
# ============================================================

print()
print("================================")
print("Tokenizer")
print("================================")

print()
print("Tokenizer:")
print(TOKENIZER_PATH)


# We only need the tokenizer vocabulary size here.
# The actual tokenization has already been performed
# during packet preparation.

from src.tokenizer import Tokenizer

tokenizer = Tokenizer.load(
    TOKENIZER_PATH
)


print()
print("Vocabulary size:")
print(tokenizer.vocab_size)

print()
print("BPE merges:")
print(len(tokenizer.merges))


# ============================================================
# Load Packet Datasets
# ============================================================

print()
print("================================")
print("Packet Datasets")
print("================================")


train_dataset = PacketDataset(
    TRAIN_PACKET_DIR,
    block_size=block_size,
    stride=block_size
)

validation_dataset = PacketDataset(
    VALIDATION_PACKET_DIR,
    block_size=block_size,
    stride=block_size
)


print()
print("Training packets:")
print(
    train_dataset.info()["packet_count"]
)

print()
print("Validation packets:")
print(
    validation_dataset.info()["packet_count"]
)

print()
print("Training tokens:")
print(
    train_dataset.info()["tokens"]
)

print()
print("Validation tokens:")
print(
    validation_dataset.info()["tokens"]
)

print()
print("Training samples:")
print(
    len(train_dataset)
)

print()
print("Validation samples:")
print(
    len(validation_dataset)
)


# ============================================================
# Verify Dataset
# ============================================================

print()
print("================================")
print("Dataset Verification")
print("================================")


train_x, train_y = train_dataset[0]

print()
print("Training X shape:")
print(train_x.shape)

print()
print("Training Y shape:")
print(train_y.shape)

print()
print("First 10 X tokens:")
print(train_x[:10])

print()
print("First 10 Y tokens:")
print(train_y[:10])


if train_x.shape[0] != block_size:

    raise RuntimeError(
        "Training dataset returned "
        "an unexpected block size."
    )


if train_y.shape[0] != block_size:

    raise RuntimeError(
        "Training target returned "
        "an unexpected block size."
    )


# ============================================================
# Model
# ============================================================

print()
print("================================")
print("Model")
print("================================")


model = FourmulaMindLM(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print()
print("Vocabulary size:")
print(tokenizer.vocab_size)

print()
print("Block size:")
print(block_size)

print()
print("Embedding dimension:")
print(embedding_dim)

print()
print("Attention heads:")
print(num_heads)

print()
print("Hidden dimension:")
print(hidden_dim)

print()
print("Transformer layers:")
print(num_layers)

print()
print("Total parameters:")
print(parameter_count)


# ============================================================
# Loss
# ============================================================

print()
print("================================")
print("Loss")
print("================================")


loss_function = (
    LanguageModelLoss()
)


print(
    "Loss function created."
)


# ============================================================
# Trainer
# ============================================================

print()
print("================================")
print("Trainer")
print("================================")


trainer = Trainer(
    model=model,
    dataset=train_dataset,
    validation_dataset=validation_dataset,
    loss_function=loss_function,
    learning_rate=learning_rate,
    batch_size=batch_size,
    warmup_steps=warmup_steps,
    max_steps=training_steps,
    patience=patience
)


print(
    "Trainer created."
)


# ============================================================
# Training
# ============================================================

print()
print("================================")
print("Training")
print("================================")

print()
print("Training steps:")
print(training_steps)

print()
print("Learning rate:")
print(learning_rate)

print()
print("Batch size:")
print(batch_size)

print()
print("Block size:")
print(block_size)


best_val_loss = trainer.train(
    steps=training_steps
)


# ============================================================
# Completed
# ============================================================

print()
print("================================")
print("Training Completed")
print("================================")

print()
print("Best validation loss:")
print(best_val_loss)

print()
print("Vocabulary:")
print(tokenizer.vocab_size)

print()
print("Parameters:")
print(parameter_count)

print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Training completed successfully.")