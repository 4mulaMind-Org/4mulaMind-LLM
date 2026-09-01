import json
import torch

from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.checkpoint import load_checkpoint
from src.generate import generate


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = (
    "checkpoints/4mulamind_bpe.pt"
)

VALIDATION_PATH = (
    "data/validation.jsonl"
)


# ============================================================
# Model Configuration
# Must match training
# ============================================================

block_size = 128

embedding_dim = 128
num_heads = 4
hidden_dim = 512
num_layers = 4


# ============================================================
# Load Tokenizer
# ============================================================

print()
print("================================")
print("Loading Tokenizer")
print("================================")

tokenizer = Tokenizer.load(
    TOKENIZER_PATH
)

print(
    "Vocabulary:",
    tokenizer.vocab_size
)

print(
    "BPE merges:",
    len(tokenizer.merges)
)


# ============================================================
# Create Model
# ============================================================

print()
print("================================")
print("Creating Model")
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

print(
    "Parameters:",
    parameter_count
)


# ============================================================
# Load Checkpoint
# ============================================================

print()
print("================================")
print("Loading Checkpoint")
print("================================")

model = load_checkpoint(
    model,
    CHECKPOINT_PATH
)


# ============================================================
# Load Validation Documents
# ============================================================

validation_texts = []

with open(
    VALIDATION_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        record = json.loads(line)

        text = record.get(
            "text",
            ""
        ).strip()

        if text:

            validation_texts.append(
                text
            )


# ============================================================
# Select Validation Examples
# ============================================================

# Use fixed examples so every evaluation
# can be compared consistently.

examples = validation_texts[:5]


# ============================================================
# Evaluation
# ============================================================

print()
print("================================")
print("Validation Generation")
print("================================")


for example_number, actual_text in enumerate(
    examples,
    start=1
):

    # --------------------------------
    # Convert document to tokens
    # --------------------------------

    actual_tokens = tokenizer.encode(
        actual_text
    )

    if len(actual_tokens) < 20:
        continue

    # --------------------------------
    # Use beginning as prompt
    # --------------------------------

    prompt_tokens = actual_tokens[
        :32
    ]

    prompt = tokenizer.decode(
        prompt_tokens
    )

    # --------------------------------
    # Actual continuation
    # --------------------------------

    actual_continuation = tokenizer.decode(
        actual_tokens[32:82]
    )

    # --------------------------------
    # Model generation
    # --------------------------------

    generated = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=20
    )

    # Remove prompt from generated text
    generated_continuation = generated[
        len(prompt):
    ]

    # --------------------------------
    # Print
    # --------------------------------

    print()
    print(
        f"Example {example_number}"
    )

    print()
    print(
        "Prompt:"
    )

    print(prompt)

    print()
    print(
        "Actual continuation:"
    )

    print(actual_continuation)

    print()
    print(
        "Model continuation:"
    )

    print(generated_continuation)

    print()
    print("--------------------------------")