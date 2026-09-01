import json
import torch

from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.checkpoint import load_checkpoint
from src.generate import generate


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = "tokenizers/4mulamind_bpe.json"

CHECKPOINT_PATH = "checkpoints/4mulamind_run3.pt"

TEST_DATA_PATH = "data/test.jsonl"


# ============================================================
# Model Configuration
# ============================================================

BLOCK_SIZE = 128
EMBEDDING_DIM = 256
NUM_HEADS = 8
HIDDEN_DIM = 1024
NUM_LAYERS = 6


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

print("Vocabulary:", tokenizer.vocab_size)
print("BPE merges:", len(tokenizer.merges))


# ============================================================
# Create Model
# ============================================================

print()
print("================================")
print("Creating Model")
print("================================")

model = FourmulaMindLM(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Parameters:", parameter_count)


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

model.eval()


# ============================================================
# Load Test Documents
# ============================================================

test_texts = []

with open(
    TEST_DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        text = record.get(
            "text",
            ""
        ).strip()

        if text:
            test_texts.append(text)


print()
print("Test documents:", len(test_texts))


# ============================================================
# Select Examples
# ============================================================

examples = test_texts[:5]


# ============================================================
# Generation Evaluation
# ============================================================

print()
print("================================")
print("Generation Evaluation")
print("================================")


for example_number, actual_text in enumerate(
    examples,
    start=1
):

    actual_tokens = tokenizer.encode(
        actual_text
    )

    if len(actual_tokens) < 40:
        continue

    # --------------------------------------------------------
    # First 32 tokens = prompt
    # --------------------------------------------------------

    prompt_tokens = actual_tokens[:32]

    prompt = tokenizer.decode(
        prompt_tokens
    )

    # --------------------------------------------------------
    # Actual continuation
    # --------------------------------------------------------

    actual_continuation = tokenizer.decode(
        actual_tokens[32:82]
    )

    # ========================================================
    # Greedy-like generation
    # ========================================================

    greedy_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=1.0,
        top_k=1
    )

    greedy_continuation = greedy_output[
        len(prompt):
    ]

    # ========================================================
    # Sampling generation
    # ========================================================

    sampled_output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=20
    )

    sampled_continuation = sampled_output[
        len(prompt):
    ]

    # ========================================================
    # Print
    # ========================================================

    print()
    print(
        f"Example {example_number}"
    )

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Actual continuation:")
    print(actual_continuation)

    print()
    print("Top-1 continuation:")
    print(greedy_continuation)

    print()
    print("Top-K sampled continuation:")
    print(sampled_continuation)

    print()
    print("--------------------------------")


print()
print("Evaluation completed.")