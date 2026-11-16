import math
import torch

from src.data.packet_dataset import PacketDataset
from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.checkpoint import load_checkpoint


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = (
    "checkpoints/4mulamind_run4.pt"
)

TEST_PACKET_DIR = (
    "data/packets/test"
)


# ============================================================
# Evaluation Configuration
# ============================================================

BATCH_SIZE = 8

BLOCK_SIZE = 128

# None = full test set
NUM_BATCHES = None


# ============================================================
# Load Checkpoint Metadata
# ============================================================

print()
print("================================")
print("Loading Checkpoint Metadata")
print("================================")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

if (
    not isinstance(checkpoint, dict)
    or "model_state_dict" not in checkpoint
):

    raise RuntimeError(
        "Invalid checkpoint format."
    )


checkpoint_config = {
    "vocab_size": checkpoint["vocab_size"],
    "block_size": checkpoint["block_size"],
    "embedding_dim": checkpoint["embedding_dim"],
    "num_heads": checkpoint["num_heads"],
    "hidden_dim": checkpoint["hidden_dim"],
    "num_layers": checkpoint["num_layers"]
}


print()
print("Checkpoint:")
print(CHECKPOINT_PATH)

print()
print("Step:")
print(checkpoint.get("step"))

print()
print("Best validation loss:")
print(
    checkpoint.get(
        "best_val_loss"
    )
)

print()
print("Configuration:")
print(
    checkpoint_config
)


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

print()
print("Vocabulary:")
print(tokenizer.vocab_size)

print()
print("BPE merges:")
print(len(tokenizer.merges))


# ============================================================
# Verify Tokenizer / Checkpoint
# ============================================================

if tokenizer.vocab_size != (
    checkpoint_config["vocab_size"]
):

    raise RuntimeError(
        "Tokenizer vocabulary size does not "
        "match checkpoint vocabulary size."
    )


# ============================================================
# Load Test Dataset
# ============================================================

print()
print("================================")
print("Loading Test Dataset")
print("================================")

test_dataset = PacketDataset(
    TEST_PACKET_DIR,
    block_size=checkpoint_config[
        "block_size"
    ],
    stride=checkpoint_config[
        "block_size"
    ]
)


print()
print("Test packets:")
print(
    test_dataset.info()["packet_count"]
)

print()
print("Source tokens:")
print(
    test_dataset.info()["tokens"]
)

print()
print("Test samples:")
print(len(test_dataset))


# ============================================================
# Create Model
# ============================================================

print()
print("================================")
print("Creating Model")
print("================================")

model = FourmulaMindLM(
    vocab_size=checkpoint_config[
        "vocab_size"
    ],
    block_size=checkpoint_config[
        "block_size"
    ],
    embedding_dim=checkpoint_config[
        "embedding_dim"
    ],
    num_heads=checkpoint_config[
        "num_heads"
    ],
    hidden_dim=checkpoint_config[
        "hidden_dim"
    ],
    num_layers=checkpoint_config[
        "num_layers"
    ]
)


parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print()
print("Parameters:")
print(parameter_count)


# ============================================================
# Load Weights
# ============================================================

print()
print("================================")
print("Loading Model Weights")
print("================================")

model = load_checkpoint(
    model,
    CHECKPOINT_PATH
)

model.eval()


# ============================================================
# Number of Batches
# ============================================================

available_batches = (
    len(test_dataset)
    + BATCH_SIZE
    - 1
) // BATCH_SIZE


if NUM_BATCHES is None:

    num_batches = available_batches

else:

    num_batches = min(
        NUM_BATCHES,
        available_batches
    )


# ============================================================
# Evaluation
# ============================================================

print()
print("================================")
print("Test Evaluation")
print("================================")

print()
print("Batches:")
print(num_batches)

print()
print("Batch size:")
print(BATCH_SIZE)

print()
print("Block size:")
print(
    checkpoint_config["block_size"]
)


total_loss = 0.0

total_tokens = 0


with torch.no_grad():

    for batch_number in range(
        num_batches
    ):

        start = (
            batch_number
            * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            len(test_dataset)
        )

        inputs = []

        targets = []

        for index in range(
            start,
            end
        ):

            x, y = test_dataset[
                index
            ]

            inputs.append(x)

            targets.append(y)

        x = torch.stack(
            inputs
        )

        y = torch.stack(
            targets
        )

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1)
        )

        token_count = y.numel()

        total_loss += (
            loss.item()
            * token_count
        )

        total_tokens += (
            token_count
        )

        if (
            (batch_number + 1) % 100 == 0
            or
            (batch_number + 1) == num_batches
        ):

            running_loss = (
                total_loss
                / total_tokens
            )

            print(
                f"Batch "
                f"{batch_number + 1}/"
                f"{num_batches}"
                f" | Loss: "
                f"{running_loss:.4f}"
            )


# ============================================================
# Final Metrics
# ============================================================

test_loss = (
    total_loss
    / total_tokens
)

perplexity = math.exp(
    test_loss
)


# ============================================================
# Results
# ============================================================

print()
print("================================")
print("Run 3 Test Results")
print("================================")

print()
print("Batches evaluated:")
print(num_batches)

print()
print("Tokens evaluated:")
print(total_tokens)

print()
print(
    f"Test Loss: {test_loss:.6f}"
)

print()
print(
    f"Perplexity: {perplexity:.4f}"
)

print()
print("Evaluation completed.")