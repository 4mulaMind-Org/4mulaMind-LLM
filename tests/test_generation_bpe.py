import torch

from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.generation import Generator


# ============================================================
# Configuration
# ============================================================

BLOCK_SIZE = 64

# IMPORTANT:
# These values MUST match the model that was trained.
EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 512
NUM_LAYERS = 4

TOKENIZER_PATH = "tokenizers/4mulamind_bpe.json"
CHECKPOINT_PATH = "checkpoints/4mulamind_bpe.pt"


# ============================================================
# Header
# ============================================================

print()
print("================================")
print("4mulaMind LLM")
print("================================")


# ============================================================
# Load Tokenizer
# ============================================================

print()
print("Loading tokenizer...")

tokenizer = Tokenizer.load(
    TOKENIZER_PATH
)

print("✓ Tokenizer loaded")
print("Vocabulary size:", tokenizer.vocab_size)
print("BPE merges:", len(tokenizer.merges))


# ============================================================
# Create Model
# ============================================================

print()
print("Loading model...")

model = FourmulaMindLM(
    vocab_size=tokenizer.vocab_size,
    block_size=BLOCK_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

print("✓ Model created")


# ============================================================
# Load Checkpoint
# ============================================================

print()
print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

checkpoint_state = checkpoint["model_state_dict"]

checkpoint_vocab_size = (
    checkpoint_state["token_embedding.weight"].shape[0]
)

checkpoint_embedding_dim = (
    checkpoint_state["token_embedding.weight"].shape[1]
)

print()
print("Checkpoint vocabulary:", checkpoint_vocab_size)
print("Current tokenizer vocabulary:", tokenizer.vocab_size)

print("Checkpoint embedding:", checkpoint_embedding_dim)
print("Current model embedding:", EMBEDDING_DIM)


# ============================================================
# Validate configuration
# ============================================================

if tokenizer.vocab_size != checkpoint_vocab_size:

    raise RuntimeError(
        "\nTokenizer / checkpoint mismatch!\n"
        f"Tokenizer vocabulary = {tokenizer.vocab_size}\n"
        f"Checkpoint vocabulary = {checkpoint_vocab_size}\n\n"
        "Delete the old tokenizer and retrain the model."
    )


if EMBEDDING_DIM != checkpoint_embedding_dim:

    raise RuntimeError(
        "\nModel / checkpoint mismatch!\n"
        f"Model embedding dimension = {EMBEDDING_DIM}\n"
        f"Checkpoint embedding dimension = {checkpoint_embedding_dim}\n\n"
        "Use the same model configuration used during training."
    )


# ============================================================
# Load Weights
# ============================================================

model.load_state_dict(
    checkpoint_state
)

print()
print("✓ Checkpoint loaded")

print(
    "Best validation loss:",
    checkpoint["best_val_loss"]
)


# ============================================================
# Evaluation Mode
# ============================================================

model.eval()


# ============================================================
# Generator
# ============================================================

generator = Generator(
    model=model,
    tokenizer=tokenizer,
    device="cpu"
)

print("✓ Generator ready")


# ============================================================
# Interactive Chat
# ============================================================

print()
print("================================")
print("4mulaMind Interactive Generation")
print("================================")

print()
print("Type your prompt and press Enter.")
print("Type 'exit' or 'quit' to stop.")
print()


while True:

    prompt = input("You: ").strip()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if prompt.lower() in {
        "exit",
        "quit"
    }:

        print()
        print("4mulaMind: Goodbye!")
        break


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not prompt:

        print(
            "4mulaMind: Please enter a prompt."
        )

        continue


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    try:

        generated_text = generator.generate(
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.15
        )

        print()
        print("4mulaMind:")
        print(generated_text)
        print()

    except Exception as error:

        print()
        print(
            "Generation error:",
            error
        )
        print()
