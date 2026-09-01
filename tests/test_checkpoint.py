import torch

from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM


# --------------------------------
# Configuration
# --------------------------------

text = "hello 4mulaMind " * 20

block_size = 8

embedding_dim = 32
num_heads = 4
hidden_dim = 128

checkpoint_path = "checkpoints/4mulamind_best.pt"


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = Tokenizer(text)

vocab_size = len(tokenizer.char_to_id)


# --------------------------------
# Create model
# --------------------------------

model = FourmulaMindLM(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    hidden_dim=hidden_dim
)


# --------------------------------
# Load checkpoint
# --------------------------------

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


print("Model loaded successfully.")

print(
    f"Best validation loss: "
    f"{checkpoint['validation_loss']:.4f}"
)


# --------------------------------
# Evaluation mode
# --------------------------------

model.eval()


# --------------------------------
# Generate test input
# --------------------------------

prompt = "hello"

input_ids = tokenizer.encode(prompt)

x = torch.tensor(
    [input_ids],
    dtype=torch.long
)


# --------------------------------
# Forward pass
# --------------------------------

with torch.no_grad():

    logits = model(x)


# --------------------------------
# Get predicted token
# --------------------------------

next_token_id = torch.argmax(
    logits[:, -1, :],
    dim=-1
).item()


next_token = tokenizer.id_to_char[
    next_token_id
]


# --------------------------------
# Result
# --------------------------------

print("\nPrompt:")
print(prompt)

print("\nPredicted next character:")
print(next_token)