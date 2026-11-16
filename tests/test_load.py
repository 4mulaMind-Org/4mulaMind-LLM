from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.training.checkpoint import load_checkpoint
from src.generate import generate


# --------------------------------
# Training text
# --------------------------------

text = "hello 4mulaMind " * 20


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = Tokenizer(text)


# --------------------------------
# Model configuration
# --------------------------------

vocab_size = len(tokenizer.char_to_id)

model = FourmulaMindLM(
    vocab_size=vocab_size,
    block_size=8,
    embedding_dim=32,
    num_heads=4,
    hidden_dim=128
)


# --------------------------------
# Load trained model
# --------------------------------

load_checkpoint(
    model,
    "checkpoints/4mulamind.pt"
)


# --------------------------------
# Generate
# --------------------------------

result = generate(
    model=model,
    tokenizer=tokenizer,
    prompt="hello",
    max_new_tokens=30
)


print("\n================================")
print("Generated Text:")
print("================================")

print(result)