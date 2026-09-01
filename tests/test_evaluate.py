from src.tokenizer import Tokenizer
from src.dataset import TextDataset
from src.language_model import FourmulaMindLM
from src.evaluate import evaluate


# --------------------------------
# Text
# --------------------------------

text = "hello 4mulaMind " * 20


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = Tokenizer(text)

tokens = tokenizer.encode(text)


# --------------------------------
# Train / Validation split
# --------------------------------

split_index = int(len(tokens) * 0.9)

train_tokens = tokens[:split_index]
validation_tokens = tokens[split_index:]


# --------------------------------
# Validation dataset
# --------------------------------

validation_dataset = TextDataset(
    validation_tokens,
    block_size=8
)


# --------------------------------
# Model
# --------------------------------

model = FourmulaMindLM(
    vocab_size=len(tokenizer.char_to_id),
    block_size=8,
    embedding_dim=32,
    num_heads=4,
    hidden_dim=128
)


# --------------------------------
# Evaluate
# --------------------------------

validation_loss = evaluate(
    model,
    validation_dataset
)


print("Validation Loss:")
print(validation_loss)