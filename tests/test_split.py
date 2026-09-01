from src.tokenizer import Tokenizer
from src.dataset import TextDataset


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
# Datasets
# --------------------------------

train_dataset = TextDataset(
    train_tokens,
    block_size=8
)

validation_dataset = TextDataset(
    validation_tokens,
    block_size=8
)


# --------------------------------
# Information
# --------------------------------

print("Total tokens:")
print(len(tokens))

print("\nTraining tokens:")
print(len(train_tokens))

print("\nValidation tokens:")
print(len(validation_tokens))

print("\nTraining samples:")
print(len(train_dataset))

print("\nValidation samples:")
print(len(validation_dataset))


# --------------------------------
# Sample
# --------------------------------

train_x, train_y = train_dataset[0]

val_x, val_y = validation_dataset[0]

print("\nTraining input:")
print(train_x)

print("\nValidation input:")
print(val_x)