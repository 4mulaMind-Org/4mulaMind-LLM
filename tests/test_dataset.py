from src.tokenizer import Tokenizer
from src.dataset import TextDataset


# --------------------------------
# Training text
# --------------------------------

text = "hello hello hello " * 10


# --------------------------------
# Tokenizer
# --------------------------------

tokenizer = Tokenizer(text)


# --------------------------------
# Train BPE
# --------------------------------

tokenizer.train_bpe(
    text,
    num_merges=5
)


# --------------------------------
# Dataset
# --------------------------------

dataset = TextDataset(
    text,
    tokenizer,
    block_size=8
)


# --------------------------------
# Results
# --------------------------------

print("BPE vocabulary size:")
print(tokenizer.vocab_size)

print("\nTotal BPE tokens:")
print(len(tokenizer.encode(text)))

print("\nTraining samples:")
print(len(dataset))


x, y = dataset[0]

print("\nInput:")
print(x)

print("\nTarget:")
print(y)

print("\nInput shape:")
print(x.shape)

print("\nTarget shape:")
print(y.shape)