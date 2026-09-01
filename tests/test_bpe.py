from src.tokenizer import Tokenizer


# --------------------------------
# Training text
# --------------------------------

text = "hello hello hello"

tokenizer = Tokenizer(text)


# --------------------------------
# Train BPE
# --------------------------------

print("Training BPE:\n")

tokenizer.train_bpe(
    text,
    num_merges=5
)


# --------------------------------
# Save
# --------------------------------

tokenizer.save(
    "tokenizer.json"
)

print("\nTokenizer saved.")


# --------------------------------
# Encode before loading
# --------------------------------

test_text = "hello hello"

encoded_before = tokenizer.encode(
    test_text
)

print("\nEncoded before loading:")
print(encoded_before)


# --------------------------------
# Load
# --------------------------------

loaded_tokenizer = Tokenizer.load(
    "tokenizer.json"
)

print("\nTokenizer loaded.")


# --------------------------------
# Encode after loading
# --------------------------------

encoded_after = loaded_tokenizer.encode(
    test_text
)

print("\nEncoded after loading:")
print(encoded_after)


# --------------------------------
# Decode
# --------------------------------

decoded = loaded_tokenizer.decode(
    encoded_after
)

print("\nDecoded:")
print(decoded)


# --------------------------------
# Verification
# --------------------------------

print("\n==============================")
print("Verification")
print("==============================")


if encoded_before == encoded_after:

    print("✓ Encoding preserved")

else:

    print("✗ Encoding mismatch")


if decoded == test_text:

    print("✓ Decode successful")

else:

    print("✗ Decode failed")