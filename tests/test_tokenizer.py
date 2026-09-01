from src.tokenizer import Tokenizer


text = "hello 4mulaMind"

tokenizer = Tokenizer(text)

encoded = tokenizer.encode(text)

decoded = tokenizer.decode(encoded)


print("Characters:")
print(tokenizer.chars)

print("\nCharacter to ID:")
print(tokenizer.char_to_id)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decoded)