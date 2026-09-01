import json
import random


TRAIN_SOURCE = "data/train.jsonl"
VALIDATION_SOURCE = "data/validation.jsonl"

TRAIN_OUTPUT = "data/train.jsonl"
VALIDATION_OUTPUT = "data/validation.jsonl"
TEST_OUTPUT = "data/test.jsonl"

SEED = 42


# ============================================================
# Load existing 10,000 examples
# ============================================================

data = []

for path in [
    TRAIN_SOURCE,
    VALIDATION_SOURCE
]:

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if line:
                data.append(
                    json.loads(line)
                )


print(
    "Total examples:",
    len(data)
)


# ============================================================
# Shuffle
# ============================================================

random.seed(SEED)
random.shuffle(data)


# ============================================================
# 80 / 10 / 10
# ============================================================

total = len(data)

train_end = int(
    total * 0.80
)

validation_end = int(
    total * 0.90
)

train_data = data[:train_end]

validation_data = data[
    train_end:validation_end
]

test_data = data[
    validation_end:
]


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(path, records):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


save_jsonl(
    TRAIN_OUTPUT,
    train_data
)

save_jsonl(
    VALIDATION_OUTPUT,
    validation_data
)

save_jsonl(
    TEST_OUTPUT,
    test_data
)


# ============================================================
# Result
# ============================================================

print()
print("Train:", len(train_data))
print("Validation:", len(validation_data))
print("Test:", len(test_data))
print()
print("80/10/10 split complete.")