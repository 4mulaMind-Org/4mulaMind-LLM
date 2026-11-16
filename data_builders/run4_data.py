import hashlib
import json
import random


SOURCE = "/Users/abdulqadir/Downloads/000_00041.jsonl"

VALIDATION_FILE = "data/validation.jsonl"
TEST_FILE = "data/test.jsonl"

OUTPUT = "data/run4_train.jsonl"

TARGET_DOCS = 50_000

SEED = 42


# ============================================================
# Text Hash
# ============================================================

def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# Load Held-Out Texts
# ============================================================

def load_hashes(path):

    hashes = set()

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                "text",
                ""
            )

            if not isinstance(text, str):
                continue

            text = text.strip()

            if text:

                hashes.add(
                    text_hash(text)
                )

    return hashes


# ============================================================
# Leakage-Safe Reservoir Sampling
# ============================================================

def reservoir_sample(
    source,
    excluded_hashes,
    target,
    seed=42
):

    rng = random.Random(seed)

    reservoir = []

    valid_source = 0
    excluded = 0

    with open(
        source,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            text = record.get(
                "text",
                ""
            )

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            fingerprint = text_hash(text)

            # --------------------------------
            # Prevent test/validation leakage
            # --------------------------------

            if fingerprint in excluded_hashes:

                excluded += 1
                continue

            valid_source += 1

            clean_record = {
                "text": text
            }

            # --------------------------------
            # Reservoir sampling
            # --------------------------------

            if len(reservoir) < target:

                reservoir.append(
                    clean_record
                )

                continue

            index = rng.randint(
                0,
                valid_source - 1
            )

            if index < target:

                reservoir[index] = (
                    clean_record
                )

    print()
    print(
        "Valid source documents:",
        valid_source
    )

    print(
        "Excluded held-out documents:",
        excluded
    )

    print(
        "Sampled training documents:",
        len(reservoir)
    )

    return reservoir


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(
    path,
    records
):

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


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaMind Run 4 Data Builder")
    print("================================")

    # --------------------------------
    # Load validation/test hashes
    # --------------------------------

    print()
    print(
        "Loading validation set..."
    )

    validation_hashes = load_hashes(
        VALIDATION_FILE
    )

    print(
        "Validation hashes:",
        len(validation_hashes)
    )

    print()
    print(
        "Loading test set..."
    )

    test_hashes = load_hashes(
        TEST_FILE
    )

    print(
        "Test hashes:",
        len(test_hashes)
    )

    # --------------------------------
    # Combined exclusion set
    # --------------------------------

    excluded_hashes = (
        validation_hashes
        | test_hashes
    )

    print()
    print(
        "Total held-out hashes:",
        len(excluded_hashes)
    )

    # --------------------------------
    # Sample training data
    # --------------------------------

    records = reservoir_sample(
        SOURCE,
        excluded_hashes,
        TARGET_DOCS,
        SEED
    )

    # --------------------------------
    # Save
    # --------------------------------

    save_jsonl(
        OUTPUT,
        records
    )

    print()
    print("================================")
    print("Run 4 Dataset Complete")
    print("================================")

    print()
    print(
        "Output:",
        OUTPUT
    )

    print(
        "Documents:",
        len(records)
    )


if __name__ == "__main__":
    main()