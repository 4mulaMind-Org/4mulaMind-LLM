import json
import os


SOURCE_FILES = {
    "train": "data/4mulaquery/train.jsonl",
    "validation": "data/4mulaquery/validation.jsonl",
    "test": "data/4mulaquery/test.jsonl",
}

OUTPUT_DIR = "data/4mulaquery_v2"


def intent_from_output(output):
    output = output.strip().lower()

    if output == "invalid":
        return "INVALID"

    if output.startswith("search"):
        return "SEARCH"

    if output.startswith("delete"):
        return "DELETE"

    if output.startswith("insert"):
        return "INSERT"

    if output == "all":
        return "ALL"

    raise ValueError(
        f"Unknown output: {output}"
    )


def convert_file(input_path, output_path):

    examples = []
    seen = set()

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            instruction = item["instruction"].strip()
            output = item["output"].strip()

            intent = intent_from_output(
                output
            )

            key = (
                instruction.lower(),
                intent
            )

            if key in seen:
                continue

            seen.add(key)

            examples.append(
                {
                    "instruction": instruction,
                    "output": intent
                }
            )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        for item in examples:

            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    return examples


def statistics(name, examples):

    counts = {}

    for item in examples:

        intent = item["output"]

        counts[intent] = (
            counts.get(intent, 0) + 1
        )

    print()
    print(
        f"{name}: {len(examples)}"
    )

    for intent in [
        "SEARCH",
        "DELETE",
        "INSERT",
        "ALL",
        "INVALID"
    ]:

        print(
            f"  {intent}: "
            f"{counts.get(intent, 0)}"
        )


def main():

    print()
    print("================================")
    print("4mulaQuery SFT v2")
    print("Intent Dataset Builder")
    print("================================")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for split, source in SOURCE_FILES.items():

        output = (
            f"{OUTPUT_DIR}/{split}.jsonl"
        )

        examples = convert_file(
            source,
            output
        )

        statistics(
            split.upper(),
            examples
        )

        print(
            "Output:",
            output
        )

    print()
    print("Dataset creation complete.")


if __name__ == "__main__":
    main()
