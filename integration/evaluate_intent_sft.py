import json
import re

import torch

from src.tokenizer import Tokenizer
from src.model.language_model import FourmulaMindLM
from src.checkpoint import load_checkpoint


TOKENIZER_PATH = "tokenizers/4mulamind_bpe.json"

CHECKPOINT_PATH = (
    "checkpoints/4mulamind_4mulaquery_intent_sft.pt"
)

TEST_FILE = (
    "data/4mulaquery_v2/test.jsonl"
)

MAX_NEW_TOKENS = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def load_examples(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            item = json.loads(line)

            examples.append(
                {
                    "instruction":
                        item["instruction"].strip(),

                    "expected":
                        item["output"].strip().upper()
                }
            )

    return examples


def normalize(text):

    text = text.strip()

    text = text.split(
        "\n",
        1
    )[0].strip()

    text = re.sub(
        r"^(assistant|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip().upper()


def predict(
    model,
    tokenizer,
    instruction
):

    prompt = (
        "User: "
        + instruction
        + "\nAssistant: "
    )

    prompt_ids = tokenizer.encode(
        prompt
    )

    generated = prompt_ids.copy()

    with torch.no_grad():

        for _ in range(
            MAX_NEW_TOKENS
        ):

            context = generated[
                -128:
            ]

            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(x)

            next_token = int(
                torch.argmax(
                    logits[0, -1]
                ).item()
            )

            generated.append(
                next_token
            )

            if (
                next_token
                == tokenizer.eos_token_id
            ):
                break

    output = tokenizer.decode(
        generated[len(prompt_ids):]
    )

    return normalize(output)


def main():

    print()
    print("================================")
    print("4mulaQuery Intent SFT Evaluation")
    print("================================")

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print()
    print("Device:")
    print(DEVICE)

    tokenizer = Tokenizer.load(
        TOKENIZER_PATH
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    config = {
        "vocab_size":
            checkpoint["vocab_size"],
        "block_size":
            checkpoint["block_size"],
        "embedding_dim":
            checkpoint["embedding_dim"],
        "num_heads":
            checkpoint["num_heads"],
        "hidden_dim":
            checkpoint["hidden_dim"],
        "num_layers":
            checkpoint["num_layers"],
    }

    model = FourmulaMindLM(
        **config
    )

    model = load_checkpoint(
        model,
        CHECKPOINT_PATH
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    examples = load_examples(
        TEST_FILE
    )

    print()
    print(
        "Test examples:",
        len(examples)
    )

    correct = 0

    total = len(examples)

    by_intent = {}

    for index, example in enumerate(
        examples,
        start=1
    ):

        predicted = predict(
            model,
            tokenizer,
            example["instruction"]
        )

        expected = example["expected"]

        intent_result = {
            "expected": expected,
            "correct": 0,
        }

        if predicted == expected:

            correct += 1

            intent_result["correct"] = 1

        by_intent.setdefault(
            expected,
            {
                "total": 0,
                "correct": 0
            }
        )

        by_intent[
            expected
        ]["total"] += 1

        by_intent[
            expected
        ]["correct"] += (
            intent_result["correct"]
        )

        if predicted != expected:

            print()
            print(
                f"WRONG {index}"
            )

            print(
                "Instruction:",
                example["instruction"]
            )

            print(
                "Expected:",
                expected
            )

            print(
                "Predicted:",
                predicted
            )

            print(
                "--------------------------------"
            )

        if index % 100 == 0:

            print(
                f"Evaluated "
                f"{index}/{total}"
            )

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print()
    print("================================")
    print("Final Intent Results")
    print("================================")

    print()
    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Overall Intent Accuracy: "
        f"{accuracy:.2f}%"
    )

    print()
    print("Per-intent accuracy:")

    for intent in [
        "SEARCH",
        "DELETE",
        "INSERT",
        "ALL",
        "INVALID"
    ]:

        stats = by_intent.get(
            intent,
            {
                "total": 0,
                "correct": 0
            }
        )

        intent_accuracy = (
            stats["correct"]
            / stats["total"]
            * 100
            if stats["total"]
            else 0
        )

        print(
            f"{intent:8s}: "
            f"{stats['correct']}/"
            f"{stats['total']} = "
            f"{intent_accuracy:.2f}%"
        )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()
