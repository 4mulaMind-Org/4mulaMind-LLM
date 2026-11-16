import json
import re

import torch

from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.training.checkpoint import load_checkpoint


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = (
    "checkpoints/4mulamind_4mulaquery_sft.pt"
)

TEST_FILE = (
    "data/4mulaquery/test.jsonl"
)


# ============================================================
# Config
# ============================================================

MAX_NEW_TOKENS = 64

TEMPERATURE = 1.0

TOP_K = 1


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load JSONL
# ============================================================

def load_examples(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            instruction = record[
                "instruction"
            ].strip()

            output = record[
                "output"
            ].strip()

            if instruction and output:

                examples.append(
                    {
                        "instruction": instruction,
                        "output": output
                    }
                )

    return examples


# ============================================================
# Normalize output
# ============================================================

def normalize_output(text):

    text = text.strip()

    # Remove common conversational prefixes.
    text = re.sub(
        r"^(assistant|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Keep only first generated line.
    text = text.split(
        "\n",
        1
    )[0].strip()

    return text


# ============================================================
# Generation
# ============================================================

def generate_command(
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

    model.eval()

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

            next_logits = logits[
                0,
                -1,
                :
            ]

            # --------------------------------------------
            # Greedy / top-k=1
            # --------------------------------------------

            if TOP_K == 1:

                next_token = int(
                    torch.argmax(
                        next_logits
                    ).item()
                )

            else:

                scaled = (
                    next_logits
                    / TEMPERATURE
                )

                values, indices = (
                    torch.topk(
                        scaled,
                        TOP_K
                    )
                )

                probabilities = (
                    torch.softmax(
                        values,
                        dim=-1
                    )
                )

                selected = (
                    torch.multinomial(
                        probabilities,
                        1
                    )
                )

                next_token = int(
                    indices[selected].item()
                )

            generated.append(
                next_token
            )

            # --------------------------------------------
            # Stop at EOS
            # --------------------------------------------

            if next_token == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(
        generated[len(prompt_ids):]
    )

    return normalize_output(
        generated_text
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaQuery Test Evaluation")
    print("================================")

    print()
    print("Checkpoint:")
    print(CHECKPOINT_PATH)

    print()
    print("Device:")
    print(DEVICE)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.load(
        TOKENIZER_PATH
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    config = {
        "vocab_size": checkpoint[
            "vocab_size"
        ],
        "block_size": checkpoint[
            "block_size"
        ],
        "embedding_dim": checkpoint[
            "embedding_dim"
        ],
        "num_heads": checkpoint[
            "num_heads"
        ],
        "hidden_dim": checkpoint[
            "hidden_dim"
        ],
        "num_layers": checkpoint[
            "num_layers"
        ],
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

    # --------------------------------------------------------
    # Load test set
    # --------------------------------------------------------

    test_examples = load_examples(
        TEST_FILE
    )

    print()
    print(
        "Test examples:",
        len(test_examples)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(
        test_examples
    )

    correct = 0

    command_total = {}
    command_correct = {}

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for index, example in enumerate(
        test_examples,
        start=1
    ):

        instruction = example[
            "instruction"
        ]

        expected = normalize_output(
            example["output"]
        )

        predicted = generate_command(
            model,
            tokenizer,
            instruction
        )

        is_correct = (
            predicted.lower()
            == expected.lower()
        )

        if is_correct:

            correct += 1

        # --------------------------------
        # Command category
        # --------------------------------

        if expected.upper() == "INVALID":

            command = "invalid"

        else:

            command = expected.split(
                " ",
                1
            )[0].lower()

        command_total[command] = (
            command_total.get(
                command,
                0
            )
            + 1
        )

        if is_correct:

            command_correct[command] = (
                command_correct.get(
                    command,
                    0
                )
                + 1
            )

        # --------------------------------
        # Show mistakes
        # --------------------------------

        if (
            not is_correct
            and index <= 30
        ):

            print()
            print(
                f"Example {index}"
            )

            print(
                "Instruction:",
                instruction
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

        # --------------------------------
        # Progress
        # --------------------------------

        if index % 100 == 0:

            print(
                f"Evaluated "
                f"{index}/{total}"
            )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    overall_accuracy = (
        correct / total
        if total
        else 0.0
    )

    print()
    print("================================")
    print("Final Results")
    print("================================")

    print()
    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Overall Accuracy: "
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print("Per-command accuracy:")

    for command in [
        "search",
        "delete",
        "insert",
        "all",
        "invalid"
    ]:

        total_command = (
            command_total.get(
                command,
                0
            )
        )

        correct_command = (
            command_correct.get(
                command,
                0
            )
        )

        accuracy = (
            correct_command
            / total_command
            if total_command
            else 0.0
        )

        print(
            f"{command:8s}: "
            f"{correct_command}/"
            f"{total_command}"
            f" = "
            f"{accuracy * 100:.2f}%"
        )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()
