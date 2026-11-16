import os
import re
import sys

# Project root ko Python import path mein add karo
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )

INTEGRATION_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if INTEGRATION_DIR not in sys.path:
    sys.path.insert(
        0,
        INTEGRATION_DIR
    )

import torch

from src.tokenizer import Tokenizer
from src.model.language_model import FourmulaMindLM
from src.training.checkpoint import load_checkpoint

from validator import validate_and_translate


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = (
    "checkpoints/4mulamind_4mulaquery_sft.pt"
)


# ============================================================
# Configuration
# ============================================================

MAX_NEW_TOKENS = 64

CONTEXT_SIZE = 128


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load Model
# ============================================================

def load_model():

    tokenizer = Tokenizer.load(
        TOKENIZER_PATH
    )

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
        ]
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

    return model, tokenizer


# ============================================================
# Generate Model Command
# ============================================================

def generate_command(
    model,
    tokenizer,
    instruction
):

    prompt = (
        "User: "
        + instruction.strip()
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
                -CONTEXT_SIZE:
            ]

            x = torch.tensor(
                [context],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(x)

            next_token = int(
                torch.argmax(
                    logits[
                        0,
                        -1,
                        :
                    ]
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

    output_tokens = generated[
        len(prompt_ids):
    ]

    text = tokenizer.decode(
        output_tokens
    ).strip()

    # --------------------------------------------------------
    # Only use the first line.
    # --------------------------------------------------------

    text = text.split(
        "\n",
        1
    )[0].strip()

    # --------------------------------------------------------
    # Remove accidental prefixes.
    # --------------------------------------------------------

    text = re.sub(
        r"^(assistant|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


# ============================================================
# Dry Run
# ============================================================

def process_query(
    model,
    tokenizer,
    user_query
):

    model_command = generate_command(
        model,
        tokenizer,
        user_query
    )

    validation = (
        validate_and_translate(
            model_command
        )
    )

    return {
        "query": user_query,
        "model_command": model_command,
        "valid": validation["valid"],
        "engine_command": validation[
            "engine_command"
        ],
        "reason": validation["reason"]
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaMind → 4mulaQuery Bridge")
    print("DRY RUN MODE")
    print("================================")

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Device:",
        DEVICE
    )

    model, tokenizer = load_model()

    print()
    print(
        "Model loaded successfully."
    )

    print()
    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            user_query = input(
                "\n4mulaQuery > "
            )

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print()
            break

        user_query = user_query.strip()

        if user_query.lower() == "exit":

            break

        if not user_query:

            continue

        result = process_query(
            model,
            tokenizer,
            user_query
        )

        print()
        print(
            "Model output:"
        )

        print(
            result["model_command"]
        )

        print()
        print(
            "Validated:"
        )

        print(
            result["valid"]
        )

        print()
        print(
            "Engine command:"
        )

        print(
            result["engine_command"]
        )

        if result["reason"]:

            print()
            print(
                "Reason:"
            )

            print(
                result["reason"]
            )

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
