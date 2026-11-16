import json
import math
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.tokenizer import Tokenizer
from src.model.language_model import FourmulaMindLM


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = "tokenizers/4mulamind_bpe.json"

BASE_CHECKPOINT = (
    "checkpoints/4mulamind_run4.pt"
)

TRAIN_FILE = (
    "data/4mulaquery_v2/train.jsonl"
)

VALIDATION_FILE = (
    "data/4mulaquery_v2/validation.jsonl"
)

OUTPUT_CHECKPOINT = (
    "checkpoints/"
    "4mulamind_4mulaquery_intent_sft.pt"
)


# ============================================================
# Configuration
# ============================================================

MAX_LENGTH = 64

BATCH_SIZE = 8

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 0.01

MAX_STEPS = 3000

WARMUP_STEPS = 200

EVAL_EVERY = 100

PATIENCE = 5

GRAD_CLIP = 1.0


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

def load_jsonl(path):

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

            instruction = (
                record["instruction"].strip()
            )

            output = (
                record["output"].strip()
            )

            if instruction and output:

                examples.append(
                    {
                        "instruction": instruction,
                        "output": output
                    }
                )

    return examples


# ============================================================
# Intent Dataset
# ============================================================

class IntentDataset(Dataset):

    def __init__(
        self,
        examples,
        tokenizer,
        max_length
    ):

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        item = self.examples[index]

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = (
            "User: "
            + item["instruction"]
            + "\nAssistant: "
        )

        prompt_ids = self.tokenizer.encode(
            prompt
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        output_ids = self.tokenizer.encode(
            item["output"],
            add_eos=True
        )

        # ----------------------------------------------------
        # Protect response
        # ----------------------------------------------------

        if len(output_ids) >= self.max_length:

            output_ids = output_ids[
                : self.max_length - 1
            ]

            output_ids.append(
                self.tokenizer.eos_token_id
            )

        # ----------------------------------------------------
        # Prompt truncation if required
        # ----------------------------------------------------

        available_for_prompt = (
            self.max_length
            - len(output_ids)
        )

        if available_for_prompt <= 0:

            prompt_ids = []

        elif len(prompt_ids) > available_for_prompt:

            prompt_ids = prompt_ids[
                -available_for_prompt:
            ]

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        input_ids = (
            prompt_ids
            + output_ids
        )

        # -100 means:
        # do not include this token in loss.
        labels = (
            [-100] * len(prompt_ids)
            + output_ids
        )

        # ----------------------------------------------------
        # Shift for next-token prediction
        # ----------------------------------------------------

        x = input_ids[:-1]

        y = labels[1:]

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padding = (
            self.max_length
            - 1
            - len(x)
        )

        if padding > 0:

            x.extend(
                [self.tokenizer.pad_token_id]
                * padding
            )

            y.extend(
                [-100] * padding
            )

        else:

            x = x[: self.max_length - 1]

            y = y[: self.max_length - 1]

        return (
            torch.tensor(
                x,
                dtype=torch.long
            ),
            torch.tensor(
                y,
                dtype=torch.long
            )
        )


# ============================================================
# Load Base Model
# ============================================================

def load_base_model(tokenizer):

    checkpoint = torch.load(
        BASE_CHECKPOINT,
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

    if (
        config["vocab_size"]
        != tokenizer.vocab_size
    ):

        raise RuntimeError(
            "Tokenizer vocabulary does not "
            "match checkpoint vocabulary."
        )

    model = FourmulaMindLM(
        **config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return (
        model,
        checkpoint,
        config
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            y.reshape(-1),
            ignore_index=-100,
            reduction="sum"
        )

        valid_tokens = (
            y != -100
        ).sum().item()

        total_loss += loss.item()

        total_tokens += valid_tokens

    if total_tokens == 0:

        return float("inf")

    return (
        total_loss
        / total_tokens
    )


# ============================================================
# Learning Rate
# ============================================================

def get_learning_rate(step):

    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (
                (step + 1)
                / WARMUP_STEPS
            )
        )

    progress = (
        step - WARMUP_STEPS
    ) / max(
        1,
        MAX_STEPS - WARMUP_STEPS
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    cosine = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )

    min_lr = (
        LEARNING_RATE * 0.1
    )

    return (
        min_lr
        + (
            LEARNING_RATE
            - min_lr
        )
        * cosine
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaQuery Intent SFT")
    print("================================")

    print()
    print("Device:")
    print(DEVICE)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = Tokenizer.load(
        TOKENIZER_PATH
    )

    print()
    print("Vocabulary:")
    print(tokenizer.vocab_size)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_examples = load_jsonl(
        TRAIN_FILE
    )

    validation_examples = load_jsonl(
        VALIDATION_FILE
    )

    print()
    print(
        "Training examples:",
        len(train_examples)
    )

    print(
        "Validation examples:",
        len(validation_examples)
    )

    train_dataset = IntentDataset(
        train_examples,
        tokenizer,
        MAX_LENGTH
    )

    validation_dataset = IntentDataset(
        validation_examples,
        tokenizer,
        MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    (
        model,
        base_checkpoint,
        config
    ) = load_base_model(
        tokenizer
    )

    model = model.to(
        DEVICE
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        "Base checkpoint:",
        BASE_CHECKPOINT
    )

    print(
        "Base step:",
        base_checkpoint.get("step")
    )

    print(
        "Base validation loss:",
        base_checkpoint.get(
            "best_val_loss"
        )
    )

    print()
    print("Model configuration:")

    for key, value in config.items():

        print(
            f"{key}: {value}"
        )

    print()
    print(
        "Parameters:",
        parameter_count
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    patience_counter = 0

    step = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("================================")
    print("Intent SFT Training")
    print("================================")

    while step < MAX_STEPS:

        for x, y in train_loader:

            if step >= MAX_STEPS:
                break

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            lr = get_learning_rate(
                step
            )

            for group in (
                optimizer.param_groups
            ):

                group["lr"] = lr

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                y.reshape(-1),
                ignore_index=-100
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

            step += 1

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if step % EVAL_EVERY == 0:

                val_loss = evaluate(
                    model,
                    validation_loader
                )

                print(
                    f"Step {step:5d} | "
                    f"Train Loss: "
                    f"{loss.item():.4f} | "
                    f"Val Loss: "
                    f"{val_loss:.4f} | "
                    f"LR: "
                    f"{lr:.6f}"
                )

                # --------------------------------------------
                # Best checkpoint
                # --------------------------------------------

                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    patience_counter = 0

                    checkpoint = {
                        "model_state_dict":
                            model.state_dict(),

                        "best_val_loss":
                            best_val_loss,

                        "step":
                            step,

                        "base_checkpoint":
                            BASE_CHECKPOINT,

                        **config
                    }

                    torch.save(
                        checkpoint,
                        OUTPUT_CHECKPOINT
                    )

                    print(
                        "✓ Best intent SFT model saved | "
                        f"Val Loss: "
                        f"{val_loss:.4f}"
                    )

                else:

                    patience_counter += 1

                # --------------------------------------------
                # Early stopping
                # --------------------------------------------

                if (
                    patience_counter
                    >= PATIENCE
                ):

                    print()
                    print(
                        "================================"
                    )

                    print(
                        "Early stopping."
                    )

                    print(
                        f"No improvement for "
                        f"{PATIENCE} evaluations."
                    )

                    print()
                    print(
                        "Best validation loss:",
                        best_val_loss
                    )

                    print(
                        "Checkpoint:",
                        OUTPUT_CHECKPOINT
                    )

                    return

    print()
    print("================================")
    print("Intent SFT Completed")
    print("================================")

    print()
    print(
        "Best validation loss:",
        best_val_loss
    )

    print()
    print(
        "Checkpoint:",
        OUTPUT_CHECKPOINT
    )


if __name__ == "__main__":
    main()
