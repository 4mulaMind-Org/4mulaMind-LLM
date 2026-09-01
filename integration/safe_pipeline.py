import re
import torch

from integration.intent_router import rule_based_intent
from src.tokenizer import Tokenizer
from src.language_model import FourmulaMindLM
from src.checkpoint import load_checkpoint

from integration.argument_parser import build_safe_command

# ============================================================
# Paths
# ============================================================

TOKENIZER_PATH = (
    "tokenizers/4mulamind_bpe.json"
)

CHECKPOINT_PATH = (
    "checkpoints/"
    "4mulamind_4mulaquery_intent_sft.pt"
)


# ============================================================
# Configuration
# ============================================================

MAX_NEW_TOKENS = 16
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
# Model Loading
# ============================================================

def load_intent_model():

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

    return model, tokenizer, checkpoint


# ============================================================
# Normalize intent output
# ============================================================

def normalize_intent(text):

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


# ============================================================
# Predict intent
# ============================================================

def predict_intent(model, tokenizer, query):

    prompt = (
        "User: "
        + query.strip()
        + "\nAssistant: "
    )

    prompt_ids = tokenizer.encode(prompt)

    intents = [
        "SEARCH",
        "INSERT",
        "DELETE",
        "ALL",
        "INVALID",
    ]

    scores = {}

    with torch.no_grad():

        for intent in intents:

            intent_ids = tokenizer.encode(intent)

            generated_ids = prompt_ids.copy()

            score = 0.0

            for token_id in intent_ids:

                context = generated_ids[-CONTEXT_SIZE:]

                x = torch.tensor(
                    [context],
                    dtype=torch.long,
                    device=DEVICE
                )

                logits = model(x)

                log_probs = torch.log_softmax(
                    logits[0, -1],
                    dim=-1
                )

                score += log_probs[token_id].item()

                generated_ids.append(token_id)

            scores[intent] = score

    result = max(
        scores,
        key=scores.get
    )

    print()
    print("DEBUG INTENT SCORES:")

    for intent, score in scores.items():
        print(f"{intent:8} : {score:.4f}")

    print("DEBUG SELECTED INTENT:")
    print(result)

    return result
# ============================================================
# Full Safe Pipeline
# ============================================================

def process_query(
    model,
    tokenizer,
    query
):

    # --------------------------------------------------------
    # Deterministic high-confidence routing
    # --------------------------------------------------------

    intent = rule_based_intent(
        query
    )

    # --------------------------------------------------------
    # Model fallback
    #
    # If rules cannot confidently classify the query,
    # ask the trained Intent SFT model.
    # --------------------------------------------------------

    if intent is None:

        intent = predict_intent(
            model,
            tokenizer,
            query
        )

        # ----------------------------------------------------
        # Strict intent allowlist
        #
        # Never allow arbitrary model output to reach
        # the argument parser / executor.
        # ----------------------------------------------------

        allowed_intents = {
            "SEARCH",
            "INSERT",
            "DELETE",
            "ALL",
        }

        if intent not in allowed_intents:

            return {
                "query": query,
                "intent": "INVALID",
                "arguments": None,
                "valid": False,
                "engine_command": None,
                "reason": (
                    "Intent model produced an "
                    "unsupported intent."
                ),
            }

    # --------------------------------------------------------
    # Parse + validate
    # --------------------------------------------------------

    result = build_safe_command(
        query,
        intent
    )

    return {
        "query": query,
        "intent": intent,
        "arguments": result["arguments"],
        "valid": result["valid"],
        "engine_command": result[
            "engine_command"
        ],
        "reason": result["reason"],
    }
# ============================================================
# Interactive mode
# ============================================================

def main():

    print()
    print("================================")
    print("4mulaMind Safe 4mulaQuery Pipeline")
    print("DRY RUN ONLY")
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

    model, tokenizer, checkpoint = (
        load_intent_model()
    )

    print()
    print(
        "Checkpoint step:",
        checkpoint.get("step")
    )

    print(
        "Validation loss:",
        checkpoint.get(
            "best_val_loss"
        )
    )

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

            query = input(
                "\n4mulaQuery > "
            )

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print()
            break

        query = query.strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        result = process_query(
            model,
            tokenizer,
            query
        )

        print()
        print(
            "Predicted intent:"
        )

        print(
            result["intent"]
        )

        print()
        print(
            "Extracted arguments:"
        )

        print(
            result["arguments"]
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
