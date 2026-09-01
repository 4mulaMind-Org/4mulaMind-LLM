import sys
from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from integration.safe_pipeline import (
    load_intent_model,
    process_query,
)

from integration.executor import (
    execute_engine,
)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("============================================")
    print("4mulaMind → Safe Pipeline → 4mulaQuery")
    print("============================================")

    print()
    print("Mode: ISOLATED RUNTIME")
    print("Core database: NOT USED")

    print()

    model, tokenizer, checkpoint = (
        load_intent_model()
    )

    print("Intent model loaded successfully.")

    print(
        "Checkpoint step:",
        checkpoint.get("step")
    )

    print(
        "Validation loss:",
        checkpoint.get("best_val_loss")
    )

    print()
    print("Type 'exit' to quit.")

    while True:

        try:

            query = input(
                "\n4mulaMind > "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print()
            break

        if query.lower() == "exit":
            break

        if not query:
            continue

        # ----------------------------------------------------
        # Safe pipeline
        # ----------------------------------------------------

        result = process_query(
            model,
            tokenizer,
            query
        )

        print()
        print("Intent:")
        print(result["intent"])

        print()
        print("Arguments:")
        print(result["arguments"])

        print()
        print("Validated:")
        print(result["valid"])

        print()
        print("Engine command:")
        print(result["engine_command"])

        if result["reason"]:

            print()
            print("Reason:")
            print(result["reason"])

        # ----------------------------------------------------
        # NEVER execute invalid commands
        # ----------------------------------------------------

        if not result["valid"]:

            print()
            print(
                "Execution: BLOCKED"
            )

            print()
            print("--------------------------------")

            continue

        command = result[
            "engine_command"
        ]

        # ----------------------------------------------------
        # Final executor
        # ----------------------------------------------------

        execution = execute_engine(
            command
        )

        print()
        print("Execution:")
        print(
            execution["success"]
        )

        print()
        print("Engine output:")
        print(
            execution["output"]
        )

        if execution["error"]:

            print()
            print("Executor error:")
            print(
                execution["error"]
            )

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()