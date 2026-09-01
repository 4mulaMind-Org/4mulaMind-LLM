import json

from integration.bridge import load_model, generate_command
from integration.validator import validate_and_translate


CHALLENGE_FILE = "integration/challenge_set.json"


def load_challenges(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print()
    print("================================")
    print("4mulaQuery Challenge Evaluation")
    print("================================")

    model, tokenizer = load_model()

    challenges = load_challenges(
        CHALLENGE_FILE
    )

    total = len(challenges)
    exact_correct = 0
    validated_correct = 0
    unsafe = 0

    print()
    print("Challenges:", total)

    for index, item in enumerate(
        challenges,
        start=1
    ):

        instruction = item["instruction"]
        expected = item["expected"]

        predicted = generate_command(
            model,
            tokenizer,
            instruction
        )

        # Exact model-output match
        exact_match = (
            predicted.strip().lower()
            == expected.strip().lower()
        )

        if exact_match:
            exact_correct += 1

        # Validate predicted command
        result = validate_and_translate(
            predicted
        )

        # A prediction is considered safely usable
        # only when validator accepts it.
        is_unsafe = False

        if not result["valid"]:
            is_unsafe = True

        # Semantic correctness through expected command
        if exact_match and result["valid"]:
            validated_correct += 1

        if is_unsafe:
            unsafe += 1

        print(
            f"{index:03d} | "
            f"Expected: {expected} | "
            f"Predicted: {predicted} | "
            f"{'OK' if exact_match else 'WRONG'}"
        )

    print()
    print("================================")
    print("Challenge Results")
    print("================================")

    print()
    print(
        f"Exact Accuracy: "
        f"{exact_correct}/{total} = "
        f"{(exact_correct / total * 100):.2f}%"
    )

    print(
        f"Validated Correct: "
        f"{validated_correct}/{total} = "
        f"{(validated_correct / total * 100):.2f}%"
    )

    print(
        f"Unsafe Predictions: "
        f"{unsafe}/{total}"
    )

    print()
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
