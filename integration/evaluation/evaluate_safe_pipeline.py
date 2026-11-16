import json

from integration.core.safe_pipeline import (
    load_intent_model,
    process_query,
)


CHALLENGE_FILE = (
    "integration/challenge_set.json"
)


def load_challenges(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def expected_intent(expected):

    expected = expected.strip().lower()

    if expected == "invalid":
        return "INVALID"

    if expected == "all":
        return "ALL"

    if expected.startswith("search "):
        return "SEARCH"

    if expected.startswith("delete "):
        return "DELETE"

    if expected.startswith("insert "):
        return "INSERT"

    return "INVALID"


def expected_engine_command(expected):

    expected = expected.strip()

    if expected.upper() == "INVALID":
        return None

    if expected.lower() == "all":
        return "all"

    if expected.lower().startswith(
        "search id="
    ):

        value = expected[
            len("search id="):
        ]

        return f"search,{value}"

    if expected.lower().startswith(
        "delete id="
    ):

        value = expected[
            len("delete id="):
        ]

        return f"delete,{value}"

    if expected.lower().startswith(
        "insert id="
    ):

        remainder = expected[
            len("insert id="):
        ]

        parts = remainder.split(
            " username=",
            1
        )

        if len(parts) != 2:
            return None

        user_id = parts[0]

        username_and_email = parts[1].split(
            " email=",
            1
        )

        if len(username_and_email) != 2:
            return None

        username = username_and_email[0]
        email = username_and_email[1]

        return (
            f"insert,"
            f"{user_id},"
            f"{username},"
            f"{email}"
        )

    return None


def main():

    print()
    print("================================")
    print("4mulaMind Safe Pipeline Evaluation")
    print("================================")

    model, tokenizer, checkpoint = (
        load_intent_model()
    )

    challenges = load_challenges(
        CHALLENGE_FILE
    )

    print()
    print(
        "Checkpoint step:",
        checkpoint.get("step")
    )

    print(
        "Validation loss:",
        checkpoint.get("best_val_loss")
    )

    print()
    print(
        "Challenges:",
        len(challenges)
    )

    total = len(challenges)

    intent_correct = 0
    executable_correct = 0
    safe_rejections = 0
    unsafe_commands = 0

    for index, item in enumerate(
        challenges,
        start=1
    ):

        query = item["instruction"]

        expected = item["expected"].strip()

        exp_intent = expected_intent(
            expected
        )

        exp_command = (
            expected_engine_command(
                expected
            )
        )

        result = process_query(
            model,
            tokenizer,
            query
        )

        if index == 107:
            print("\n================ DEBUG CASE 107 ================")
            print("QUERY:", repr(query))
            print("RESULT:", result)
            print("EXPECTED:", repr(expected))
            print("EXPECTED INTENT:", repr(exp_intent))
            print("EXPECTED COMMAND:", repr(exp_command))
            print("=================================================\n")

        actual_intent = (
            result["intent"]
            .strip()
            .upper()
        )

        actual_command = (
            result["engine_command"]
        )

        intent_ok = (
            actual_intent
            == exp_intent
        )

        # ----------------------------------------------------
        # Intent metric
        # ----------------------------------------------------

        if intent_ok:
            intent_correct += 1

        # ----------------------------------------------------
        # Execution safety
        # ----------------------------------------------------

        if exp_intent == "INVALID":

            if actual_command is None:

                executable_ok = True
                safe_rejections += 1

            else:

                executable_ok = False
                unsafe_commands += 1

        elif exp_command is None:

            # Invalid/incomplete expected request.
            executable_ok = (
                actual_command is None
            )

            if executable_ok:
                safe_rejections += 1

        else:

            executable_ok = (
                actual_command is not None
                and actual_command.lower()
                == exp_command.lower()
            )

        if executable_ok:
            executable_correct += 1

        status = (
            "OK"
            if executable_ok
            else "WRONG"
        )

        print(
            f"{index:03d} | "
            f"ExpectedIntent={exp_intent:<7} | "
            f"ActualIntent={actual_intent:<7} | "
            f"Command={str(actual_command):<55} | "
            f"{status}"
        )

    print()
    print("================================")
    print("Final Safe Pipeline Metrics")
    print("================================")

    print()

    print(
        f"Intent Accuracy: "
        f"{intent_correct}/{total} = "
        f"{intent_correct / total * 100:.2f}%"
    )

    print(
        f"Safe Command Accuracy: "
        f"{executable_correct}/{total} = "
        f"{executable_correct / total * 100:.2f}%"
    )

    print(
        f"Safe Rejections: "
        f"{safe_rejections}"
    )

    print(
        f"Unsafe Commands: "
        f"{unsafe_commands}"
    )

    print()
    print(
        "Evaluation completed."
    )


if __name__ == "__main__":
    main()
