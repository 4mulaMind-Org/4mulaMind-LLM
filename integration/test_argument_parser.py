from argument_parser import build_safe_command


TESTS = [
    (
        "Find the user with id 42",
        "SEARCH"
    ),
    (
        "Who is user number 317?",
        "SEARCH"
    ),
    (
        "Please remove account 25",
        "DELETE"
    ),
    (
        "bhai user 91 hata do",
        "DELETE"
    ),
    (
        "Show all users",
        "ALL"
    ),
    (
        "saare users dikhao",
        "ALL"
    ),
    (
        "Insert user with id 9000, username abdul, and email abdul@gmail.com",
        "INSERT"
    ),
    (
        "Create a user with id 9012, username ali, and email ali@example.com",
        "INSERT"
    ),
    (
        "Create a new user",
        "INVALID"
    ),
    (
        "Delete a user",
        "INVALID"
    ),
]


def main():

    print()
    print("================================")
    print("4mulaQuery Argument Parser Test")
    print("================================")

    for query, intent in TESTS:

        result = build_safe_command(
            query,
            intent
        )

        print()
        print("Query:")
        print(query)

        print()
        print("Intent:")
        print(intent)

        print()
        print("Arguments:")
        print(result["arguments"])

        print()
        print("Valid:")
        print(result["valid"])

        print()
        print("Engine command:")
        print(result["engine_command"])

        if result["reason"]:
            print()
            print("Reason:")
            print(result["reason"])

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
