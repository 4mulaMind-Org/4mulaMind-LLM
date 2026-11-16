from validator import validate_and_translate


TESTS = [
    "search id=42",
    "delete id=42",
    "insert id=42 username=abdul email=abdul@gmail.com",
    "all",
    "INVALID",
    "search username=abdul",
    "delete email=abdul@gmail.com",
    "random garbage",
]


def main():

    print()
    print("================================")
    print("4mulaQuery Validator Test")
    print("================================")

    for command in TESTS:

        result = validate_and_translate(
            command
        )

        print()
        print("Model:")
        print(command)

        print()
        print("Valid:")
        print(result["valid"])

        print()
        print("Engine:")
        print(result["engine_command"])

        print()
        print("Reason:")
        print(result["reason"])

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
