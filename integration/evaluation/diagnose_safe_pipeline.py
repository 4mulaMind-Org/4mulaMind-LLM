from integration.core.safe_pipeline import (
    load_intent_model,
    process_query,
)

from integration.core.intent_router import (
    rule_based_intent,
)

from integration.core.argument_parser import (
    extract_id,
    extract_username,
    extract_email,
    extract_insert_arguments,
    parse_arguments,
)


CASES = [
    "Add a new account: id 2345, username emma, email emma@gmail.com",
    "Save user john with id 3456 and email john@hotmail.com",
    "Insert a record for sam, id 6781, email sam@mail.com",
    "42 wala user remove karo",
    "Could you find user 654 for me?",
]


def main():

    model, tokenizer, checkpoint = (
        load_intent_model()
    )

    print()
    print("================================")
    print("4mulaMind Safe Pipeline Diagnosis")
    print("================================")

    print()
    print(
        "Checkpoint step:",
        checkpoint.get("step")
    )

    for index, query in enumerate(
        CASES,
        start=1
    ):

        print()
        print("--------------------------------")
        print(f"CASE {index}")
        print("--------------------------------")

        print()
        print("QUERY:")
        print(query)

        # ----------------------------------------------------
        # Rule based intent
        # ----------------------------------------------------

        rule_intent = rule_based_intent(
            query
        )

        print()
        print("RULE INTENT:")
        print(repr(rule_intent))

        # ----------------------------------------------------
        # Individual extractors
        # ----------------------------------------------------

        print()
        print("EXTRACTED ID:")
        print(repr(extract_id(query)))

        print()
        print("EXTRACTED USERNAME:")
        print(repr(extract_username(query)))

        print()
        print("EXTRACTED EMAIL:")
        print(repr(extract_email(query)))

        print()
        print("INSERT ARGUMENTS:")
        print(
            repr(
                extract_insert_arguments(
                    query
                )
            )
        )

        # ----------------------------------------------------
        # Full pipeline
        # ----------------------------------------------------

        result = process_query(
            model,
            tokenizer,
            query
        )

        print()
        print("FINAL PIPELINE RESULT:")
        print(result)


if __name__ == "__main__":
    main()