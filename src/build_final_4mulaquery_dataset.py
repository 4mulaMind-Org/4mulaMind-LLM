import json
import os
import random


INPUT_FILE = "data/4mulaquery_sft_raw.jsonl"

OUTPUT_DIR = "data/4mulaquery"

TRAIN_FILE = f"{OUTPUT_DIR}/train.jsonl"
VALIDATION_FILE = f"{OUTPUT_DIR}/validation.jsonl"
TEST_FILE = f"{OUTPUT_DIR}/test.jsonl"

SEED = 42


TARGETS = {
    "search": 2500,
    "delete": 2500,
    "insert": 2000,
    "all": 1500,
    "invalid": 1500,
}


# ============================================================
# Helpers
# ============================================================

def add_example(
    store,
    instruction,
    output
):
    store.append(
        {
            "instruction": instruction.strip(),
            "output": output.strip()
        }
    )


def command_of(example):

    output = example["output"].strip().lower()

    if output == "invalid":
        return "invalid"

    return output.split(" ", 1)[0]


def deduplicate(examples):

    seen = set()
    result = []

    for example in examples:

        key = (
            example["instruction"].lower(),
            example["output"].lower()
        )

        if key not in seen:

            seen.add(key)
            result.append(example)

    return result


def split_group(
    examples,
    rng
):

    examples = examples.copy()

    rng.shuffle(
        examples
    )

    n = len(examples)

    train_end = int(
        n * 0.80
    )

    val_end = int(
        n * 0.90
    )

    return (
        examples[:train_end],
        examples[train_end:val_end],
        examples[val_end:]
    )


def save_jsonl(
    path,
    examples
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# Generate INSERT
# ============================================================

def generate_insert():

    examples = []

    templates = [

        "Insert user with id {id}, username {username}, and email {email}",
        "Create a user with id {id}, username {username}, and email {email}",
        "Add a new user with id {id}, username {username}, and email {email}",
        "Create user {username} with id {id} and email {email}",
        "Add {username} to the database with id {id} and email {email}",
        "Register user {username} with id {id} and email {email}",
        "Store user {username} with id {id} and email {email}",
        "Please create user {username}, id {id}, email {email}",
        "Please insert user {username}, id {id}, email {email}",
        "Add this user: id {id}, username {username}, email {email}",
        "Create this account: id {id}, username {username}, email {email}",
        "Save a new user with id {id}, username {username}, email {email}",
    ]

    first_names = [
        "abdul",
        "ali",
        "ahmed",
        "rahul",
        "rohit",
        "amit",
        "arjun",
        "vikas",
        "raj",
        "karan",
        "sarah",
        "john",
        "alex",
        "david",
        "maria",
        "emma",
        "james",
        "jack",
        "daniel",
        "olivia",
        "michael",
        "chris",
        "sam",
        "tom",
        "anna",
        "ravi",
        "neha",
        "aman",
        "sumit",
        "priya",
        "nisha",
        "farhan",
    ]

    domains = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "example.com",
        "mail.com",
        "proton.me",
        "icloud.com",
        "hotmail.com",
    ]

    counter = 1

    for username_base in first_names:

        for variant in range(1, 101):

            username = (
                f"{username_base}{variant}"
            )

            user_id = (
                1000 + counter
            )

            domain = domains[
                counter % len(domains)
            ]

            email = (
                f"{username}@{domain}"
            )

            template = templates[
                counter % len(templates)
            ]

            add_example(
                examples,
                template.format(
                    id=user_id,
                    username=username,
                    email=email
                ),
                (
                    f"insert id={user_id} "
                    f"username={username} "
                    f"email={email}"
                )
            )

            counter += 1

            if len(examples) >= 3000:
                return examples

    return examples


# ============================================================
# Generate ALL
# ============================================================

def generate_all():

    examples = []

    verbs = [
        "Show",
        "List",
        "Get",
        "Display",
        "Fetch",
        "Return",
        "Give me",
        "Provide",
        "Retrieve",
        "Print",
        "Load",
        "View",
        "Present",
        "Show me",
        "I want",
        "I need",
        "Can you show",
        "Can you list",
        "Can you get",
        "Please show",
    ]

    objects = [
        "all users",
        "every user",
        "all user records",
        "every user record",
        "the complete user list",
        "the full user database",
        "the entire user table",
        "all records",
        "every record",
        "the complete list of users",
        "the whole user list",
        "the complete user table",
        "all accounts",
        "every account",
        "all user accounts",
        "the complete set of users",
        "the complete set of records",
        "all entries",
        "every entry",
        "all user entries",
    ]

    suffixes = [
        "",
        " please",
        " for me",
        " from the database",
        " from the user table",
        " in the system",
        " right now",
        " today",
        " for this request",
        " from storage",
        " in the database",
        " in the user table",
        " available in the system",
        " currently stored",
        " currently available",
    ]

    examples_seen = set()

    for verb in verbs:

        for obj in objects:

            for suffix in suffixes:

                instruction = (
                    f"{verb} {obj}{suffix}"
                ).strip()

                key = instruction.lower()

                if key in examples_seen:
                    continue

                examples_seen.add(key)

                add_example(
                    examples,
                    instruction,
                    "all"
                )

                if len(examples) >= 2500:
                    return examples

    return examples


# ============================================================
# Generate INVALID
# ============================================================

def generate_invalid():

    examples = []

    base_requests = [

        "Create a new user",
        "Add a user",
        "Delete a user",
        "Search for a user",
        "Find someone",
        "Remove the account",
        "Create user Abdul",
        "Add Abdul to the database",
        "Delete whoever is logged in",
        "Show me the database",
        "Do something with user 10",
        "Find the user",
        "Remove someone",
        "Add a new account",
        "Create an account",
        "Delete an account",
        "Search the database",
        "Get someone",
        "Update the user",
        "Change user details",
        "Insert a user",
        "Delete by username",
        "Search by email",
        "Find by username",
        "Get a user",
        "Remove a record",
        "Create an account for Abdul",
        "Delete the account",
        "Search for someone",
        "Make a user",
    ]

    prefixes = [
        "",
        "Please ",
        "Can you ",
        "Could you ",
        "I want you to ",
        "I need you to ",
        "Would you ",
        "Kindly ",
        "Please help me ",
        "I would like you to ",
    ]

    suffixes = [
        "",
        " please",
        " for me",
        " from the database",
        " in the system",
        " right now",
        " today",
        " somehow",
        " as soon as possible",
        " if possible",
    ]

    counter = 0

    for prefix in prefixes:

        for request in base_requests:

            for suffix in suffixes:

                instruction = (
                    prefix
                    + request[0].lower()
                    + request[1:]
                    + suffix
                )

                add_example(
                    examples,
                    instruction,
                    "INVALID"
                )

                counter += 1

                if counter >= 2500:
                    return examples

    return examples


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("================================")
    print("Final 4mulaQuery Dataset")
    print("================================")

    # --------------------------------
    # Load raw examples
    # --------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw = []

        for line in file:

            if line.strip():

                raw.append(
                    json.loads(line)
                )

    raw = deduplicate(raw)

    print()
    print(
        "Raw examples:",
        len(raw)
    )

    # --------------------------------
    # Group raw examples
    # --------------------------------

    groups = {
        "search": [],
        "delete": [],
        "insert": [],
        "all": [],
        "invalid": [],
    }

    for example in raw:

        command = command_of(
            example
        )

        if command in groups:

            groups[command].append(
                example
            )

    # --------------------------------
    # Add generated examples
    # --------------------------------

    groups["insert"].extend(
        generate_insert()
    )

    groups["all"].extend(
        generate_all()
    )

    groups["invalid"].extend(
        generate_invalid()
    )

    # --------------------------------
    # Deduplicate groups
    # --------------------------------

    for command in groups:

        groups[command] = deduplicate(
            groups[command]
        )

    print()
    print("Available after generation:")

    for command in [
        "search",
        "delete",
        "insert",
        "all",
        "invalid",
    ]:

        print(
            f"  {command}: "
            f"{len(groups[command])}"
        )

    # --------------------------------
    # Verify enough examples
    # --------------------------------

    for command, target in TARGETS.items():

        available = len(
            groups[command]
        )

        if available < target:

            raise RuntimeError(
                f"Not enough examples for "
                f"{command}: "
                f"need {target}, "
                f"have {available}"
            )

    # --------------------------------
    # Exact balancing
    # --------------------------------

    rng = random.Random(
        SEED
    )

    balanced = []

    for command, target in TARGETS.items():

        candidates = groups[
            command
        ].copy()

        rng.shuffle(
            candidates
        )

        selected = candidates[
            :target
        ]

        balanced.extend(
            selected
        )

        print(
            f"{command}: "
            f"{len(selected)} selected"
        )

    # --------------------------------
    # Shuffle
    # --------------------------------

    rng.shuffle(
        balanced
    )

    print()
    print(
        "Total balanced examples:",
        len(balanced)
    )

    # --------------------------------
    # Split separately by category
    # --------------------------------

    train = []
    validation = []
    test = []

    for command in TARGETS:

        command_examples = [
            example
            for example in balanced
            if command_of(example)
            == command
        ]

        (
            train_group,
            validation_group,
            test_group
        ) = split_group(
            command_examples,
            rng
        )

        train.extend(
            train_group
        )

        validation.extend(
            validation_group
        )

        test.extend(
            test_group
        )

    # --------------------------------
    # Final shuffle
    # --------------------------------

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    # --------------------------------
    # Create directory
    # --------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------
    # Save
    # --------------------------------

    save_jsonl(
        TRAIN_FILE,
        train
    )

    save_jsonl(
        VALIDATION_FILE,
        validation
    )

    save_jsonl(
        TEST_FILE,
        test
    )

    # --------------------------------
    # Statistics
    # --------------------------------

    print()
    print("================================")
    print("Final Split")
    print("================================")

    for name, dataset in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:

        counts = {}

        for example in dataset:

            command = command_of(
                example
            )

            counts[command] = (
                counts.get(
                    command,
                    0
                )
                + 1
            )

        print()
        print(
            f"{name}: {len(dataset)}"
        )

        for command in [
            "search",
            "delete",
            "insert",
            "all",
            "invalid",
        ]:

            print(
                f"  {command}: "
                f"{counts.get(command, 0)}"
            )

    print()
    print("================================")
    print("Dataset Creation Complete")
    print("================================")


if __name__ == "__main__":
    main()
