import json
import random


OUTPUT_FILE = "data/4mulaquery_sft.jsonl"

SEED = 42

examples = []


def add(instruction, output):
    examples.append(
        {
            "instruction": instruction,
            "output": output
        }
    )


# ============================================================
# SEARCH - ID
# ============================================================

id_patterns = [
    "Find the user with id {id}",
    "Find user {id}",
    "Show user number {id}",
    "Show me user {id}",
    "Get user {id}",
    "Get the user whose ID is {id}",
    "Search for user {id}",
    "Search user with id {id}",
    "Look up user {id}",
    "I need the record for user {id}",
    "Can you find user {id}?",
    "Who is user {id}?",
    "Fetch the user with ID {id}",
    "Retrieve user {id}",
]

for user_id in range(1, 501):

    for pattern in id_patterns:

        add(
            pattern.format(id=user_id),
            f"search id={user_id}"
        )


# ============================================================
# DELETE - ID
# ============================================================

delete_id_patterns = [
    "Delete user {id}",
    "Delete the user with id {id}",
    "Remove user {id}",
    "Remove the user with ID {id}",
    "Delete account {id}",
    "Remove account {id}",
    "Erase user {id}",
    "Get rid of user {id}",
    "Delete record {id}",
]

for user_id in range(1, 301):

    for pattern in delete_id_patterns:

        add(
            pattern.format(id=user_id),
            f"delete id={user_id}"
        )


# ============================================================
# USERNAMES
# ============================================================

usernames = [
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
]


# ============================================================
# SEARCH - USERNAME
# ============================================================

search_username_patterns = [
    "Find the user named {username}",
    "Find user {username}",
    "Search for username {username}",
    "Search user {username}",
    "Get user {username}",
    "Show user {username}",
    "Show me {username}",
    "Look up {username}",
    "Find {username}'s account",
    "Find the account for {username}",
    "Who is {username}?",
    "Retrieve user {username}",
]

for username in usernames:

    for pattern in search_username_patterns:

        add(
            pattern.format(username=username),
            f"search username={username}"
        )


# ============================================================
# DELETE - USERNAME
# ============================================================

delete_username_patterns = [
    "Delete user {username}",
    "Delete the user named {username}",
    "Remove user {username}",
    "Remove the account of {username}",
    "Delete {username}'s account",
    "Erase user {username}",
    "Remove username {username}",
]

for username in usernames:

    for pattern in delete_username_patterns:

        add(
            pattern.format(username=username),
            f"delete username={username}"
        )


# ============================================================
# EMAILS
# ============================================================

emails = [
    "abdul@gmail.com",
    "ali@gmail.com",
    "ahmed@gmail.com",
    "rahul@gmail.com",
    "rohit@gmail.com",
    "john@gmail.com",
    "alex@gmail.com",
    "sarah@gmail.com",
    "david@gmail.com",
    "maria@gmail.com",
    "admin@example.com",
    "user@example.com",
    "test@example.com",
    "support@example.com",
]


# ============================================================
# SEARCH - EMAIL
# ============================================================

search_email_patterns = [
    "Find the user with email {email}",
    "Find user by email {email}",
    "Search user by email {email}",
    "Search for email {email}",
    "Show the account using {email}",
    "Look up the email {email}",
    "Find the account for {email}",
    "Get the user whose email is {email}",
    "Retrieve user with email {email}",
]

for email in emails:

    for pattern in search_email_patterns:

        add(
            pattern.format(email=email),
            f"search email={email}"
        )


# ============================================================
# DELETE - EMAIL
# ============================================================

delete_email_patterns = [
    "Delete the user with email {email}",
    "Remove user with email {email}",
    "Delete account {email}",
    "Remove account {email}",
    "Erase the account using {email}",
]

for email in emails:

    for pattern in delete_email_patterns:

        add(
            pattern.format(email=email),
            f"delete email={email}"
        )


# ============================================================
# ALL
# ============================================================

all_patterns = [
    "Show all users",
    "List all users",
    "Get all users",
    "Display every user",
    "Show everyone",
    "Give me all users",
    "List every user",
    "Display all records",
    "Fetch all users",
    "Show all records",
    "I want every user",
    "Give me the complete user list",
    "Get the full user database",
    "Display all user records",
    "Return all users",
]

for pattern in all_patterns:

    add(
        pattern,
        "all"
    )


# ============================================================
# INSERT
# ============================================================

insert_records = [
    ("1", "abdul", "abdul@gmail.com"),
    ("2", "ali", "ali@gmail.com"),
    ("3", "ahmed", "ahmed@gmail.com"),
    ("4", "rahul", "rahul@gmail.com"),
    ("5", "rohit", "rohit@gmail.com"),
    ("6", "john", "john@gmail.com"),
    ("7", "alex", "alex@gmail.com"),
    ("8", "sarah", "sarah@gmail.com"),
    ("9", "david", "david@gmail.com"),
    ("10", "maria", "maria@gmail.com"),
]


insert_patterns = [
    (
        "Insert user with id {id}, username {username}, "
        "and email {email}"
    ),
    (
        "Create a user with id {id}, username {username}, "
        "and email {email}"
    ),
    (
        "Add a new user: id {id}, username {username}, "
        "email {email}"
    ),
    (
        "Create user {username} with id {id} "
        "and email {email}"
    ),
    (
        "Add {username} to the database with id {id} "
        "and email {email}"
    ),
]

for user_id, username, email in insert_records:

    for pattern in insert_patterns:

        add(
            pattern.format(
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


# ============================================================
# INVALID / AMBIGUOUS
# ============================================================

invalid_examples = [
    (
        "Create a new user",
        "INVALID"
    ),
    (
        "Add a user",
        "INVALID"
    ),
    (
        "Delete a user",
        "INVALID"
    ),
    (
        "Search for a user",
        "INVALID"
    ),
    (
        "Find someone",
        "INVALID"
    ),
    (
        "Remove the account",
        "INVALID"
    ),
    (
        "Create user Abdul",
        "INVALID"
    ),
    (
        "Add Abdul to the database",
        "INVALID"
    ),
    (
        "Delete whoever is logged in",
        "INVALID"
    ),
    (
        "Show me the database",
        "INVALID"
    ),
    (
        "Do something with user 10",
        "INVALID"
    ),
]

for instruction, output in invalid_examples:

    add(
        instruction,
        output
    )


# ============================================================
# Exact command understanding
# ============================================================

command_examples = [
    (
        "search id=42",
        "search id=42"
    ),
    (
        "delete id=42",
        "delete id=42"
    ),
    (
        "all",
        "all"
    ),
    (
        "search username=abdul",
        "search username=abdul"
    ),
    (
        "delete email=abdul@gmail.com",
        "delete email=abdul@gmail.com"
    ),
]

for instruction, output in command_examples:

    add(
        instruction,
        output
    )


# ============================================================
# Deduplicate
# ============================================================

unique = {}

for example in examples:

    key = (
        example["instruction"].strip(),
        example["output"].strip()
    )

    unique[key] = example

examples = list(unique.values())


# ============================================================
# Shuffle
# ============================================================

random.seed(SEED)

random.shuffle(
    examples
)


# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_FILE,
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
# Statistics
# ============================================================

counts = {}

for example in examples:

    command = example["output"].split(
        " ",
        1
    )[0]

    counts[command] = (
        counts.get(command, 0) + 1
    )


print()
print("================================")
print("4mulaQuery Dataset Builder")
print("================================")

print()
print("Total examples:", len(examples))

print()
print("By command:")

for command, count in sorted(
    counts.items()
):

    print(
        f"{command}: {count}"
    )

print()
print("Output:", OUTPUT_FILE)

print()
print("Dataset created successfully.")
