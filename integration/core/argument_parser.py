import re


# ============================================================
# Query normalization
# ============================================================

def normalize_query(text):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# ============================================================
# Negative number detection
# ============================================================

def contains_negative_number(query):

    return bool(
        re.search(
            r"-\s*\d+\b",
            query
        )
    )


# ============================================================
# ID extraction
# ============================================================

def extract_id(query):

    query = normalize_query(
        query
    )

    # Never silently turn -42 into 42.
    if contains_negative_number(query):

        return None

    patterns = [

        # id 42
        r"\bid\s*(?:is|=|:)?\s*(\d+)\b",

        # user 42
        r"\buser\s+(?:number|no\.?)?\s*(\d+)\b",

        # user number 42
        r"\buser\s+number\s+(\d+)\b",

        # account 42
        r"\baccount\s+(?:number|no\.?)?\s*(\d+)\b",

        # record 42
        r"\brecord\s+(?:number|no\.?)?\s*(\d+)\b",

        # number 42
        r"\bnumber\s+(\d+)\b",

        # no 42
        r"\bno\.?\s*(\d+)\b",

        # Hindi/Hinglish:
        # "42 wale user"
        r"\b(\d+)\s+wal(?:a|e)\s+user\b",

        # "user 42 ko"
        r"\buser\s+(\d+)\s+ko\b",

        # "user 42 ka"
        r"\buser\s+(\d+)\s+ka\b",

        # "user 42 ki"
        r"\buser\s+(\d+)\s+ki\b",

        # "user 42"
        r"\buser\s+(\d+)\b",

        # "account 42"
        r"\baccount\s+(\d+)\b",

        # "record 42"
        r"\brecord\s+(\d+)\b",
    ]

    matches = []

    for pattern in patterns:

        for value in re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        ):

            matches.append(
                str(value)
            )

    # Unique values.
    matches = list(
        dict.fromkeys(matches)
    )

    if len(matches) != 1:

        return None

    try:

        value = int(
            matches[0]
        )

    except ValueError:

        return None

    if value < 0:
        return None

    # uint32_t upper bound.
    if value > 4294967295:
        return None

    return str(value)


# ============================================================
# Username extraction
# ============================================================

def extract_username(query):

    query = normalize_query(
        query
    )

    patterns = [

        # username abdul
        r"\busername\s*(?:is|=|:)?\s*([A-Za-z0-9_.-]+)\b",

        # user named abdul
        r"\buser\s+named\s+([A-Za-z0-9_.-]+)\b",

        # user abdul
        r"\buser\s+([A-Za-z][A-Za-z0-9_.-]*)\b",

        # account of abdul
        r"\baccount\s+(?:of|for)\s+([A-Za-z][A-Za-z0-9_.-]*)\b",

        # named abdul
        r"\bnamed\s+([A-Za-z][A-Za-z0-9_.-]*)\b",
    ]

    blocked = {
        "the",
        "a",
        "an",
        "new",
        "with",
        "whose",
        "is",
        "id",
        "number",
        "no",
        "and",
        "email",
        "please",
        "user",
        "account",
        "record",
    }

    candidates = []

    for pattern in patterns:

        for value in re.findall(
            pattern,
            query,
            flags=re.IGNORECASE
        ):

            value = value.strip()

            if (
                value.lower()
                not in blocked
            ):

                candidates.append(
                    value
                )

    candidates = list(
        dict.fromkeys(
            value.lower()
            for value in candidates
        )
    )

    if len(candidates) != 1:

        return None

    username = candidates[0]

    if len(username) >= 32:
        return None

    return username


# ============================================================
# Email extraction
# ============================================================

def extract_email(query):

    query = normalize_query(
        query
    )

    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        query
    )

    emails = list(
        dict.fromkeys(
            email.lower()
            for email in emails
        )
    )

    if len(emails) != 1:

        return None

    email = emails[0]

    if len(email) >= 255:
        return None

    return email


# ============================================================
# INSERT extraction
# ============================================================

def extract_insert_arguments(query):

    query = normalize_query(
        query
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    user_id = extract_id(
        query
    )

    email = extract_email(
        query
    )

    # --------------------------------------------------------
    # Username: prefer explicit forms.
    # --------------------------------------------------------

    username = None

    explicit_patterns = [

        # username abdul
        r"\busername\s*(?:is|=|:)?\s*"
        r"([A-Za-z0-9_.-]+)\b",

        # named rahul
        r"\bnamed\s+"
        r"([A-Za-z0-9_.-]+)\b",

        # record for sam
        r"\brecord\s+(?:for|of)\s+"
        r"([A-Za-z0-9_.-]+)\b",

        # create user rahul
        r"\bcreate\s+user\s+"
        r"([A-Za-z0-9_.-]+)\b",

        # add user rahul
        r"\badd\s+user\s+"
        r"([A-Za-z0-9_.-]+)\b",

        # store user rahul
        r"\bstore\s+user\s+"
        r"([A-Za-z0-9_.-]+)\b",

        # save user rahul
        r"\bsave\s+user\s+"
        r"([A-Za-z0-9_.-]+)\b",
    ]

    for pattern in explicit_patterns:

        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE
        )

        if match:

            candidate = (
                match.group(1)
                .strip()
                .lower()
            )

            if candidate not in {
                "a",
                "an",
                "the",
                "new",
                "with",
                "and",
                "user",
                "username",
            }:

                username = candidate

                break

    # --------------------------------------------------------
    # Pattern:
    # "user 7654 named rahul"
    # --------------------------------------------------------

    if username is None:

        match = re.search(
            r"\buser\s+\d+\s+named\s+"
            r"([A-Za-z0-9_.-]+)\b",
            query,
            flags=re.IGNORECASE
        )

        if match:

            username = (
                match.group(1)
                .lower()
            )

    # --------------------------------------------------------
    # Pattern:
    # "7654 named rahul"
    # --------------------------------------------------------

    if username is None:

        match = re.search(
            r"\b\d+\s+named\s+"
            r"([A-Za-z0-9_.-]+)\b",
            query,
            flags=re.IGNORECASE
        )

        if match:

            username = (
                match.group(1)
                .lower()
            )

    # --------------------------------------------------------
    # Validate required fields.
    # --------------------------------------------------------

    if (
        user_id is None
        or username is None
        or email is None
    ):

        return None

    if len(username) >= 32:

        return None

    if "," in username:

        return None

    if "," in email:

        return None

    return {
        "id": user_id,
        "username": username,
        "email": email,
    }


# ============================================================
# Intent-aware parser
# ============================================================

def parse_arguments(
    query,
    intent
):

    query = normalize_query(
        query
    )

    intent = (
        intent.strip().upper()
    )

    if not query:

        return {
            "valid": False,
            "reason": "Empty query.",
            "arguments": None,
        }

    if intent == "INVALID":

        return {
            "valid": False,
            "reason": (
                "Request classified as INVALID."
            ),
            "arguments": None,
        }

    if intent == "ALL":

        return {
            "valid": True,
            "reason": None,
            "arguments": {},
        }

    if intent in {
        "SEARCH",
        "DELETE",
    }:

        user_id = extract_id(
            query
        )

        if user_id is None:

            return {
                "valid": False,
                "reason": (
                    "Could not safely extract "
                    "exactly one user ID."
                ),
                "arguments": None,
            }

        return {
            "valid": True,
            "reason": None,
            "arguments": {
                "id": user_id
            },
        }

    if intent == "INSERT":

        arguments = (
            extract_insert_arguments(
                query
            )
        )

        if arguments is None:

            return {
                "valid": False,
                "reason": (
                    "Could not safely extract "
                    "id, username and email."
                ),
                "arguments": None,
            }

        return {
            "valid": True,
            "reason": None,
            "arguments": arguments,
        }

    return {
        "valid": False,
        "reason": (
            f"Unsupported intent: {intent}"
        ),
        "arguments": None,
    }


# ============================================================
# Engine command
# ============================================================

def build_engine_command(
    intent,
    arguments
):

    intent = intent.strip().upper()

    if intent == "SEARCH":

        return (
            f"search,{arguments['id']}"
        )

    if intent == "DELETE":

        return (
            f"delete,{arguments['id']}"
        )

    if intent == "ALL":

        return "all"

    if intent == "INSERT":

        return (
            f"insert,"
            f"{arguments['id']},"
            f"{arguments['username']},"
            f"{arguments['email']}"
        )

    return None


# ============================================================
# Full safe command
# ============================================================

def build_safe_command(
    query,
    intent
):

    parsed = parse_arguments(
        query,
        intent
    )

    if not parsed["valid"]:

        return {
            "valid": False,
            "intent": intent,
            "arguments": None,
            "engine_command": None,
            "reason": parsed["reason"],
        }

    if intent.strip().upper() == "INVALID":

        return {
            "valid": False,
            "intent": "INVALID",
            "arguments": None,
            "engine_command": None,
            "reason": (
                "Request classified as INVALID."
            ),
        }

    command = build_engine_command(
        intent,
        parsed["arguments"]
    )

    if command is None:

        return {
            "valid": False,
            "intent": intent,
            "arguments": parsed["arguments"],
            "engine_command": None,
            "reason": (
                "Could not build engine command."
            ),
        }

    return {
        "valid": True,
        "intent": intent.strip().upper(),
        "arguments": parsed["arguments"],
        "engine_command": command,
        "reason": None,
    }
