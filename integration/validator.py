import re


# ============================================================
# Allowed 4mulaQuery model output formats
# ============================================================

SEARCH_ID = re.compile(
    r"^search id=(\d+)$",
    re.IGNORECASE
)

SEARCH_USERNAME = re.compile(
    r"^search username=([A-Za-z0-9_.-]+)$",
    re.IGNORECASE
)

SEARCH_EMAIL = re.compile(
    r"^search email=([^\s]+)$",
    re.IGNORECASE
)


DELETE_ID = re.compile(
    r"^delete id=(\d+)$",
    re.IGNORECASE
)

DELETE_USERNAME = re.compile(
    r"^delete username=([A-Za-z0-9_.-]+)$",
    re.IGNORECASE
)

DELETE_EMAIL = re.compile(
    r"^delete email=([^\s]+)$",
    re.IGNORECASE
)


INSERT = re.compile(
    r"^insert "
    r"id=(\d+) "
    r"username=([A-Za-z0-9_.-]+) "
    r"email=([^\s]+)$",
    re.IGNORECASE
)

ALL = re.compile(
    r"^all$",
    re.IGNORECASE
)

INVALID = re.compile(
    r"^INVALID$",
    re.IGNORECASE
)


# ============================================================
# Validation + translation
# ============================================================

def validate_and_translate(
    model_output
):
    """
    Convert the structured model output into
    the existing 4mulaQuery CSV command format.

    Returns:
        {
            "valid": bool,
            "model_command": str,
            "engine_command": str | None,
            "reason": str | None
        }
    """

    if not isinstance(
        model_output,
        str
    ):

        return {
            "valid": False,
            "model_command": "",
            "engine_command": None,
            "reason": "Model output is not a string."
        }

    command = model_output.strip()

    # --------------------------------------------------------
    # INVALID
    # --------------------------------------------------------

    if INVALID.fullmatch(command):

        return {
            "valid": True,
            "model_command": command,
            "engine_command": None,
            "reason": "Model rejected the request."
        }

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    if ALL.fullmatch(command):

        return {
            "valid": True,
            "model_command": command,
            "engine_command": "all",
            "reason": None
        }

    # --------------------------------------------------------
    # SEARCH ID
    # --------------------------------------------------------

    match = SEARCH_ID.fullmatch(command)

    if match:

        user_id = match.group(1)

        return {
            "valid": True,
            "model_command": command,
            "engine_command": f"search,{user_id}",
            "reason": None
        }

    # --------------------------------------------------------
    # SEARCH USERNAME
    #
    # Current C++ engine does not support this yet.
    # Reject safely.
    # --------------------------------------------------------

    if SEARCH_USERNAME.fullmatch(command):

        return {
            "valid": False,
            "model_command": command,
            "engine_command": None,
            "reason": (
                "Current 4mulaQuery engine does not "
                "support username search."
            )
        }

    # --------------------------------------------------------
    # SEARCH EMAIL
    # --------------------------------------------------------

    if SEARCH_EMAIL.fullmatch(command):

        return {
            "valid": False,
            "model_command": command,
            "engine_command": None,
            "reason": (
                "Current 4mulaQuery engine does not "
                "support email search."
            )
        }

    # --------------------------------------------------------
    # DELETE ID
    # --------------------------------------------------------

    match = DELETE_ID.fullmatch(command)

    if match:

        user_id = match.group(1)

        return {
            "valid": True,
            "model_command": command,
            "engine_command": f"delete,{user_id}",
            "reason": None
        }

    # --------------------------------------------------------
    # DELETE USERNAME
    # --------------------------------------------------------

    if DELETE_USERNAME.fullmatch(command):

        return {
            "valid": False,
            "model_command": command,
            "engine_command": None,
            "reason": (
                "Current 4mulaQuery engine does not "
                "support username deletion."
            )
        }

    # --------------------------------------------------------
    # DELETE EMAIL
    # --------------------------------------------------------

    if DELETE_EMAIL.fullmatch(command):

        return {
            "valid": False,
            "model_command": command,
            "engine_command": None,
            "reason": (
                "Current 4mulaQuery engine does not "
                "support email deletion."
            )
        }

    # --------------------------------------------------------
    # INSERT
    # --------------------------------------------------------

    match = INSERT.fullmatch(command)

    if match:

        user_id = match.group(1)
        username = match.group(2)
        email = match.group(3)

        # ----------------------------------------------------
        # Defensive field validation
        # ----------------------------------------------------

        if len(username) == 0:
            return {
                "valid": False,
                "model_command": command,
                "engine_command": None,
                "reason": "Username cannot be empty."
            }

        if len(username) >= 32:
            return {
                "valid": False,
                "model_command": command,
                "engine_command": None,
                "reason": "Username exceeds engine limit."
            }

        if len(email) >= 255:
            return {
                "valid": False,
                "model_command": command,
                "engine_command": None,
                "reason": "Email exceeds engine limit."
            }

        if "," in username:
            return {
                "valid": False,
                "model_command": command,
                "engine_command": None,
                "reason": "Username contains CSV delimiter."
            }

        if "," in email:
            return {
                "valid": False,
                "model_command": command,
                "engine_command": None,
                "reason": "Email contains CSV delimiter."
            }

        return {
            "valid": True,
            "model_command": command,
            "engine_command": (
                f"insert,"
                f"{user_id},"
                f"{username},"
                f"{email}"
            ),
            "reason": None
        }

    # --------------------------------------------------------
    # Unknown command
    # --------------------------------------------------------

    return {
        "valid": False,
        "model_command": command,
        "engine_command": None,
        "reason": "Unsupported model output."
    }
