import re
import subprocess
from pathlib import Path


# ============================================================
# Existing 4mulaQuery binary
# ============================================================

ENGINE_BINARY = (
    Path.home()
    / "Desktop/workSpace/project/4mulaQuery/core/4mulaQuery"
)


# ============================================================
# Isolated working directory
#
# IMPORTANT:
# The C++ engine uses:
#     data/4mulaQuery.db
#
# relative to its current working directory.
# Therefore we intentionally run it inside:
#     integration/runtime/
#
# This protects the real project database.
# ============================================================

RUNTIME_DIR = (
    Path(__file__).resolve().parent
    / "runtime"
)


# ============================================================
# Allowed engine commands
# ============================================================

SEARCH_PATTERN = re.compile(
    r"^search,\d+$"
)

DELETE_PATTERN = re.compile(
    r"^delete,\d+$"
)

INSERT_PATTERN = re.compile(
    r"^insert,"
    r"\d+,"
    r"[A-Za-z0-9_.-]+,"
    r"[A-Za-z0-9._%+\-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)


def validate_engine_command(command):

    if not isinstance(command, str):
        return False

    command = command.strip()

    if command == "all":
        return True

    if SEARCH_PATTERN.fullmatch(command):
        return True

    if DELETE_PATTERN.fullmatch(command):
        return True

    if INSERT_PATTERN.fullmatch(command):
        return True

    return False


def execute_engine(command):

    command = command.strip()

    # --------------------------------------------------------
    # Never execute arbitrary strings.
    # --------------------------------------------------------

    if not validate_engine_command(command):

        return {
            "success": False,
            "command": command,
            "output": "",
            "error": (
                "Engine command rejected by executor."
            ),
        }

    # --------------------------------------------------------
    # Binary existence
    # --------------------------------------------------------

    if not ENGINE_BINARY.exists():

        return {
            "success": False,
            "command": command,
            "output": "",
            "error": (
                f"Engine binary not found: "
                f"{ENGINE_BINARY}"
            ),
        }

    # --------------------------------------------------------
    # Ensure isolated runtime DB directory exists.
    # --------------------------------------------------------

    runtime_data = (
        RUNTIME_DIR / "data"
    )

    runtime_data.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Execute one command.
    #
    # shell=False is intentional.
    # --------------------------------------------------------

    try:

        process = subprocess.run(
            [str(ENGINE_BINARY)],
            input=command + "\n",
            text=True,
            capture_output=True,
            cwd=str(RUNTIME_DIR),
            timeout=10,
            shell=False,
        )

    except subprocess.TimeoutExpired:

        return {
            "success": False,
            "command": command,
            "output": "",
            "error": (
                "4mulaQuery execution timed out."
            ),
        }

    except OSError as exc:

        return {
            "success": False,
            "command": command,
            "output": "",
            "error": (
                f"Could not start engine: {exc}"
            ),
        }

    stdout = process.stdout.strip()

    stderr = process.stderr.strip()

    # --------------------------------------------------------
    # Process result
    # --------------------------------------------------------

    if process.returncode != 0:

        return {
            "success": False,
            "command": command,
            "output": stdout,
            "error": (
                stderr
                or
                f"Engine exited with code "
                f"{process.returncode}"
            ),
        }

    return {
        "success": True,
        "command": command,
        "output": stdout,
        "error": None,
    }


def main():

    print()
    print("================================")
    print("4mulaQuery Isolated Executor")
    print("================================")

    print()
    print(
        "Binary:",
        ENGINE_BINARY
    )

    print(
        "Runtime:",
        RUNTIME_DIR
    )

    print()
    print(
        "Real project database is NOT used."
    )

    print()
    print(
        "Type 'exit' to quit."
    )

    while True:

        try:

            command = input(
                "\nEngine > "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print()
            break

        if command.lower() == "exit":
            break

        if not command:
            continue

        result = execute_engine(
            command
        )

        print()
        print(
            "Success:",
            result["success"]
        )

        print(
            "Output:",
            result["output"]
        )

        if result["error"]:

            print(
                "Error:",
                result["error"]
            )

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()
