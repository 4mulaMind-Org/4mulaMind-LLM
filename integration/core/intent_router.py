import re


def rule_based_intent(query):

    q = query.strip().lower()

    # ========================================================
    # ALL
    # ========================================================

    if (
        any(word in q for word in [
            "all users",
            "every user",
            "all records",
            "every record",
            "all accounts",
            "everyone",
            "entire database",
            "full list",
            "complete list",
            "whole database",
            "poora database",
            "saare users",
            "sabhi users",
            "sab users",
            "saare records",
        ])
    ):
        return "ALL"

    # ========================================================
    # DELETE
    # ========================================================

    delete_words = [
        "delete",
        "remove",
        "erase",
        "destroy",
        "drop",
        "hata",
        "hatao",
        "hata do",
        "delete karo",
        "remove karo",
    ]

    if any(word in q for word in delete_words):

        # DELETE needs an ID
        if re.search(r"\b\d+\b", q):
            return "DELETE"

    # ========================================================
    # INSERT
    # ========================================================

    insert_words = [
        "insert",
        "add",
        "create",
        "register",
        "save",
        "store",
        "new user",
        "new account",
        "account bana",
        "user bana",
        "user add",
        "user create",
    ]

    if any(word in q for word in insert_words):

        # INSERT should contain user/account information
        if (
            re.search(r"\b\d+\b", q)
            or "@" in q
            or "username" in q
            or "email" in q
            or "user" in q
            or "account" in q
        ):
            return "INSERT"

    # ========================================================
    # SEARCH
    # ========================================================

    search_words = [
        "find",
        "search",
        "look for",
        "lookup",
        "look up",
        "fetch",
        "get",
        "show",
        "locate",
        "retrieve",
        "need",
        "find karo",
        "search karo",
        "dikhao",
        "dikha do",
        "record lao",
        "chahiye",
    ]

    if any(word in q for word in search_words):

        # SEARCH needs an ID
        if re.search(r"\b\d+\b", q):
            return "SEARCH"

    # ========================================================
    # Hindi / Hinglish search without explicit English keyword
    # ========================================================

    if (
        re.search(r"\b\d+\b", q)
        and any(word in q for word in [
            "wala user",
            "user ka record",
            "user ki details",
            "user ki information",
            "user chahiye",
            "user dikhao",
        ])
    ):
        return "SEARCH"

    # ========================================================
    # No confident rule
    # ========================================================

    return None