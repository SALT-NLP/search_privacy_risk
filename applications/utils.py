from rapidfuzz import fuzz

def string_contains(string: str, keyword: str) -> bool:
    """
    Check if any of the keywords are present in the string.

    Using fuzzy matching.
    """
    return fuzz.token_set_ratio(keyword, string) >= 50