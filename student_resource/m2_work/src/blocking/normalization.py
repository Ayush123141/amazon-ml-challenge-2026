import re
import unicodedata


def normalize_text(value):
    """
    Unicode-safe text normalization.

    Preserves:
    - Unicode letters
    - Unicode numbers
    - Unicode combining marks

    Replaces:
    - punctuation
    - symbols
    with spaces.
    """

    if value is None:
        return ""

    value = str(value)

    if not value:
        return ""

    # Canonical compatibility normalization
    value = unicodedata.normalize("NFKC", value)

    # Case normalization
    value = value.casefold()

    # Keep letters, numbers, combining marks and whitespace.
    # Replace punctuation/symbols with spaces.
    cleaned = []

    for char in value:
        category = unicodedata.category(char)

        if (
            category.startswith("L")   # Letter
            or category.startswith("N") # Number
            or category.startswith("M") # Combining mark
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def tokenize(value):
    normalized = normalize_text(value)

    if not normalized:
        return []

    return normalized.split()


def unique_tokens(value):
    return set(tokenize(value))


def normalize_country(value):
    return normalize_text(value)


def normalize_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)