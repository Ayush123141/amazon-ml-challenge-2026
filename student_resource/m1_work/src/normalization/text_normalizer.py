"""
Text normalization utilities for Business Entity Resolution.

Amazon ML Challenge 2026
M1 - Data + Normalization

Design principles:
    1. Preserve original values.
    2. Generate multiple representations.
    3. Avoid destructive normalization.
    4. Preserve Unicode characters/scripts.
    5. Preserve digits.
    6. Keep token-order representation separate.
    7. Keep digit signatures separate from normal matching keys.

This module does NOT:
    - perform entity matching
    - perform blocking
    - perform fuzzy matching
    - expand legal suffixes
    - call external APIs
"""


from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List


# ============================================================
# CONSTANTS
# ============================================================

WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================
# BASIC VALUE HANDLING
# ============================================================

def safe_text(value: Any) -> str:
    """
    Convert a value into a safe string.

    None and NaN are converted to an empty string.

    Parameters
    ----------
    value:
        Any input value.

    Returns
    -------
    str
        Safe string representation.
    """

    if value is None:
        return ""

    try:
        if value != value:
            return ""
    except Exception:
        pass

    return str(value)


# ============================================================
# UNICODE NORMALIZATION
# ============================================================

def normalize_unicode(value: Any) -> str:
    """
    Apply Unicode NFKC normalization.

    NFKC can normalize compatibility forms while preserving
    the underlying Unicode script.

    Examples:
        full-width characters -> normalized forms
        compatibility characters -> normalized forms
    """

    text = safe_text(value)

    if not text:
        return ""

    return unicodedata.normalize("NFKC", text)


# ============================================================
# CASEFOLD
# ============================================================

def normalize_casefold(value: Any) -> str:
    """
    Unicode-aware case-insensitive representation.
    """

    text = normalize_unicode(value)

    if not text:
        return ""

    return text.casefold()


# ============================================================
# WHITESPACE NORMALIZATION
# ============================================================

def normalize_whitespace(value: Any) -> str:
    """
    Normalize repeated whitespace.

    Examples:
        'ABC   Pvt   Ltd'
        ->
        'abc pvt ltd'
    """

    text = normalize_casefold(value)

    if not text:
        return ""

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


# ============================================================
# PUNCTUATION NORMALIZATION
# ============================================================

def normalize_punctuation(value: Any) -> str:
    """
    Replace Unicode punctuation characters with spaces.

    Digits and letters are preserved.

    Example:
        'ABC-Pvt. Ltd.'
        ->
        'abc pvt ltd'
    """

    text = normalize_whitespace(value)

    if not text:
        return ""

    output = []

    for char in text:

        category = unicodedata.category(char)

        if category.startswith("P"):
            output.append(" ")
        else:
            output.append(char)

    text = "".join(output)

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


# ============================================================
# ALPHANUMERIC REPRESENTATION
# ============================================================

def normalize_alphanumeric(value: Any) -> str:
    """
    Keep Unicode letters and Unicode digits only.

    Spaces and punctuation are removed.

    Example:
        'ABC-Pvt. Ltd.'
        ->
        'abcpvt ltd' -> 'abcpvtltd'
    """

    text = normalize_casefold(value)

    if not text:
        return ""

    output = []

    for char in text:

        if char.isalnum():
            output.append(char)

    return "".join(output)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value: Any) -> List[str]:
    """
    Generate normalized tokens.

    Punctuation becomes token boundaries.
    """

    text = normalize_punctuation(value)

    if not text:
        return []

    return text.split()


# ============================================================
# SORTED TOKEN REPRESENTATION
# ============================================================

def normalize_sorted_tokens(value: Any) -> str:
    """
    Generate a token-order-insensitive representation.

    Example:

        'Global Tech Solutions'
        ->
        'global solutions tech'

    This representation is kept separate from the standard
    normalized representation because token order can carry
    information.
    """

    tokens = tokenize(value)

    if not tokens:
        return ""

    return " ".join(sorted(tokens))


# ============================================================
# DIGIT SIGNATURE
# ============================================================

def normalize_digits(value: Any) -> str:
    """
    Extract digits while preserving their order.

    Example:

        'Shop No. 24, Road 7'
        ->
        '247'

    This representation is diagnostic/feature-oriented and
    should not be used as a standalone entity key.
    """

    text = safe_text(value)

    if not text:
        return ""

    return "".join(
        char
        for char in text
        if char.isdigit()
    )


# ============================================================
# TOKEN LIST
# ============================================================

def normalized_tokens(value: Any) -> List[str]:
    """
    Return normalized token list.
    """

    return tokenize(value)


# ============================================================
# COMPLETE REPRESENTATION BUNDLE
# ============================================================

def normalize_text(value: Any) -> Dict[str, Any]:
    """
    Generate the complete multi-representation bundle.

    Returns
    -------
    dict

    Keys
    ----
    original
    unicode
    casefold
    whitespace
    punctuation
    alphanumeric
    tokens
    sorted_tokens
    digits
    """

    original = safe_text(value)

    unicode_value = normalize_unicode(original)
    casefold_value = normalize_casefold(unicode_value)
    whitespace_value = normalize_whitespace(casefold_value)
    punctuation_value = normalize_punctuation(whitespace_value)
    alphanumeric_value = normalize_alphanumeric(
        punctuation_value
    )

    tokens = tokenize(punctuation_value)

    sorted_token_value = (
        " ".join(sorted(tokens))
        if tokens
        else ""
    )

    digit_value = normalize_digits(original)

    return {
        "original": original,
        "unicode": unicode_value,
        "casefold": casefold_value,
        "whitespace": whitespace_value,
        "punctuation": punctuation_value,
        "alphanumeric": alphanumeric_value,
        "tokens": tokens,
        "sorted_tokens": sorted_token_value,
        "digits": digit_value,
    }


# ============================================================
# NAME REPRESENTATION
# ============================================================

def normalize_business_name(value: Any) -> Dict[str, Any]:
    """
    Generate representations for a business name.

    No legal-suffix expansion is performed.
    """

    return normalize_text(value)


# ============================================================
# ADDRESS REPRESENTATION
# ============================================================

def normalize_business_address(value: Any) -> Dict[str, Any]:
    """
    Generate representations for a business address.
    """

    return normalize_text(value)


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def add_normalized_columns(
    df,
    column: str,
    prefix: str | None = None
):
    """
    Add normalized representation columns to a pandas DataFrame.

    Parameters
    ----------
    df:
        Input pandas DataFrame.

    column:
        Source column containing text.

    prefix:
        Prefix for generated columns.

        If omitted, the source column name is used.

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with normalized columns.
    """

    result = df.copy()

    if prefix is None:
        prefix = column

    representations = [
        "unicode",
        "casefold",
        "whitespace",
        "punctuation",
        "alphanumeric",
        "sorted_tokens",
        "digits",
    ]

    bundles = result[column].map(normalize_text)

    for representation in representations:

        result[
            f"{prefix}_{representation}"
        ] = bundles.map(
            lambda x, key=representation: x[key]
        )

    return result


# ============================================================
# COMPARISON HELPERS
# ============================================================

def exact_equal(
    left: Any,
    right: Any,
    representation: str = "punctuation"
) -> bool:
    """
    Compare two values using one normalized representation.

    Empty values are never considered equal.
    """

    left_bundle = normalize_text(left)
    right_bundle = normalize_text(right)

    left_value = left_bundle[representation]
    right_value = right_bundle[representation]

    if not left_value or not right_value:
        return False

    return left_value == right_value


def compare_representations(
    left: Any,
    right: Any
) -> Dict[str, bool]:
    """
    Compare all normalized representations.

    Returns
    -------
    dict
        Representation -> equality boolean
    """

    left_bundle = normalize_text(left)
    right_bundle = normalize_text(right)

    representations = [
        "original",
        "unicode",
        "casefold",
        "whitespace",
        "punctuation",
        "alphanumeric",
        "sorted_tokens",
        "digits",
    ]

    result = {}

    for representation in representations:

        left_value = left_bundle[representation]
        right_value = right_bundle[representation]

        if representation == "digits":

            result[representation] = (
                bool(left_value)
                and bool(right_value)
                and left_value == right_value
            )

        else:

            result[representation] = (
                bool(left_value)
                and bool(right_value)
                and left_value == right_value
            )

    return result


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "safe_text",
    "normalize_unicode",
    "normalize_casefold",
    "normalize_whitespace",
    "normalize_punctuation",
    "normalize_alphanumeric",
    "tokenize",
    "normalized_tokens",
    "normalize_sorted_tokens",
    "normalize_digits",
    "normalize_text",
    "normalize_business_name",
    "normalize_business_address",
    "add_normalized_columns",
    "exact_equal",
    "compare_representations",
]