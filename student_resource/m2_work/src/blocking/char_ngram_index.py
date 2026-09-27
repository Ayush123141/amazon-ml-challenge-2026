from collections import defaultdict

from .normalization import (
    normalize_country,
    normalize_name,
)


def char_ngrams(value, n=3):
    """
    Generate character n-grams from normalized text.

    Spaces are retained because they carry useful
    word-boundary information.

    Example:
        "abc ltd"

    produces:
        "abc"
        "bc "
        "c l"
        " lt"
        "ltd"
    """

    value = normalize_name(value)

    if not value:
        return set()

    if len(value) < n:
        return {value}

    return {
        value[i:i + n]
        for i in range(len(value) - n + 1)
    }


def build_char_ngram_frequency(rows, n=3):
    """
    Count (country, ngram) document frequency.
    """

    frequency = defaultdict(int)

    for row in rows:

        country = normalize_country(
            row["country"]
        )

        if not country:
            continue

        grams = char_ngrams(
            row["business_name"],
            n=n,
        )

        for gram in grams:
            frequency[
                (country, gram)
            ] += 1

    return frequency


def build_char_ngram_index(
    rows,
    frequency,
    max_df=500,
    n=3,
):
    """
    Build country-aware inverted index.

    High-frequency n-grams are excluded because
    they generate huge candidate sets.
    """

    index = defaultdict(list)

    for row in rows:

        entity_id = row["entity_id"]

        country = normalize_country(
            row["country"]
        )

        if not country:
            continue

        grams = char_ngrams(
            row["business_name"],
            n=n,
        )

        for gram in grams:

            key = (country, gram)

            if frequency.get(
                key,
                0,
            ) <= max_df:

                index[key].append(
                    entity_id
                )

    return index