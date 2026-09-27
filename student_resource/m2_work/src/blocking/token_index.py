from collections import Counter, defaultdict

from .normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)


def tokenize_row(row):
    """
    Return unique normalized tokens from business name
    and business address.
    """

    name_tokens = set(
        normalize_name(row["business_name"]).split()
    )

    address_tokens = set(
        normalize_address(row["business_address"]).split()
    )

    return name_tokens | address_tokens


def build_token_frequency(rows):
    """
    Count document frequency of tokens within country.

    Key:
        (country, token)

    Value:
        number of entities containing that token
    """

    frequency = Counter()

    for row in rows:

        country = normalize_country(
            row["country"]
        )

        if not country:
            continue

        tokens = tokenize_row(row)

        for token in tokens:
            frequency[(country, token)] += 1

    return frequency


def build_rare_token_index(
    rows,
    frequency,
    max_df,
):
    """
    Build:

        (country, token) -> entity IDs

    only for tokens whose document frequency
    is <= max_df.
    """

    index = defaultdict(list)

    for row in rows:

        entity_id = row["entity_id"]

        country = normalize_country(
            row["country"]
        )

        if not country:
            continue

        tokens = tokenize_row(row)

        for token in tokens:

            key = (country, token)

            if frequency.get(key, 0) <= max_df:
                index[key].append(entity_id)

    return index