from collections import defaultdict

from .normalization import (
    normalize_country,
    normalize_name,
    normalize_address,
)


def tokenize(value):
    value = value.strip()

    if not value:
        return set()

    return set(value.split())


def row_tokens(row):
    name_tokens = tokenize(
        normalize_name(row["business_name"])
    )

    address_tokens = tokenize(
        normalize_address(row["business_address"])
    )

    return name_tokens | address_tokens


def token_variants(token):
    """
    Generate compact variants for approximate token retrieval.

    Examples:
        restaurant -> restaurant
        restaurant -> restau...
        restaurant -> ...urant
    """

    variants = {token}

    if len(token) >= 5:
        variants.add(token[:4])
        variants.add(token[:5])
        variants.add(token[-4:])
        variants.add(token[-5:])

    return variants


def build_approx_index(rows, frequency, max_df=50):
    """
    Build an inverted index over rare tokens and compact token variants.

    Only variants originating from rare tokens are indexed.
    """

    index = defaultdict(list)

    for row in rows:
        entity_id = row["entity_id"]

        country = normalize_country(row["country"])

        if not country:
            continue

        tokens = row_tokens(row)

        for token in tokens:

            key = (country, token)

            if frequency.get(key, 0) > max_df:
                continue

            for variant in token_variants(token):
                index[(country, variant)].append(entity_id)

    return index


def get_approx_candidates(
    row,
    index,
    max_candidates=200,
):
    """
    Retrieve candidates using exact rare tokens plus compact
    token-prefix/suffix variants.

    Candidates are ranked by the number of independent token
    signals they share with the query.
    """

    country = normalize_country(row["country"])

    if not country:
        return set()

    tokens = row_tokens(row)

    counts = defaultdict(int)

    for token in tokens:

        variants = token_variants(token)

        for variant in variants:

            entity_ids = index.get(
                (country, variant),
                [],
            )

            for entity_id in entity_ids:
                counts[entity_id] += 1

    if not counts:
        return set()

    # Only retain the strongest candidates.
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return {
        entity_id
        for entity_id, _ in ranked[:max_candidates]
    }