from collections import defaultdict

from .normalization import (
    normalize_country,
    normalize_name,
    normalize_address,
)


def make_name_signature(value):
    """
    Create multiple coarse signatures for a business name.

    These are retrieval keys, not final similarity scores.
    """

    value = normalize_name(value)

    if not value:
        return set()

    tokens = value.split()

    signatures = set()

    # Full normalized name.
    signatures.add(value)

    # Sorted-token representation.
    if tokens:
        signatures.add(
            " ".join(sorted(tokens))
        )

    # Remove spaces.
    signatures.add(
        value.replace(" ", "")
    )

    # Token initials.
    if tokens:
        initials = "".join(
            token[0]
            for token in tokens
            if token
        )

        if initials:
            signatures.add(initials)

    # First meaningful token.
    if tokens:
        signatures.add(tokens[0])

    return signatures


def make_address_signature(value):
    """
    Create coarse address signatures.
    """

    value = normalize_address(value)

    if not value:
        return set()

    tokens = value.split()

    signatures = set()

    signatures.add(value)

    signatures.add(
        value.replace(" ", "")
    )

    if tokens:
        signatures.add(
            " ".join(sorted(tokens))
        )

        signatures.add(tokens[0])

        # Numeric address tokens.
        numeric_tokens = [
            token
            for token in tokens
            if any(
                char.isdigit()
                for char in token
            )
        ]

        for token in numeric_tokens:
            signatures.add(token)

    return signatures


def build_similarity_index(rows):
    """
    Build country-aware indexes over coarse
    name and address signatures.
    """

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in rows:

        entity_id = row["entity_id"]

        country = normalize_country(
            row["country"]
        )

        if not country:
            continue

        name_signatures = (
            make_name_signature(
                row["business_name"]
            )
        )

        for signature in name_signatures:

            name_index[
                (country, signature)
            ].append(entity_id)

        address_signatures = (
            make_address_signature(
                row["business_address"]
            )
        )

        for signature in address_signatures:

            address_index[
                (country, signature)
            ].append(entity_id)

    return {
        "name": name_index,
        "address": address_index,
    }


def get_similarity_candidates(
    row,
    indexes,
    max_candidates=500,
):
    """
    Retrieve candidates using coarse name/address
    signatures.

    Candidates are ranked by number of shared
    signatures and truncated to max_candidates.
    """

    country = normalize_country(
        row["country"]
    )

    if not country:
        return set()

    counts = defaultdict(int)

    name_signatures = make_name_signature(
        row["business_name"]
    )

    for signature in name_signatures:

        entity_ids = indexes[
            "name"
        ].get(
            (country, signature),
            [],
        )

        for entity_id in entity_ids:
            counts[entity_id] += 1

    address_signatures = make_address_signature(
        row["business_address"]
    )

    for signature in address_signatures:

        entity_ids = indexes[
            "address"
        ].get(
            (country, signature),
            [],
        )

        for entity_id in entity_ids:
            counts[entity_id] += 1

    if not counts:
        return set()

    ranked = sorted(
        counts.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )

    return {
        entity_id
        for entity_id, _ in ranked[
            :max_candidates
        ]
    }