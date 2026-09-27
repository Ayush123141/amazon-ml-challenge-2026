from .indexes import (
    make_name_key,
    make_address_key,
    make_name_address_key,
)


def get_candidates(row, indexes):
    """
    Return the union of candidates produced by:

    1. country + normalized name
    2. country + normalized address
    3. country + normalized name + normalized address
    """

    candidates = set()

    name_key = make_name_key(
        row["country"],
        row["business_name"],
    )

    if name_key is not None:
        candidates.update(
            indexes["name"].get(name_key, [])
        )

    address_key = make_address_key(
        row["country"],
        row["business_address"],
    )

    if address_key is not None:
        candidates.update(
            indexes["address"].get(address_key, [])
        )

    name_address_key = make_name_address_key(
        row["country"],
        row["business_name"],
        row["business_address"],
    )

    if name_address_key is not None:
        candidates.update(
            indexes["name_address"].get(name_address_key, [])
        )

    return candidates