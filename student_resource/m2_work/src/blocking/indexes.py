from collections import defaultdict

from .normalization import (
    normalize_name,
    normalize_address,
    normalize_country,
)


def make_name_key(country, business_name):
    country = normalize_country(country)
    name = normalize_name(business_name)

    if not country or not name:
        return None

    return country, name


def make_address_key(country, business_address):
    country = normalize_country(country)
    address = normalize_address(business_address)

    if not country or not address:
        return None

    return country, address


def make_name_address_key(country, business_name, business_address):
    country = normalize_country(country)
    name = normalize_name(business_name)
    address = normalize_address(business_address)

    if not country or not name or not address:
        return None

    return country, name, address


def build_name_index(rows):
    """
    country + normalized business name
        ->
    list of entity IDs
    """

    index = defaultdict(list)

    for row in rows:
        key = make_name_key(
            row["country"],
            row["business_name"],
        )

        if key is not None:
            index[key].append(row["entity_id"])

    return index


def build_address_index(rows):
    """
    country + normalized business address
        ->
    list of entity IDs
    """

    index = defaultdict(list)

    for row in rows:
        key = make_address_key(
            row["country"],
            row["business_address"],
        )

        if key is not None:
            index[key].append(row["entity_id"])

    return index


def build_name_address_index(rows):
    """
    country + normalized name + normalized address
        ->
    list of entity IDs
    """

    index = defaultdict(list)

    for row in rows:
        key = make_name_address_key(
            row["country"],
            row["business_name"],
            row["business_address"],
        )

        if key is not None:
            index[key].append(row["entity_id"])

    return index


def build_all_indexes(rows):
    """
    Build all exact blocking indexes.
    """

    return {
        "name": build_name_index(rows),
        "address": build_address_index(rows),
        "name_address": build_name_address_index(rows),
    }