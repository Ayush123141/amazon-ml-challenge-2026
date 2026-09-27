from .char_ngram_index import char_ngrams
from .normalization import normalize_country


def get_char_ngram_candidates(
    row,
    index,
    n=3,
    min_shared_ngrams=1,
):
    """
    Retrieve candidates sharing at least
    min_shared_ngrams character n-grams
    with the Source 1 business name.
    """

    country = normalize_country(
        row["country"]
    )

    if not country:
        return set()

    grams = char_ngrams(
        row["business_name"],
        n=n,
    )

    candidate_counts = {}

    for gram in grams:

        key = (country, gram)

        entity_ids = index.get(
            key,
            [],
        )

        for entity_id in entity_ids:

            candidate_counts[entity_id] = (
                candidate_counts.get(
                    entity_id,
                    0,
                ) + 1
            )

    return {
        entity_id
        for entity_id, count
        in candidate_counts.items()
        if count >= min_shared_ngrams
    }