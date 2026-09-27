import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blocking.normalization import (
    normalize_text,
    normalize_name,
    normalize_address,
    normalize_country,
    tokenize,
    unique_tokens,
)


def main():

    test_values = [
        "ABC-Pvt. Ltd.",
        "  Prime   Money  ",
        "Léarning CENTER",
        "राम मार्केटिंग प्राइवेट लिमिटेड",
        "17560 Ellis Road, Tahlequah, OK",
        "",
        None,
    ]

    print("=" * 70)
    print("NORMALIZATION TEST")
    print("=" * 70)

    for value in test_values:
        print(f"\nOriginal : {repr(value)}")
        print(f"Text     : {repr(normalize_text(value))}")
        print(f"Name     : {repr(normalize_name(value))}")
        print(f"Address  : {repr(normalize_address(value))}")
        print(f"Tokens   : {tokenize(value)}")
        print(f"Unique   : {unique_tokens(value)}")

    print("\n" + "=" * 70)
    print("COUNTRY TEST")
    print("=" * 70)

    countries = [
        "US",
        "India",
        "INDIA",
        "  India  ",
        "",
    ]

    for country in countries:
        print(f"{repr(country)} -> {repr(normalize_country(country))}")


if __name__ == "__main__":
    main()