from pathlib import Path

import pandas as pd


def validate_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    return True


def read_tsv(path: Path, usecols=None, nrows=None):
    validate_file(path)

    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )


def read_tsv_chunks(path: Path, usecols=None, chunksize=100_000):
    validate_file(path)

    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )


def get_columns(path: Path):
    validate_file(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        nrows=0,
    )

    return list(df.columns)