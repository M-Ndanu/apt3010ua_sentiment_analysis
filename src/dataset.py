"""
Dataset utilities for the Kenyan News Sentiment Analysis project.

This module centralizes all dataset operations used by the training,
evaluation and model comparison scripts.

Responsibilities
----------------
- Load datasets
- Validate required columns
- Clean headlines
- Encode sentiment labels
- Split datasets consistently

This module intentionally contains NO model training or evaluation logic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import clean_text

REQUIRED_COLUMNS = frozenset(
    {
        "headline",
        "sentiment",
    }
)


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load and validate the labelled dataset.

    Parameters
    ----------
    path
        Path to the CSV dataset.

    Returns
    -------
    pd.DataFrame
        Validated dataframe.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=[
            "headline",
            "sentiment",
        ]
    ).copy()

    if df.empty:
        raise ValueError(
            "Dataset is empty after removing missing values."
        )

    return df


def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply preprocessing to every headline.

    Parameters
    ----------
    df
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe containing a clean_text column.
    """

    df = df.copy()

    df["clean_text"] = df["headline"].apply(clean_text)

    return df


def encode_labels(
    df: pd.DataFrame,
    encoder: LabelEncoder | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode sentiment labels.

    Parameters
    ----------
    df
        Prepared dataframe.

    encoder
        Existing fitted LabelEncoder. If omitted,
        a new encoder is created and fitted.

    Returns
    -------
    tuple[pd.DataFrame, LabelEncoder]
        Encoded dataframe and the encoder used.
    """

    df = df.copy()

    if encoder is None:
        encoder = LabelEncoder()
        df["label"] = encoder.fit_transform(
            df["sentiment"]
        )
    else:
        df["label"] = encoder.transform(
            df["sentiment"]
        )

    return df, encoder


def split_dataset(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
):
    """
    Split the dataset into train and test sets.

    Parameters
    ----------
    df
        Encoded dataframe.

    test_size
        Fraction reserved for testing.

    random_state
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train,
        X_test,
        y_train,
        y_test
    """

    return train_test_split(
        df["clean_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )