"""Preprocessing utilities for educational tabular and EEG-like examples."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a DataFrame with duplicate rows removed.

    Parameters
    ----------
    df:
        Input DataFrame that may contain repeated rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with duplicate rows removed and row index reset.
    """
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_with_median(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Fill missing numeric values with each column's median.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric column names where missing values should be filled.

    Returns
    -------
    pd.DataFrame
        Copy of the DataFrame with selected missing values replaced.
    """
    result = df.copy()
    for col in columns:
        result[col] = result[col].fillna(result[col].median())
    return result


def detect_outliers_zscore(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using absolute z-score thresholding.

    Parameters
    ----------
    values:
        Numeric one-dimensional array.
    threshold:
        Absolute z-score above which a sample is marked as an outlier.

    Returns
    -------
    np.ndarray
        Boolean mask where ``True`` indicates an outlier.
    """
    x = np.asarray(values, dtype=float)
    std = np.std(x, ddof=1)
    if np.isclose(std, 0.0):
        return np.zeros_like(x, dtype=bool)
    z = (x - np.mean(x)) / std
    return np.abs(z) > threshold


def cap_outliers_iqr(values: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """Winsorize outliers using the interquartile range rule.

    Parameters
    ----------
    values:
        Numeric one-dimensional array.
    multiplier:
        IQR multiplier used to define lower and upper fences.

    Returns
    -------
    np.ndarray
        Array where values outside the IQR fences are clipped to the fence.
    """
    x = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return np.clip(x, lower, upper)


def build_preprocessing_pipeline(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Build a preprocessing-only scikit-learn ColumnTransformer.

    Parameters
    ----------
    numeric_features:
        Names of numeric columns to standardize.
    categorical_features:
        Names of categorical columns to one-hot encode.

    Returns
    -------
    ColumnTransformer
        Transformer that standardizes numeric columns and one-hot encodes
        categorical columns. It performs no model training.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def reproducible_train_test_split(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test partitions for leakage demonstrations only.

    Parameters
    ----------
    df:
        Input DataFrame to split.
    test_size:
        Fraction of rows assigned to the test partition.
    random_state:
        Fixed random state for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames. No model is fitted or evaluated.
    """
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)
