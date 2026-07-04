"""Utility functions for educational statistics examples in EEG analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


def descriptive_statistics(values: np.ndarray) -> Dict[str, float]:
    """Compute basic descriptive statistics for a one-dimensional signal.

    Parameters
    ----------
    values:
        One-dimensional NumPy array containing numeric samples, such as EEG
        amplitudes from one channel or one synthetic signal.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean, median, mode, variance, standard deviation,
        minimum, maximum, and range. Variance and standard deviation use
        ``ddof=1`` to estimate sample statistics.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("values must contain at least one sample")
    mode_result = stats.mode(x, keepdims=False)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "mode": float(mode_result.mode),
        "variance": float(np.var(x, ddof=1)) if x.size > 1 else 0.0,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "range": float(np.max(x) - np.min(x)),
    }


def z_score(values: np.ndarray) -> np.ndarray:
    """Standardize a signal into z-scores.

    Parameters
    ----------
    values:
        Numeric samples to standardize.

    Returns
    -------
    np.ndarray
        Array where each value is transformed as ``(x - mean) / std``. If the
        standard deviation is zero, an array of zeros is returned to avoid
        division by zero.
    """
    x = np.asarray(values, dtype=float)
    std = np.std(x, ddof=1)
    if np.isclose(std, 0.0):
        return np.zeros_like(x, dtype=float)
    return (x - np.mean(x)) / std


def covariance_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute covariance and Pearson correlation between two signals.

    Parameters
    ----------
    x:
        First one-dimensional numeric signal.
    y:
        Second one-dimensional numeric signal with the same length as ``x``.

    Returns
    -------
    Tuple[float, float]
        Sample covariance and Pearson correlation coefficient.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("x and y must have the same number of samples")
    if a.size < 2:
        raise ValueError("x and y must contain at least two samples")
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    corr = float(np.corrcoef(a, b)[0, 1])
    return cov, corr


def add_gaussian_noise(signal: np.ndarray, noise_std: float = 0.2, seed: int = 42) -> np.ndarray:
    """Add reproducible Gaussian noise to a signal.

    Parameters
    ----------
    signal:
        Clean input signal.
    noise_std:
        Standard deviation of the Gaussian noise.
    seed:
        Random seed used to make the generated noise reproducible.

    Returns
    -------
    np.ndarray
        Noisy signal with the same shape as ``signal``.
    """
    np.random.seed(seed)
    x = np.asarray(signal, dtype=float)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=x.shape)
    return x + noise


def root_mean_squared_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute root mean squared error between a reference and an estimate.

    Parameters
    ----------
    reference:
        Ground-truth or clean signal.
    estimate:
        Noisy, reconstructed, or altered signal with the same shape.

    Returns
    -------
    float
        Root mean squared error. Lower values mean the estimate is closer to
        the reference.
    """
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape")
    return float(np.sqrt(np.mean((ref - est) ** 2)))
