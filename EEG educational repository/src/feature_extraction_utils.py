"""Feature extraction utilities for educational EEG signal analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal, stats

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_power(values: np.ndarray, sampling_rate: float, band: Tuple[float, float]) -> float:
    """Compute absolute band power from Welch PSD.

    Parameters
    ----------
    values:
        One-dimensional EEG signal.
    sampling_rate:
        Sampling rate in Hertz.
    band:
        Tuple containing lower and upper band limits in Hertz.

    Returns
    -------
    float
        Integrated PSD inside the requested band.
    """
    x = np.asarray(values, dtype=float)
    freqs, psd = signal.welch(x, fs=sampling_rate, nperseg=min(512, x.size))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(idx):
        return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))


def relative_band_powers(values: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """Compute relative EEG band powers for canonical frequency bands.

    Parameters
    ----------
    values:
        One-dimensional EEG signal.
    sampling_rate:
        Sampling rate in Hertz.

    Returns
    -------
    Dict[str, float]
        Mapping from band name to relative power. Values sum approximately to 1
        across the included bands when total band power is nonzero.
    """
    absolute = {name: band_power(values, sampling_rate, band) for name, band in BANDS.items()}
    total = sum(absolute.values())
    if np.isclose(total, 0.0):
        return {name: 0.0 for name in absolute}
    return {name: power / total for name, power in absolute.items()}


def hjorth_parameters(values: np.ndarray) -> Dict[str, float]:
    """Compute Hjorth activity, mobility, and complexity.

    Parameters
    ----------
    values:
        One-dimensional EEG signal.

    Returns
    -------
    Dict[str, float]
        Hjorth activity, mobility, and complexity values.
    """
    x = np.asarray(values, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    mobility = np.sqrt(var_dx / var_x) if not np.isclose(var_x, 0.0) else 0.0
    mobility_dx = np.sqrt(var_ddx / var_dx) if not np.isclose(var_dx, 0.0) else 0.0
    complexity = mobility_dx / mobility if not np.isclose(mobility, 0.0) else 0.0
    return {"activity": float(var_x), "mobility": float(mobility), "complexity": float(complexity)}


def spectral_entropy(values: np.ndarray, sampling_rate: float) -> float:
    """Compute normalized spectral entropy from Welch PSD.

    Parameters
    ----------
    values:
        One-dimensional EEG signal.
    sampling_rate:
        Sampling rate in Hertz.

    Returns
    -------
    float
        Normalized spectral entropy between 0 and 1. Higher values indicate
        power is more spread across frequencies.
    """
    x = np.asarray(values, dtype=float)
    _, psd = signal.welch(x, fs=sampling_rate, nperseg=min(512, x.size))
    psd_sum = np.sum(psd)
    if np.isclose(psd_sum, 0.0):
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


def time_frequency_features(values: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """Compute grouped time-domain and frequency-domain descriptive features.

    Parameters
    ----------
    values:
        One-dimensional EEG signal.
    sampling_rate:
        Sampling rate in Hertz.

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean, standard deviation, skewness, kurtosis,
        Hjorth parameters, spectral entropy, and relative band powers.
    """
    x = np.asarray(values, dtype=float)
    features: Dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
        "spectral_entropy": spectral_entropy(x, sampling_rate),
    }
    features.update({f"hjorth_{k}": v for k, v in hjorth_parameters(x).items()})
    features.update({f"rel_power_{k}": v for k, v in relative_band_powers(x, sampling_rate).items()})
    return features
