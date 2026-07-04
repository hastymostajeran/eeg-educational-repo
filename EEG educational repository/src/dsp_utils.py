"""Digital signal processing utilities for EEG education."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def make_sine_wave(freq_hz: float, sampling_rate: float, duration_s: float, amplitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a sine wave with a fixed frequency.

    Parameters
    ----------
    freq_hz:
        Frequency of the sine wave in Hertz.
    sampling_rate:
        Sampling rate in samples per second.
    duration_s:
        Duration of the generated signal in seconds.
    amplitude:
        Peak amplitude of the sine wave.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Time vector and sine-wave samples.
    """
    t = np.arange(0, duration_s, 1 / sampling_rate)
    x = amplitude * np.sin(2 * np.pi * freq_hz * t)
    return t, x


def compute_fft(signal_values: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the single-sided FFT magnitude spectrum.

    Parameters
    ----------
    signal_values:
        One-dimensional time-domain signal.
    sampling_rate:
        Sampling rate in Hertz.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency bins and normalized magnitude spectrum.
    """
    x = np.asarray(signal_values, dtype=float)
    n = x.size
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate)
    magnitudes = np.abs(np.fft.rfft(x)) / n
    return freqs, magnitudes


def compute_psd_welch(signal_values: np.ndarray, sampling_rate: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate power spectral density using Welch's method.

    Parameters
    ----------
    signal_values:
        One-dimensional time-domain signal.
    sampling_rate:
        Sampling rate in Hertz.
    nperseg:
        Segment length for Welch averaging.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequency bins and PSD values.
    """
    x = np.asarray(signal_values, dtype=float)
    freqs, psd = signal.welch(x, fs=sampling_rate, nperseg=min(nperseg, x.size))
    return freqs, psd


def butter_filter(
    signal_values: np.ndarray,
    sampling_rate: float,
    low_cut: float | None = None,
    high_cut: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass, high-pass, or band-pass filter.

    Parameters
    ----------
    signal_values:
        One-dimensional time-domain signal.
    sampling_rate:
        Sampling rate in Hertz.
    low_cut:
        Lower cutoff frequency. Use ``None`` for low-pass filtering.
    high_cut:
        Upper cutoff frequency. Use ``None`` for high-pass filtering.
    order:
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as input.
    """
    nyquist = sampling_rate / 2
    if low_cut is not None and high_cut is not None:
        btype = "bandpass"
        cutoff = [low_cut / nyquist, high_cut / nyquist]
    elif low_cut is not None:
        btype = "highpass"
        cutoff = low_cut / nyquist
    elif high_cut is not None:
        btype = "lowpass"
        cutoff = high_cut / nyquist
    else:
        raise ValueError("At least one cutoff frequency must be provided")
    sos = signal.butter(order, cutoff, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(signal_values, dtype=float))
