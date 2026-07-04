# %% [markdown]
# # Notebook 04 — Digital Signal Processing
#
# This notebook follows the 10-step teaching format for **Group 4: DSP**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# DSP lets us understand EEG as waves sampled over time. Sampling, Nyquist, aliasing, FFT, PSD, and filtering explain how time-domain voltage becomes frequency-domain information.

# %% [markdown]
# ## 2. Necessary mathematics
# Sampling interval: $\Delta t = 1/f_s$.
#
# Nyquist frequency: $f_N=f_s/2$.
#
# A sine wave: $x(t)=A\sin(2\pi ft)$.
#
# Discrete Fourier Transform: $X_k=\sum_{n=0}^{N-1}x_n e^{-j2\pi kn/N}$.
#
# PSD describes how signal power is distributed across frequency.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from dsp_utils import make_sine_wave, compute_fft, compute_psd_welch, butter_filter

fs = 250
t, alpha = make_sine_wave(10, fs, 3, amplitude=1.0)
_, noise = make_sine_wave(50, fs, 3, amplitude=0.4)
noisy = alpha + noise + np.random.normal(0, 0.2, size=t.shape)
filtered = butter_filter(noisy, fs, low_cut=8, high_cut=13, order=4)

freqs, mag = compute_fft(noisy, fs)
psd_freqs, psd = compute_psd_welch(noisy, fs)
psd_freqs_f, psd_f = compute_psd_welch(filtered, fs)

plt.figure(figsize=(10, 4))
plt.plot(t, noisy, label='Noisy signal')
plt.plot(t, filtered, label='8–13 Hz filtered signal')
plt.xlim(0, 1)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time domain before and after bandpass filtering')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(freqs, mag)
plt.xlim(0, 80)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT magnitude spectrum')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.semilogy(psd_freqs, psd, label='Before filtering')
plt.semilogy(psd_freqs_f, psd_f, label='After filtering')
plt.xlim(0, 80)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('PSD before and after filtering')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Interpretation
# The FFT and PSD reveal strong energy near 10 Hz and 50 Hz. The 8–13 Hz bandpass keeps the alpha-like 10 Hz component and suppresses other frequencies.

# %% [markdown]
# ## 4–7. Practical EEG example using real MNE data

# %%
from eeg_preprocessing_utils import load_mne_sample_raw

raw = load_mne_sample_raw(max_minutes=0.25)
channel = raw.get_data(picks=[0])[0] * 1e6
fs = raw.info['sfreq']
freqs, psd = compute_psd_welch(channel, fs)

plt.figure(figsize=(8, 4))
plt.semilogy(freqs, psd)
plt.xlim(0, 60)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (µV²/Hz)')
plt.title(f'Welch PSD of real EEG channel {raw.ch_names[0]}')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 9. Exercise
# Generate a 30 Hz sine wave sampled at 50 Hz and then at 250 Hz. Compare what happens and explain aliasing using the Nyquist frequency.

# %% [markdown]
# ## 10. References
# - Oppenheim and Schafer, *Discrete-Time Signal Processing*.
# - van Drongelen, *Signal Processing for Neuroscientists*.
# - SciPy signal processing documentation.

# %% [markdown]
# ## Key Takeaways
# - Sampling rate determines the highest representable frequency.
# - FFT shows frequency content; PSD shows power distribution.
# - Filters should be selected based on the scientific question.
#
# ## Common Mistakes
# - Ignoring Nyquist frequency.
# - Filtering without checking PSD.
# - Using overly narrow filters without justification.
# - Treating filtering as artifact removal.
#
# ## Mini Project
# Generate a synthetic EEG signal, add noise, apply FFT, design a bandpass filter, apply it, and compare the PSD before and after filtering.
