# %% [markdown]
# # Notebook 01 — Statistics for EEG
#
# This notebook follows the 10-step teaching format for **Group 1: Statistics**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# Statistics summarize noisy EEG. Mean describes central tendency, variance and standard deviation describe spread, correlation/covariance describe co-movement, z-scores reveal unusual samples, and error metrics compare clean and noisy signals.

# %% [markdown]
# ## 2. Necessary mathematics
#
# Mean: $\mu = \frac{1}{N}\sum_{i=1}^{N}x_i$
#
# Variance: $s^2 = \frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})^2$
#
# Standard deviation: $s = \sqrt{s^2}$
#
# Z-score: $z_i = \frac{x_i-\bar{x}}{s}$
#
# Covariance: $\mathrm{cov}(x,y)=\frac{1}{N-1}\sum_i(x_i-\bar{x})(y_i-\bar{y})$
#
# Correlation: $r=\frac{\mathrm{cov}(x,y)}{s_xs_y}$
#
# RMSE: $\sqrt{\frac{1}{N}\sum_i(x_i-\hat{x}_i)^2}$

# %% [markdown]
# ## 3–7. Synthetic Python example with executable code, comments, plots, and interpretation

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from statistics_utils import descriptive_statistics, z_score, covariance_correlation, add_gaussian_noise, root_mean_squared_error

# Generate a simple 10 Hz synthetic EEG-like oscillation.
fs = 250
t = np.arange(0, 2, 1 / fs)
clean = np.sin(2 * np.pi * 10 * t)

# Add reproducible Gaussian noise.
noisy = add_gaussian_noise(clean, noise_std=0.4, seed=42)

# Compute descriptive statistics and error.
stats_table = descriptive_statistics(noisy)
rmse = root_mean_squared_error(clean, noisy)
zs = z_score(noisy)
cov, corr = covariance_correlation(clean, noisy)

print('Descriptive statistics:', stats_table)
print(f'RMSE clean vs noisy: {rmse:.4f}')
print(f'Covariance: {cov:.4f}, Correlation: {corr:.4f}')

# Plot clean vs noisy signal.
plt.figure(figsize=(10, 4))
plt.plot(t, clean, label='Clean 10 Hz signal')
plt.plot(t, noisy, label='Noisy signal', alpha=0.75)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic signal: clean vs noisy')
plt.legend()
plt.tight_layout()
plt.show()

# Plot z-score distribution.
plt.figure(figsize=(7, 4))
plt.hist(zs, bins=30, edgecolor='black')
plt.axvline(3, linestyle='--', label='+3 z')
plt.axvline(-3, linestyle='--', label='-3 z')
plt.xlabel('Z-score')
plt.ylabel('Count')
plt.title('Z-score distribution of noisy signal')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Interpretation
# The noisy signal keeps the original 10 Hz rhythm but contains random deviations. A high correlation means the noisy signal still follows the clean signal. Z-scores near ±3 indicate unusually large samples that may deserve inspection.

# %% [markdown]
# ## 4–7. Practical EEG example using real MNE data

# %%
import mne
from eeg_preprocessing_utils import load_mne_sample_raw

# Load a small real EEG segment from MNE sample data.
raw = load_mne_sample_raw(max_minutes=0.25)
data = raw.get_data(picks=[0])[0] * 1e6  # convert volts to microvolts
fs = raw.info['sfreq']
time = np.arange(data.size) / fs

print('Channel:', raw.ch_names[0])
print('Statistics in microvolts:', descriptive_statistics(data))

plt.figure(figsize=(10, 4))
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Real MNE EEG channel segment')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.hist(data, bins=40, edgecolor='black')
plt.xlabel('Amplitude (µV)')
plt.ylabel('Count')
plt.title('Amplitude distribution of one real EEG channel')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 9. Exercise
# Change the selected EEG channel index from `0` to another channel. Compare mean, standard deviation, and histogram shape. Which channel looks noisier?

# %% [markdown]
# ## 10. References
# - MNE-Python documentation: https://mne.tools/stable/
# - Cohen, M. X. *Analyzing Neural Time Series Data*.
# - van Drongelen, W. *Signal Processing for Neuroscientists*.

# %% [markdown]
# ## Key Takeaways
# - Statistics compress long EEG signals into interpretable summaries.
# - Z-scores are useful for unusual sample detection, not automatic deletion.
# - Correlation measures co-movement, not causation.
#
# ## Common Mistakes
# - Confusing variance with standard deviation.
# - Removing every high z-score without visual inspection.
# - Interpreting correlation as proof of physiological coupling.
# - Forgetting unit conversion from volts to microvolts in MNE.
#
# ## Mini Project
# Generate a 10 Hz signal, add three different noise levels, compute RMSE and correlation for each, then plot how error changes with noise level.
