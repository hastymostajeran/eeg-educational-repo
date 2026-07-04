# %% [markdown]
# # Notebook 06 — EEG Feature Extraction
#
# This notebook follows the 10-step teaching format for **Group 6: Feature Extraction**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# Feature extraction converts signals into interpretable numbers. In this notebook, features describe signal power, complexity, and distribution. They are not used for prediction or classification.

# %% [markdown]
# ## 2. Necessary mathematics
# Band power integrates PSD over a frequency band:
#
# $P_{band}=\int_{f_1}^{f_2}PSD(f)\,df$
#
# Relative power:
#
# $RP_{band}=\frac{P_{band}}{\sum_b P_b}$
#
# Hjorth activity: $\mathrm{var}(x)$
#
# Hjorth mobility: $\sqrt{\frac{\mathrm{var}(dx)}{\mathrm{var}(x)}}$
#
# Spectral entropy: $H=-\sum_i p_i\log_2(p_i)$, normalized by $\log_2(N)$.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from feature_extraction_utils import band_power, relative_band_powers, hjorth_parameters, spectral_entropy, time_frequency_features, BANDS
from dsp_utils import make_sine_wave, compute_psd_welch

fs = 250
t, alpha = make_sine_wave(10, fs, 5, amplitude=1.0)
_, beta = make_sine_wave(20, fs, 5, amplitude=0.5)
signal = alpha + beta + np.random.normal(0, 0.25, len(t))

rel = relative_band_powers(signal, fs)
hjorth = hjorth_parameters(signal)
ent = spectral_entropy(signal, fs)
features = time_frequency_features(signal, fs)
print('Relative powers:', rel)
print('Hjorth:', hjorth)
print('Spectral entropy:', ent)
print(pd.Series(features))

freqs, psd = compute_psd_welch(signal, fs)
plt.figure(figsize=(8,4))
plt.semilogy(freqs, psd)
plt.xlim(0, 45)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Synthetic signal PSD for feature extraction')
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.bar(rel.keys(), rel.values())
plt.ylabel('Relative power')
plt.title('Synthetic relative band powers')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Interpretation
# The relative-power plot should emphasize alpha because the synthetic signal contains a strong 10 Hz component. Hjorth activity summarizes variance, mobility relates to frequency content, and spectral entropy increases when power is more spread out.

# %% [markdown]
# ## 4–7. Practical EEG example using real MNE data

# %%
from eeg_preprocessing_utils import load_mne_sample_raw, apply_eeg_filters

raw = load_mne_sample_raw(max_minutes=0.5)
filtered = apply_eeg_filters(raw, l_freq=1, h_freq=45, notch_freq=60)
fs = filtered.info['sfreq']

# Extract features for the first five channels.
rows = []
for ch_name, ch_data in zip(filtered.ch_names[:5], filtered.get_data()[:5]):
    feats = time_frequency_features(ch_data * 1e6, fs)
    feats['channel'] = ch_name
    rows.append(feats)
feature_df = pd.DataFrame(rows).set_index('channel')
print(feature_df.round(4))

# Plot relative powers for each selected channel.
rel_cols = [c for c in feature_df.columns if c.startswith('rel_power_')]
feature_df[rel_cols].plot(kind='bar', figsize=(10,4))
plt.ylabel('Relative power')
plt.title('Relative band powers for real EEG channels')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 9. Exercise
# Compute features for ten channels instead of five. Which channel has the highest alpha relative power? Check the raw signal before making any claim.

# %% [markdown]
# ## 10. References
# - Hjorth, B. EEG analysis based on time domain properties.
# - Inouye et al. Quantification of EEG irregularity by entropy.
# - MNE and SciPy documentation for PSD estimation.

# %% [markdown]
# ## Key Takeaways
# - Features are summaries, not ground truth.
# - Band power depends on preprocessing, reference, and PSD settings.
# - Entropy and Hjorth features describe signal complexity and dynamics.
#
# ## Common Mistakes
# - Extracting features from uninspected noisy data.
# - Comparing features across datasets with different preprocessing.
# - Treating relative power as independent across bands.
# - Using features for classification despite this repository being non-predictive.
#
# ## Mini Project
# Filter a real MNE EEG segment, compute relative band powers and Hjorth parameters for multiple channels, create a feature table, and write a short interpretation of the strongest channel-level differences.
