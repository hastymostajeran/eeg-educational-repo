# %% [markdown]
# # Notebook 05 — EEG Preprocessing with MNE
#
# This notebook follows the 10-step teaching format for **Group 5: EEG Preprocessing**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# EEG preprocessing prepares raw voltage recordings for trustworthy analysis. We cover MNE fundamentals, bandpass/notch filters, artifact removal with ICA, epoching, and baseline correction.

# %% [markdown]
# ## 2. Necessary mathematics
# Bandpass filtering keeps frequencies $f$ such that $f_{low} \le f \le f_{high}$.
#
# A notch filter suppresses a narrow frequency such as power-line noise.
#
# ICA assumes observed EEG $X$ can be approximated as a mixture of independent sources: $X = AS$, where $A$ is a mixing matrix and $S$ are source components.
#
# Baseline correction subtracts the pre-stimulus mean:
#
# $X_{corrected}(t)=X(t)-\frac{1}{T_b}\sum_{t\in baseline}X(t)$.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from eeg_preprocessing_utils import load_mne_sample_raw, apply_eeg_filters, fit_ica_for_artifact_demo, create_epochs_from_sample

# Load a short real EEG segment.
raw = load_mne_sample_raw(max_minutes=0.5)
print(raw)
print('Sampling frequency:', raw.info['sfreq'])
print('Number of EEG channels:', len(raw.ch_names))

# Plot a few channels.
raw.copy().pick(raw.ch_names[:5]).plot(duration=5, scalings='auto', title='Raw EEG sample', show=True)


# %% [markdown]
# ## 3. Synthetic/Python example
# Before filtering real EEG, we use a synthetic signal with alpha rhythm and line noise.

# %%
from dsp_utils import make_sine_wave, butter_filter, compute_psd_welch

fs = 250
t, alpha = make_sine_wave(10, fs, 3)
_, line = make_sine_wave(50, fs, 3, amplitude=0.5)
synthetic = alpha + line + np.random.normal(0, 0.15, len(t))
synthetic_bandpassed = butter_filter(synthetic, fs, low_cut=1, high_cut=40)

freqs, psd = compute_psd_welch(synthetic, fs)
freqs_f, psd_f = compute_psd_welch(synthetic_bandpassed, fs)
plt.figure(figsize=(8,4))
plt.semilogy(freqs, psd, label='Original')
plt.semilogy(freqs_f, psd_f, label='Bandpassed')
plt.xlim(0, 70)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Synthetic filtering example')
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 4–7. Practical EEG example: bandpass, notch, ICA, epochs, baseline correction

# %%
# Filter a copy of the raw data.
filtered = apply_eeg_filters(raw, l_freq=1, h_freq=40, notch_freq=60)

# Compare PSD before and after filtering using MNE's plotting method.
raw.compute_psd(fmax=60).plot(picks='eeg', average=True, show=True)
filtered.compute_psd(fmax=60).plot(picks='eeg', average=True, show=True)

# ICA is fitted for artifact-inspection education.
# We do not automatically remove components because component rejection requires inspection.
ica = fit_ica_for_artifact_demo(filtered.copy().crop(tmax=30), n_components=10, random_state=42)
print(ica)
ica.plot_components(show=True)

# Create epochs and apply baseline correction through MNE Epochs.
epochs, evoked = create_epochs_from_sample(filtered)
print(epochs)
evoked.plot(spatial_colors=True, show=True)


# %% [markdown]
# ## 8. Interpretation
# Filtering reduces unwanted frequency content. ICA components are candidate sources that must be inspected before exclusion. Epoching converts continuous data into event-centered trials, and baseline correction makes post-stimulus activity easier to compare against pre-stimulus activity.

# %% [markdown]
# ## 9. Exercise
# Change the bandpass range from 1–40 Hz to 0.5–30 Hz. Compare the PSD and evoked plot. What changed?

# %% [markdown]
# ## 10. References
# - MNE preprocessing tutorials: https://mne.tools/stable/auto_tutorials/preprocessing/index.html
# - Makeig et al. ICA literature for EEG artifact analysis.
# - Luck, S. J. *An Introduction to the Event-Related Potential Technique*.

# %% [markdown]
# ## Key Takeaways
# - Raw EEG should be inspected before and after preprocessing.
# - ICA helps separate sources, but rejecting components is a human-guided decision.
# - Epoching and baseline correction are central for event-related analysis.
#
# ## Common Mistakes
# - Applying ICA without filtering first.
# - Removing ICA components blindly.
# - Forgetting average reference or reference documentation.
# - Comparing epochs without baseline correction when baseline matters.
#
# ## Mini Project
# Load a short MNE sample segment, filter it, compute PSD before and after filtering, create epochs for one condition, and plot the evoked response.
