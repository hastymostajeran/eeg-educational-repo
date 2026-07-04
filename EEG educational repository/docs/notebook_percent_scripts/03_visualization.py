# %% [markdown]
# # Notebook 03 — Visualization
#
# This notebook follows the 10-step teaching format for **Group 3: Visualization**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# Visualization helps you see distribution, spread, relationships, time structure, and channel-level patterns before making conclusions.

# %% [markdown]
# ## 2. Necessary mathematics
# Histograms approximate distributions. Boxplots summarize median, quartiles, and outliers. Scatter plots reveal relationships. Correlation matrices display pairwise Pearson correlations $r$ across variables.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from visualization_utils import plot_histogram, plot_boxplot, plot_scatter, plot_line, plot_correlation_heatmap

# Synthetic EEG-like features.
time = np.arange(0, 2, 1/250)
alpha = np.sin(2*np.pi*10*time) + np.random.normal(0, 0.2, len(time))
beta = 0.5*np.sin(2*np.pi*20*time) + np.random.normal(0, 0.2, len(time))
features = pd.DataFrame({'alpha_power': alpha**2, 'beta_power': beta**2, 'amplitude': alpha + beta})

plot_line(time, alpha, 'Synthetic alpha-like waveform', 'Amplitude')
plt.show()
plot_histogram(features['amplitude'], 'Amplitude histogram', 'Amplitude')
plt.show()
plot_boxplot(features['alpha_power'], 'Alpha power boxplot', 'Power')
plt.show()
plot_scatter(features['alpha_power'], features['beta_power'], 'Alpha vs beta power', 'Alpha power', 'Beta power')
plt.show()
plot_correlation_heatmap(features, 'Synthetic feature correlation matrix')
plt.show()


# %% [markdown]
# ## 8. Interpretation
# The line plot shows temporal rhythm, the histogram shows amplitude distribution, the boxplot highlights spread, the scatter plot shows whether two features move together, and the heatmap summarizes feature relationships.

# %% [markdown]
# ## 4–7. Practical EEG example using real MNE data

# %%
from eeg_preprocessing_utils import load_mne_sample_raw

raw = load_mne_sample_raw(max_minutes=0.2)
data = raw.get_data()[:4] * 1e6
fs = raw.info['sfreq']
time = np.arange(data.shape[1]) / fs

plt.figure(figsize=(10, 5))
for idx, ch_data in enumerate(data):
    plt.plot(time, ch_data + idx * 80, label=raw.ch_names[idx])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude + offset (µV)')
plt.title('Four real EEG channels with vertical offsets')
plt.legend()
plt.tight_layout()
plt.show()

channel_df = pd.DataFrame(data.T, columns=raw.ch_names[:4])
plot_correlation_heatmap(channel_df, 'Correlation among four EEG channels')
plt.show()


# %% [markdown]
# ## 9. Exercise
# Plot six EEG channels instead of four. Try changing the vertical offset and explain why offsets help visual readability.

# %% [markdown]
# ## 10. References
# - Matplotlib documentation: https://matplotlib.org/stable/
# - Seaborn documentation: https://seaborn.pydata.org/
# - MNE visualization tutorials.

# %% [markdown]
# ## Key Takeaways
# - Choose plots that answer a specific question.
# - Avoid redundant visualizations.
# - Correlation heatmaps are useful but can hide time-varying behavior.
#
# ## Common Mistakes
# - Plotting too many channels without offsets.
# - Using heatmaps without checking raw signals.
# - Overinterpreting noisy scatter plots.
# - Forgetting units on axes.
#
# ## Mini Project
# Create a dashboard-like notebook section with one line plot, one histogram, one boxplot, and one channel correlation heatmap for a real MNE EEG segment.
