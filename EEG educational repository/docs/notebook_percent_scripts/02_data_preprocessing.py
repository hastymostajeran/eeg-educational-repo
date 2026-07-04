# %% [markdown]
# # Notebook 02 — Data Preprocessing
#
# This notebook follows the 10-step teaching format for **Group 2: Preprocessing**.
#
# No classification, prediction, model training, SVM, neural networks, deep learning, or accuracy evaluation is used anywhere.
#
# We use synthetic NumPy data for concept explanations and real MNE sample data for EEG demonstrations. Randomness is fixed with `np.random.seed(42)`.

# %% [markdown]
# ## 1. Intuition
# Preprocessing means making data safe, consistent, and interpretable before analysis. Here we handle missing data, duplicates, outliers, encoding, scaling, feature engineering, train/test separation for leakage prevention, and preprocessing pipelines.

# %% [markdown]
# ## 2. Necessary mathematics
#
# Median imputation: replace missing $x_i$ with $\mathrm{median}(x)$.
#
# Standardization: $z_i = \frac{x_i-\mu}{\sigma}$.
#
# IQR fences: lower $=Q_1-1.5\,IQR$, upper $=Q_3+1.5\,IQR$.
#
# Train/test split is not modeling; it demonstrates that transformations should be fit only on training data to prevent leakage.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.append(str(PROJECT_ROOT / 'src'))
from preprocessing_utils import (remove_duplicates, fill_missing_with_median, detect_outliers_zscore, cap_outliers_iqr, build_preprocessing_pipeline, reproducible_train_test_split)

# Build a small reproducible dataset.
df = pd.DataFrame({
    'channel_power': [1.2, 1.5, np.nan, 1.4, 9.8, 1.3, 1.2],
    'signal_std': [0.2, 0.25, 0.23, np.nan, 2.1, 0.22, 0.2],
    'condition': ['rest', 'task', 'rest', 'task', 'rest', 'task', 'rest']
})
# Add a duplicate row intentionally.
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
print(df)

clean = remove_duplicates(df)
clean = fill_missing_with_median(clean, ['channel_power', 'signal_std'])
clean['power_std_ratio'] = clean['channel_power'] / clean['signal_std']
clean['is_outlier'] = detect_outliers_zscore(clean['channel_power'], threshold=2.0)
clean['capped_power'] = cap_outliers_iqr(clean['channel_power'])
print(clean)

train_df, test_df = reproducible_train_test_split(clean, test_size=0.3, random_state=42)
preprocessor = build_preprocessing_pipeline(['channel_power', 'signal_std', 'power_std_ratio'], ['condition'])
train_processed = preprocessor.fit_transform(train_df)
test_processed = preprocessor.transform(test_df)
print('Train transformed shape:', train_processed.shape)
print('Test transformed shape:', test_processed.shape)

plt.figure(figsize=(7, 4))
plt.boxplot(clean['channel_power'])
plt.ylabel('Power')
plt.title('Outlier inspection before clipping')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 8. Interpretation
# The boxplot reveals a strong high-value outlier. Median filling protects against extreme values better than mean filling. The pipeline is fit only on training rows and then applied to test rows, demonstrating leakage-safe preprocessing without training a model.

# %% [markdown]
# ## 4–7. Practical EEG example using real MNE data

# %%
from eeg_preprocessing_utils import load_mne_sample_raw

raw = load_mne_sample_raw(max_minutes=0.2)
data = raw.get_data()[:5] * 1e6
summary = pd.DataFrame({
    'channel': raw.ch_names[:5],
    'mean_uv': data.mean(axis=1),
    'std_uv': data.std(axis=1),
    'ptp_uv': data.ptp(axis=1),
})
summary['std_z_outlier'] = detect_outliers_zscore(summary['std_uv'], threshold=2.0)
print(summary)

plt.figure(figsize=(8, 4))
plt.bar(summary['channel'], summary['std_uv'])
plt.ylabel('Standard deviation (µV)')
plt.title('Simple EEG channel quality summary')
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 9. Exercise
# Add `median_uv` and `range_uv` to the EEG summary DataFrame. Then identify the channel with the largest peak-to-peak amplitude.

# %% [markdown]
# ## 10. References
# - scikit-learn preprocessing guide: https://scikit-learn.org/stable/modules/preprocessing.html
# - MNE-Python tutorials: https://mne.tools/stable/tutorials/index.html

# %% [markdown]
# ## Key Takeaways
# - Clean data before interpretation.
# - Fit preprocessing steps only on training partitions when demonstrating separation.
# - Outlier detection should guide inspection, not automatic rejection.
#
# ## Common Mistakes
# - Fitting scalers on all data before splitting.
# - Treating every outlier as bad data.
# - Forgetting categorical encoding.
# - Mixing preprocessing with model training.
#
# ## Mini Project
# Create a DataFrame of five EEG channels with mean, standard deviation, and band power. Add missing values, clean them, scale numeric columns, and explain every step.
