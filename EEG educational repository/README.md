# EEG Educational Repository

A beginner-to-intermediate Python learning repository for **EEG signal processing**, focused strictly on statistics, preprocessing, visualization, digital signal processing, EEG preprocessing, and feature extraction.

> This repository is educational only. It intentionally contains **no classification models, no deep learning, no SVMs, no neural networks, no training loops, and no prediction systems**. Scikit-learn is used only for preprocessing utilities such as scaling, pipelines, and train/test separation demonstrations.

## Educational Goals

By completing the notebooks in order, you will learn how to:

- Understand core statistics used in EEG analysis.
- Clean tabular and signal-like data safely.
- Visualize EEG and statistical data clearly.
- Understand sampling, Nyquist, aliasing, FFT, PSD, and filtering.
- Load and preprocess EEG with MNE.
- Extract interpretable EEG features such as band power, entropy, and Hjorth parameters.

## Learning Roadmap

Run the notebooks in this order:

1. `notebooks/01_statistics.ipynb` — mean, median, variance, standard deviation, normality, covariance, correlation, z-score, and noise/error analysis.
2. `notebooks/02_data_preprocessing.ipynb` — missing values, duplicates, outliers, encoding, scaling, feature engineering, train/test split, and pipelines without modeling.
3. `notebooks/03_visualization.ipynb` — histograms, boxplots, scatter plots, line plots, heatmaps, and correlation matrices.
4. `notebooks/04_dsp.ipynb` — sampling, Nyquist, aliasing, time/frequency domain, FFT, PSD, and filters.
5. `notebooks/05_eeg_preprocessing.ipynb` — MNE fundamentals, bandpass/notch filtering, artifact removal with ICA, epoching, and baseline correction.
6. `notebooks/06_eeg_feature_extraction.ipynb` — band power, relative power, entropy, Hjorth parameters, and grouped time/frequency-domain features.

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name eeg-education --display-name "Python (EEG Education)"
```

## How to Run the Notebooks

Open the repository in VSCode, PyCharm, JupyterLab, or classic Jupyter Notebook.

```bash
jupyter lab
```

Then run notebooks from `01_statistics.ipynb` to `06_eeg_feature_extraction.ipynb`.

The notebooks are generated from clean `# %%` cell-style source, so they are easy to inspect, edit, and convert.

## Expected Outputs

You should see educational plots such as:

- Normal distribution and z-score demonstrations.
- Missing/outlier detection plots.
- Histograms, boxplots, line plots, heatmaps, and correlation matrices.
- Synthetic sine waves, FFT spectra, PSD curves, and filtered signals.
- MNE raw EEG traces, PSDs, ICA component inspection plots, epochs, and evoked responses.
- Band-power bar charts and feature tables.

Generated images can be saved into the `images/` folder when you adapt the plotting code.

## Repository Structure

```text
eeg_educational_repo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_statistics.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_dsp.ipynb
│   ├── 05_eeg_preprocessing.ipynb
│   └── 06_eeg_feature_extraction.ipynb
├── src/
│   ├── statistics_utils.py
│   ├── preprocessing_utils.py
│   ├── visualization_utils.py
│   ├── dsp_utils.py
│   ├── eeg_preprocessing_utils.py
│   └── feature_extraction_utils.py
├── images/
└── docs/
```

## Reproducibility

All notebooks and examples use fixed random seeds where randomness exists:

```python
np.random.seed(42)
```

Scikit-learn utilities that accept randomness use:

```python
random_state=42
```

This keeps educational outputs consistent across runs.

## Important Notes

- MNE sample data is downloaded by MNE when first requested. The repository itself does not ship large datasets.
- The code avoids model training completely.
- Feature extraction is presented as descriptive signal analysis, not prediction.
