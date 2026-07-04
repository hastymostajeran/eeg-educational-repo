"""MNE-based EEG preprocessing utilities for educational notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import mne
from mne.io import BaseRaw
from mne.preprocessing import ICA


def load_mne_sample_raw(max_minutes: float = 1.0) -> BaseRaw:
    """Load a small EEG-focused segment of the MNE sample auditory/visual data.

    Parameters
    ----------
    max_minutes:
        Maximum duration to keep after loading, in minutes. This keeps examples
        lightweight while still using real MNE sample data.

    Returns
    -------
    BaseRaw
        Preloaded Raw object containing EEG channels from the MNE sample dataset.
    """
    data_path = mne.datasets.sample.data_path()
    raw_path = Path(data_path) / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    raw.pick("eeg")
    raw.crop(tmin=0, tmax=max_minutes * 60)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    return raw


def apply_eeg_filters(raw: BaseRaw, l_freq: float = 1.0, h_freq: float = 40.0, notch_freq: float = 60.0) -> BaseRaw:
    """Apply bandpass and notch filters to a copy of an MNE Raw object.

    Parameters
    ----------
    raw:
        Input MNE Raw EEG object.
    l_freq:
        Lower bandpass cutoff in Hertz.
    h_freq:
        Upper bandpass cutoff in Hertz.
    notch_freq:
        Notch frequency in Hertz, usually 50 or 60 depending on power-line noise.

    Returns
    -------
    BaseRaw
        Filtered copy of the input Raw object.
    """
    filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    filtered.notch_filter(freqs=[notch_freq], verbose=False)
    return filtered


def fit_ica_for_artifact_demo(raw: BaseRaw, n_components: int = 15, random_state: int = 42) -> ICA:
    """Fit ICA for educational artifact-removal demonstration.

    Parameters
    ----------
    raw:
        Filtered MNE Raw object. ICA should normally be fitted to filtered data.
    n_components:
        Number of ICA components to estimate.
    random_state:
        Fixed random state for reproducible ICA decomposition.

    Returns
    -------
    ICA
        Fitted MNE ICA object. The function does not automatically remove
        components because educational notebooks should inspect components first.
    """
    ica = ICA(n_components=n_components, random_state=random_state, max_iter="auto", verbose=False)
    ica.fit(raw, verbose=False)
    return ica


def create_epochs_from_sample(raw: BaseRaw, event_id: Dict[str, int] | None = None) -> Tuple[mne.Epochs, mne.Evoked]:
    """Create epochs and an evoked response from MNE sample events.

    Parameters
    ----------
    raw:
        MNE Raw object loaded from the sample auditory/visual dataset.
    event_id:
        Optional event dictionary. If omitted, the left-auditory condition is used.

    Returns
    -------
    Tuple[mne.Epochs, mne.Evoked]
        Baseline-corrected epochs and their averaged evoked response.
    """
    if event_id is None:
        event_id = {"auditory/left": 1}
    data_path = mne.datasets.sample.data_path()
    events_path = Path(data_path) / "MEG" / "sample" / "sample_audvis_raw-eve.fif"
    events = mne.read_events(events_path)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.5,
        baseline=(None, 0),
        preload=True,
        verbose=False,
    )
    evoked = epochs.average()
    return epochs, evoked
