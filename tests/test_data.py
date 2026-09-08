from copy import deepcopy

import mne
import numpy as np

from ssvep_b2b.config import LoadedConfig, load_config
from ssvep_b2b.data import (
    _prepare_subject,
    create_moabb_dataset,
    reference_signals,
    split_indices,
)


class SyntheticMOABBDataset:
    event_id = {"8.0": 1, "12.0": 2}

    def __init__(self, raw):
        self.raw = raw

    def get_data(self, subjects):
        return {subjects[0]: {"0": {"0": self.raw}}}


def test_reference_signal_shape_and_range():
    references = reference_signals([8.0, 12.0], 256.0, 1024, 2)
    assert references.shape == (2, 4, 1024)
    assert references.dtype == np.float32
    assert np.max(np.abs(references)) <= 1.0


def test_moabb_dataset_constructors_match_paper_cohorts_without_download():
    config = load_config()
    specifications = config.section("datasets")
    nakanishi = create_moabb_dataset("Nakanishi2015", specifications["Nakanishi2015"])
    lee = create_moabb_dataset("Lee2019_SSVEP", specifications["Lee2019_SSVEP"])
    assert nakanishi.subject_list == list(range(1, 10))
    assert nakanishi.n_sessions == 1
    assert nakanishi.interval == [0.15, 4.3]
    assert [float(label) for label in nakanishi.event_id] == specifications["Nakanishi2015"][
        "frequencies"
    ]
    assert lee.subject_list == list(range(1, 55))
    assert lee.n_sessions == 2
    assert lee.interval == [0.0, 4.0]
    assert lee.train_run is True
    assert lee.test_run is False
    assert [float(label) for label in lee.event_id] == specifications["Lee2019_SSVEP"][
        "frequencies"
    ]


def test_stratified_split_is_deterministic_and_balanced():
    config = load_config()
    labels = np.repeat(np.arange(4), 50)
    first_train, first_test = split_indices(config, labels)
    second_train, second_test = split_indices(config, labels)
    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_test, second_test)
    assert len(first_train) == 160
    assert len(first_test) == 40
    assert np.array_equal(np.bincount(labels[first_test]), np.repeat(10, 4))


def test_subject_preparation_applies_continuous_preprocessing_and_epoching():
    sampling_rate = 64
    samples = 384
    time = np.arange(samples) / sampling_rate
    stim = np.zeros(samples)
    stim[64] = 1
    stim[256] = 2
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * 8.0 * time),
            np.sin(2.0 * np.pi * 12.0 * time + 0.3),
            stim,
        ]
    )
    info = mne.create_info(
        ["O1", "O2", "STI 014"], sampling_rate, ["eeg", "eeg", "stim"]
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    base = load_config()
    values = deepcopy(base.values)
    values["datasets"]["Nakanishi2015"].update(
        {
            "channels": 2,
            "classes": 2,
            "sampling_rate": sampling_rate,
            "duration": 1.0,
            "interval": [0.0, 1.0],
            "frequencies": [8.0, 12.0],
        }
    )
    config = LoadedConfig(values, base.path, base.root, "synthetic")
    epochs, labels, metadata, channel_names = _prepare_subject(
        config, "Nakanishi2015", SyntheticMOABBDataset(raw), 1
    )
    assert epochs.shape == (2, 2, 64)
    assert np.array_equal(labels, [0, 1])
    assert channel_names == ("O1", "O2")
    assert np.allclose(epochs.sum(axis=1), 0.0, atol=1.0e-5)
    assert metadata[["subject", "session", "run"]].iloc[0].tolist() == [1, "0", "0"]
