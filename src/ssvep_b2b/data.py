from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from moabb.datasets import Lee2019_SSVEP, Nakanishi2015
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ssvep_b2b.config import LoadedConfig

DATASET_TYPES = {
    "Nakanishi2015": Nakanishi2015,
    "Lee2019_SSVEP": Lee2019_SSVEP,
}


def create_moabb_dataset(name: str, specification: dict[str, Any]):
    constructor = specification.get("constructor") or {}
    return DATASET_TYPES[name](**constructor)


def configure_download_path(config: LoadedConfig) -> Path:
    raw = config.project_path("data") / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    resolved = str(raw.resolve())
    mne.set_config("MNE_DATA", resolved, set_env=True)
    for signifier in ("NAKANISHI", "LEE2019-SSVEP"):
        os.environ[f"MNE_DATASETS_{signifier}_PATH"] = resolved
    return raw


def download_dataset(config: LoadedConfig, name: str) -> None:
    specification = config.section("datasets")[name]
    dataset = create_moabb_dataset(name, specification)
    raw = configure_download_path(config)
    dataset.download(
        subject_list=list(specification["subjects"]),
        path=str(raw),
        force_update=False,
        update_path=True,
        verbose=False,
    )


def cache_path(config: LoadedConfig, name: str) -> Path:
    return config.project_path("cache") / f"{name}_{config.fingerprint}.h5"


def metadata_path(cache: Path) -> Path:
    return cache.with_suffix(".metadata.csv")


def _label_indices(labels: Sequence[Any], frequencies: Sequence[float], dataset: Any) -> np.ndarray:
    frequency_array = np.asarray(frequencies, dtype=np.float64)
    event_order = {
        str(key): index
        for index, key in enumerate(sorted(dataset.event_id, key=dataset.event_id.get))
    }
    indices: list[int] = []
    for label in labels:
        text = str(label)
        try:
            value = float(text)
            difference = np.abs(frequency_array - value)
            index = int(difference.argmin())
            if float(difference[index]) > 0.02:
                raise ValueError
        except ValueError:
            if text not in event_order:
                raise ValueError(f"Cannot map event label {text}") from None
            index = event_order[text]
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def reference_signals(
    frequencies: Sequence[float], sampling_rate: float, n_times: int, harmonics: int
) -> np.ndarray:
    time = np.arange(n_times, dtype=np.float64) / sampling_rate
    references = []
    for frequency in frequencies:
        components = []
        for harmonic in range(1, harmonics + 1):
            phase = 2.0 * np.pi * harmonic * float(frequency) * time
            components.extend((np.sin(phase), np.cos(phase)))
        references.append(np.stack(components))
    return np.asarray(references, dtype=np.float32)


def _prepare_subject(
    config: LoadedConfig, name: str, dataset: Any, subject: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, tuple[str, ...]]:
    specification = config.section("datasets")[name]
    preprocessing = config.section("preprocessing")
    expected_channels = int(specification["channels"])
    expected_times = int(round(float(specification["duration"]) * specification["sampling_rate"]))
    subject_data = dataset.get_data(subjects=[subject])[subject]
    blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    channel_names: tuple[str, ...] | None = None
    code_to_label = {int(code): str(label) for label, code in dataset.event_id.items()}
    for session, runs in subject_data.items():
        for run, source_raw in runs.items():
            raw = source_raw.copy().load_data()
            if not np.isclose(float(raw.info["sfreq"]), float(specification["sampling_rate"])):
                raise ValueError(f"{name} subject {subject} has an unexpected sampling rate")
            stim_picks = mne.pick_types(raw.info, eeg=False, stim=True)
            if len(stim_picks):
                events = mne.find_events(
                    raw,
                    shortest_event=1,
                    consecutive=True,
                    initial_event=True,
                    verbose=False,
                )
            else:
                events, _ = mne.events_from_annotations(
                    raw, event_id=dataset.event_id, verbose=False
                )
            events = events[np.isin(events[:, 2], list(code_to_label))]
            raw.set_montage(
                str(preprocessing["montage"]),
                match_case=False,
                on_missing="ignore",
                verbose=False,
            )
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)
            raw.filter(
                l_freq=float(preprocessing["fmin"]),
                h_freq=float(preprocessing["fmax"]),
                picks="eeg",
                method=str(preprocessing["filter_method"]),
                phase=str(preprocessing["filter_phase"]),
                fir_window=str(preprocessing["fir_window"]),
                fir_design=str(preprocessing["fir_design"]),
                skip_by_annotation=(),
                verbose=False,
            )
            tmin = float(specification["interval"][0])
            tmax = tmin + (expected_times - 1) / float(specification["sampling_rate"])
            epochs_object = mne.Epochs(
                raw,
                events,
                event_id=dataset.event_id,
                tmin=tmin,
                tmax=tmax,
                baseline=None,
                picks="eeg",
                preload=True,
                reject_by_annotation=True,
                event_repeated="drop",
                on_missing="ignore",
                verbose=False,
            )
            block = epochs_object.get_data(copy=True).astype(np.float32)
            if block.shape[1:] != (expected_channels, expected_times):
                raise ValueError(
                    f"{name} subject {subject} produced {block.shape}; expected trials x "
                    f"{expected_channels} x {expected_times}"
                )
            current_channel_names = tuple(epochs_object.ch_names)
            if channel_names is None:
                channel_names = current_channel_names
            elif current_channel_names != channel_names:
                raise ValueError(f"{name} subject {subject} has inconsistent channel order")
            original_labels = np.asarray(
                [code_to_label[int(code)] for code in epochs_object.events[:, 2]]
            )
            indices = _label_indices(original_labels, specification["frequencies"], dataset)
            blocks.append(block)
            label_blocks.append(indices)
            frames.append(
                pd.DataFrame(
                    {
                        "subject": subject,
                        "session": str(session),
                        "run": str(run),
                        "run_trial": np.arange(len(block), dtype=np.int64),
                        "event_sample": epochs_object.events[:, 0],
                        "original_label": original_labels,
                        "class_index": indices,
                    }
                )
            )
    epochs = np.concatenate(blocks, axis=0)
    indices = np.concatenate(label_blocks)
    frame = pd.concat(frames, ignore_index=True)
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    if channel_names is None:
        raise ValueError(f"{name} subject {subject} produced no usable runs")
    return epochs, indices, frame, channel_names


def cache_dataset(config: LoadedConfig, name: str, force: bool = False) -> Path:
    destination = cache_path(config, name)
    if destination.exists() and metadata_path(destination).exists() and not force:
        validate_cache(config, name, destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    configure_download_path(config)
    specification = config.section("datasets")[name]
    dataset = create_moabb_dataset(name, specification)
    expected_times = int(round(float(specification["duration"]) * specification["sampling_rate"]))
    expected_channels = int(specification["channels"])
    temporary = destination.with_suffix(".h5.tmp")
    if temporary.exists():
        temporary.unlink()
    sum_channels = np.zeros(expected_channels, dtype=np.float64)
    square_channels = np.zeros(expected_channels, dtype=np.float64)
    observation_count = 0
    frames: list[pd.DataFrame] = []
    channel_names: tuple[str, ...] | None = None
    with h5py.File(temporary, "w") as handle:
        x_store = handle.create_dataset(
            "X",
            shape=(0, expected_channels, expected_times),
            maxshape=(None, expected_channels, expected_times),
            chunks=(1, expected_channels, expected_times),
            compression="lzf",
            dtype="float32",
        )
        y_store = handle.create_dataset(
            "y", shape=(0,), maxshape=(None,), chunks=True, dtype="int64"
        )
        offset = 0
        for subject in tqdm(specification["subjects"], desc=f"Caching {name}"):
            epochs, labels, frame, current_channel_names = _prepare_subject(
                config, name, dataset, int(subject)
            )
            if channel_names is None:
                channel_names = current_channel_names
            elif current_channel_names != channel_names:
                raise ValueError(f"{name} has inconsistent channel order between subjects")
            count = len(epochs)
            x_store.resize(offset + count, axis=0)
            y_store.resize(offset + count, axis=0)
            x_store[offset : offset + count] = epochs
            y_store[offset : offset + count] = labels
            frame.insert(0, "cache_index", np.arange(offset, offset + count, dtype=np.int64))
            frames.append(frame)
            for start in range(0, count, 8):
                block = epochs[start : start + 8].astype(np.float64)
                sum_channels += block.sum(axis=(0, 2))
                square_channels += np.square(block).sum(axis=(0, 2))
            observation_count += epochs.shape[0] * epochs.shape[2]
            offset += count
        mean = sum_channels / observation_count
        variance = square_channels / observation_count - np.square(mean)
        epsilon = float(config.section("preprocessing")["normalization_epsilon"])
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale[scale < epsilon] = 1.0
        for start in tqdm(range(0, offset, 8), desc=f"Normalizing {name}"):
            stop = min(start + 8, offset)
            block = x_store[start:stop]
            block = (block - mean[None, :, None]) / scale[None, :, None]
            x_store[start:stop] = block.astype(np.float32)
        handle.create_dataset("channel_mean", data=mean)
        handle.create_dataset("channel_scale", data=scale)
        handle.create_dataset(
            "channel_names",
            data=np.asarray(channel_names, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        handle.create_dataset(
            "reference_signals",
            data=reference_signals(
                specification["frequencies"],
                float(specification["sampling_rate"]),
                expected_times,
                int(config.section("preprocessing")["reference_harmonics"]),
            ),
            compression="lzf",
        )
        handle.attrs["dataset"] = name
        handle.attrs["config_fingerprint"] = config.fingerprint
        handle.attrs["configuration"] = json.dumps(config.values, sort_keys=True)
    combined = pd.concat(frames, ignore_index=True)
    metadata_temporary = metadata_path(destination).with_suffix(".csv.tmp")
    combined.to_csv(metadata_temporary, index=False)
    os.replace(temporary, destination)
    os.replace(metadata_temporary, metadata_path(destination))
    validate_cache(config, name, destination)
    return destination


def validate_cache(config: LoadedConfig, name: str, path: Path | None = None) -> None:
    source = path or cache_path(config, name)
    specification = config.section("datasets")[name]
    expected_trials = (
        len(specification["subjects"])
        * int(specification["classes"])
        * int(specification["trials_per_class"])
    )
    expected_times = int(round(float(specification["duration"]) * specification["sampling_rate"]))
    with h5py.File(source, "r") as handle:
        expected_shape = (expected_trials, int(specification["channels"]), expected_times)
        if handle["X"].shape != expected_shape:
            raise ValueError(
                f"{name} cache shape {handle['X'].shape} does not equal {expected_shape}"
            )
        labels = handle["y"][:]
        counts = np.bincount(labels, minlength=int(specification["classes"]))
        expected_per_class = len(specification["subjects"]) * int(
            specification["trials_per_class"]
        )
        if not np.all(counts == expected_per_class):
            raise ValueError(f"{name} class counts {counts.tolist()} are invalid")
        if not np.isfinite(handle["X"][: min(16, expected_trials)]).all():
            raise ValueError(f"{name} cache contains non-finite values")
    metadata_source = metadata_path(source)
    if not metadata_source.is_file():
        raise ValueError(f"{name} cache metadata is missing")
    metadata = pd.read_csv(metadata_source)
    required_columns = {"cache_index", "class_index"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"{name} cache metadata is missing columns {sorted(missing_columns)}")
    if len(metadata) != expected_trials:
        raise ValueError(
            f"{name} cache metadata has {len(metadata)} rows; expected {expected_trials}"
        )
    cache_indices = metadata["cache_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(cache_indices, np.arange(expected_trials, dtype=np.int64)):
        raise ValueError(f"{name} cache metadata indices are invalid")
    metadata_labels = metadata["class_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(metadata_labels, labels):
        raise ValueError(f"{name} cache metadata labels do not match the HDF5 labels")


def read_labels(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["y"][:], dtype=np.int64)


def split_indices(config: LoadedConfig, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels), dtype=np.int64)
    runtime = config.section("runtime")
    split = config.section("split")
    training, testing = train_test_split(
        indices,
        test_size=float(split["test_size"]),
        random_state=int(runtime["seed"]),
        shuffle=True,
        stratify=labels if split["stratified"] else None,
    )
    return np.sort(training), np.sort(testing)


class H5EEGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path, indices: Sequence[int], key: str = "X") -> None:
        self.path = Path(path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.key = key
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._file()
        row = int(self.indices[index])
        x = torch.from_numpy(np.asarray(handle[self.key][row], dtype=np.float32)).unsqueeze(0)
        y = torch.tensor(int(handle["y"][row]), dtype=torch.long)
        return x, y

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        self.close()
