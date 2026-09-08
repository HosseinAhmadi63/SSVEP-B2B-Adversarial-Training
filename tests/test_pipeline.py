from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from ssvep_b2b.config import LoadedConfig, load_config
from ssvep_b2b.data import metadata_path, read_labels, split_indices
from ssvep_b2b.experiment import ensure_baseline, run_directory, run_scenario


def test_hdf5_baseline_and_adversarial_training_pipeline(tmp_path: Path):
    base = load_config()
    values = deepcopy(base.values)
    values["paths"]["results"] = "results"
    values["datasets"]["Nakanishi2015"].update(
        {"channels": 4, "classes": 4, "batch_size": 4}
    )
    values["split"]["test_size"] = 0.25
    values["training"].update({"epochs": 1, "keep_attack_cache": False})
    config = LoadedConfig(values, tmp_path / "paper.yaml", tmp_path, "integration")
    rng = np.random.default_rng(2025)
    labels = np.repeat(np.arange(4), 8)
    signals = rng.normal(size=(len(labels), 4, 32)).astype(np.float32)
    for index, label in enumerate(labels):
        signals[index, label, :] += 1.5
    source = tmp_path / "cache.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset(
            "X", data=signals, chunks=(1, 4, 32), compression="lzf", dtype="float32"
        )
        handle.create_dataset("y", data=labels, chunks=True, dtype="int64")
    pd.DataFrame(
        {"cache_index": np.arange(len(labels)), "subject": np.ones(len(labels), dtype=int)}
    ).to_csv(metadata_path(source), index=False)
    training_indices, testing_indices = split_indices(config, read_labels(source))
    checkpoint = ensure_baseline(
        config,
        "Nakanishi2015",
        source,
        training_indices,
        testing_indices,
        torch.device("cpu"),
    )
    result = run_scenario(
        config,
        "Nakanishi2015",
        source,
        training_indices,
        testing_indices,
        checkpoint,
        ["fgsm"],
        torch.device("cpu"),
    )
    scenario = run_directory(config) / "Nakanishi2015" / "scenarios" / "fgsm"
    assert result["dataset"] == "Nakanishi2015"
    assert result["scenario"] == "fgsm"
    assert 0.0 <= result["accuracy_with_annt"] <= 1.0
    assert (scenario / "model_annt.pt").is_file()
    assert (scenario / "predictions_without_annt.npz").is_file()
    assert (scenario / "predictions_with_annt.npz").is_file()
    assert (scenario / "attack_example.npz").is_file()
    assert not (scenario / "attacked.h5").exists()
