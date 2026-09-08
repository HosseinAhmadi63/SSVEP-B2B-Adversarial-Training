from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from ssvep_b2b.attacks import combine_attacks
from ssvep_b2b.model import build_model
from ssvep_b2b.reproducibility import atomic_json, seed_everything
from ssvep_b2b.training import predict_model, train_model


def synthetic_ssvep(seed: int = 2025) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    sampling_rate = 64.0
    frequencies = np.asarray([5.0, 7.0, 9.0, 11.0])
    labels = np.repeat(np.arange(4), 10)
    time = np.arange(128) / sampling_rate
    signals = np.empty((len(labels), 8, len(time)), dtype=np.float32)
    for index, label in enumerate(labels):
        phases = rng.uniform(0.0, 2.0 * np.pi, size=8)
        weights = rng.uniform(0.7, 1.3, size=8)
        fundamental = np.sin(2.0 * np.pi * frequencies[label] * time[None, :] + phases[:, None])
        harmonic = np.sin(
            4.0 * np.pi * frequencies[label] * time[None, :] + phases[:, None] / 2.0
        )
        noise = rng.normal(0.0, 0.2, size=(8, len(time)))
        signals[index] = weights[:, None] * fundamental + 0.35 * harmonic + noise
    signals -= signals.mean(axis=1, keepdims=True)
    mean = signals.mean(axis=(0, 2), keepdims=True)
    scale = signals.std(axis=(0, 2), keepdims=True)
    signals = (signals - mean) / scale
    return torch.from_numpy(signals).unsqueeze(1), torch.from_numpy(labels).long()


def run_smoke(output: Path, seed: int = 2025) -> dict[str, Any]:
    seed_everything(seed, True)
    device = torch.device("cpu")
    inputs, labels = synthetic_ssvep(seed)
    training_indices = torch.tensor(
        [index for index in range(len(labels)) if index % 5 != 0], dtype=torch.long
    )
    testing_indices = torch.tensor(
        [index for index in range(len(labels)) if index % 5 == 0], dtype=torch.long
    )
    training = TensorDataset(inputs[training_indices], labels[training_indices])
    testing = TensorDataset(inputs[testing_indices], labels[testing_indices])
    baseline = build_model(4, seed)
    train_model(
        baseline,
        training,
        device,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=seed,
        deterministic=True,
        progress=False,
    )
    attacked = combine_attacks(
        baseline,
        inputs,
        labels,
        ["fgsm", "bim", "cw", "mim", "pgd"],
        iteration_overrides={"bim": 2, "cw": 2, "mim": 2, "pgd": 2},
    )
    attacked_training = TensorDataset(attacked[training_indices], labels[training_indices])
    attacked_testing = TensorDataset(attacked[testing_indices], labels[testing_indices])
    robust = build_model(4, seed)
    train_model(
        robust,
        ConcatDataset([training, attacked_training]),
        device,
        epochs=2,
        batch_size=8,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=seed,
        deterministic=True,
        progress=False,
    )
    _, _, clean_metrics = predict_model(baseline, testing, device, 8, seed)
    _, _, attacked_metrics = predict_model(baseline, attacked_testing, device, 8, seed)
    _, _, robust_metrics = predict_model(robust, attacked_testing, device, 8, seed)
    maximum_component = float((attacked - inputs).abs().max())
    result = {
        "status": "complete",
        "samples": len(inputs),
        "classes": 4,
        "attacks": ["fgsm", "bim", "cw", "mim", "pgd"],
        "maximum_combined_absolute_perturbation": maximum_component,
        "clean_baseline": clean_metrics,
        "attacked_without_annt": attacked_metrics,
        "attacked_with_annt": robust_metrics,
    }
    atomic_json(output, result)
    return result
