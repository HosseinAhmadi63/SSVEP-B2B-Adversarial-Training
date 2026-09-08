from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, loss: float | None = None
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError("Probabilities must be samples by classes and align with labels")
    predictions = probabilities.argmax(axis=1)
    result: dict[str, Any] = {
        "samples": int(len(labels)),
        "classes": int(probabilities.shape[1]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc_macro_ovr": float(
            roc_auc_score(
                labels,
                probabilities,
                labels=np.arange(probabilities.shape[1]),
                multi_class="ovr",
                average="macro",
            )
        ),
    }
    if loss is not None:
        result["cross_entropy"] = float(loss)
    return result


def signal_to_noise_ratio(clean: np.ndarray, attacked: np.ndarray) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    noise = np.asarray(attacked, dtype=np.float64) - clean
    signal_power = float(np.square(clean).sum())
    noise_power = float(np.square(noise).sum())
    if noise_power == 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))
