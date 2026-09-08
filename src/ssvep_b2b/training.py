from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ssvep_b2b.metrics import classification_metrics
from ssvep_b2b.reproducibility import seed_everything


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=device is not None and device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def train_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    deterministic: bool,
    num_workers: int = 0,
    progress: bool = True,
) -> list[dict[str, float]]:
    seed_everything(seed, deterministic)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = make_loader(dataset, batch_size, True, seed, num_workers, device)
    history: list[dict[str, float]] = []
    epoch_iterator = tqdm(range(1, epochs + 1), desc="Training", disable=not progress)
    for epoch in epoch_iterator:
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            batch_count = len(labels)
            loss_sum += float(loss.detach()) * batch_count
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += batch_count
        record = {
            "epoch": float(epoch),
            "loss": loss_sum / count,
            "accuracy": correct / count,
        }
        history.append(record)
        epoch_iterator.set_postfix(
            loss=f"{record['loss']:.4f}", accuracy=f"{record['accuracy']:.4f}"
        )
    return history


def predict_model(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model.to(device)
    model.eval()
    loader = make_loader(dataset, batch_size, False, seed, num_workers, device)
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            loss = functional.cross_entropy(logits, labels_device, reduction="sum")
            loss_sum += float(loss)
            count += len(labels)
            labels_all.append(labels.numpy())
            probabilities_all.append(probabilities.cpu().numpy())
    labels_array = np.concatenate(labels_all)
    probabilities_array = np.concatenate(probabilities_all)
    metrics = classification_metrics(labels_array, probabilities_array, loss_sum / count)
    return labels_array, probabilities_array, metrics


def close_datasets(datasets: Sequence[Dataset]) -> None:
    for dataset in datasets:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
