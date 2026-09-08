from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

from ssvep_b2b.attacks import combine_attacks
from ssvep_b2b.config import LoadedConfig, scenario_display, scenario_slug
from ssvep_b2b.data import (
    H5EEGDataset,
    cache_dataset,
    metadata_path,
    read_labels,
    split_indices,
)
from ssvep_b2b.metrics import signal_to_noise_ratio
from ssvep_b2b.model import build_model
from ssvep_b2b.reproducibility import (
    atomic_json,
    atomic_torch_save,
    environment_record,
    resolve_device,
    seed_everything,
)
from ssvep_b2b.training import close_datasets, make_loader, predict_model, train_model

BASELINE_ARTIFACTS = (
    "model.pt",
    "history.csv",
    "predictions_clean.npz",
    "metrics_clean.json",
)

SCENARIO_ARTIFACTS = (
    "attack_example.npz",
    "model_annt.pt",
    "history_annt.csv",
    "predictions_without_annt.npz",
    "predictions_with_annt.npz",
    "predictions_clean_with_annt.npz",
    "metrics.json",
)


def run_directory(config: LoadedConfig) -> Path:
    return config.project_path("results") / config.fingerprint


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _save_predictions(
    path: Path, indices: np.ndarray, labels: np.ndarray, probabilities: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            indices=indices,
            labels=labels,
            predictions=probabilities.argmax(axis=1),
            probabilities=probabilities,
        )
    os.replace(temporary, path)


def _write_run_manifest(config: LoadedConfig) -> None:
    root = run_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "configuration.yaml"
    manifest.write_text(yaml.safe_dump(config.values, sort_keys=False), encoding="utf-8")
    atomic_json(root / "environment.json", environment_record())


def _is_complete(output: Path, artifacts: Sequence[str], fingerprint: str) -> bool:
    marker = output / "complete.json"
    if not marker.is_file() or any(not (output / name).is_file() for name in artifacts):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload == {"config_fingerprint": fingerprint, "status": "complete"}


def _model(config: LoadedConfig, dataset_name: str) -> torch.nn.Module:
    classes = int(config.section("datasets")[dataset_name]["classes"])
    seed = int(config.section("runtime")["seed"])
    return build_model(num_classes=classes, seed=seed)


def _training_arguments(config: LoadedConfig, dataset_name: str) -> dict[str, Any]:
    training = config.section("training")
    dataset = config.section("datasets")[dataset_name]
    runtime = config.section("runtime")
    return {
        "epochs": int(training["epochs"]),
        "batch_size": int(dataset["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "seed": int(runtime["seed"]),
        "deterministic": bool(runtime["deterministic"]),
        "num_workers": int(training["num_workers"]),
    }


def _attack_arguments(config: LoadedConfig) -> dict[str, dict[str, Any]]:
    attacks = config.section("attacks")
    return {
        "fgsm": {"epsilon": float(attacks["fgsm"]["epsilon"])},
        "bim": {
            "epsilon": float(attacks["bim"]["epsilon"]),
            "step_size": float(attacks["bim"]["alpha"]),
            "iterations": int(attacks["bim"]["iterations"]),
        },
        "cw": {
            "epsilon": float(attacks["cw"]["delta_limit"]),
            "c": float(attacks["cw"]["c"]),
            "confidence": float(attacks["cw"]["kappa"]),
            "iterations": int(attacks["cw"]["iterations"]),
            "learning_rate": float(attacks["cw"]["learning_rate"]),
        },
        "mim": {
            "epsilon": float(attacks["mim"]["epsilon"]),
            "step_size": float(attacks["mim"]["alpha"]),
            "iterations": int(attacks["mim"]["iterations"]),
            "decay": float(attacks["mim"]["momentum"]),
        },
        "pgd": {
            "epsilon": float(attacks["pgd"]["epsilon"]),
            "step_size": float(attacks["pgd"]["alpha"]),
            "iterations": int(attacks["pgd"]["iterations"]),
        },
    }


def _load_model(config: LoadedConfig, dataset_name: str, checkpoint: Path, device: torch.device):
    model = _model(config, dataset_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.to(device)
    return model


def ensure_baseline(
    config: LoadedConfig,
    dataset_name: str,
    source: Path,
    training_indices: np.ndarray,
    testing_indices: np.ndarray,
    device: torch.device,
    force: bool = False,
) -> Path:
    output = run_directory(config) / dataset_name / "baseline"
    checkpoint = output / "model.pt"
    marker = output / "complete.json"
    if not force and _is_complete(output, BASELINE_ARTIFACTS, config.fingerprint):
        return checkpoint
    output.mkdir(parents=True, exist_ok=True)
    model = _model(config, dataset_name)
    training_dataset = H5EEGDataset(source, training_indices)
    arguments = _training_arguments(config, dataset_name)
    history = train_model(model, training_dataset, device, **arguments)
    training_dataset.close()
    atomic_torch_save(checkpoint, model.state_dict())
    _atomic_csv(output / "history.csv", pd.DataFrame(history))
    testing_dataset = H5EEGDataset(source, testing_indices)
    labels, probabilities, metrics = predict_model(
        model,
        testing_dataset,
        device,
        arguments["batch_size"],
        arguments["seed"],
        arguments["num_workers"],
    )
    testing_dataset.close()
    _save_predictions(output / "predictions_clean.npz", testing_indices, labels, probabilities)
    atomic_json(output / "metrics_clean.json", metrics)
    atomic_json(marker, {"status": "complete", "config_fingerprint": config.fingerprint})
    return checkpoint


def _make_adversarial_cache(
    config: LoadedConfig,
    dataset_name: str,
    source: Path,
    output: Path,
    model: torch.nn.Module,
    attacks: Sequence[str],
    device: torch.device,
) -> None:
    labels = read_labels(source)
    indices = np.arange(len(labels), dtype=np.int64)
    dataset = H5EEGDataset(source, indices)
    specification = config.section("datasets")[dataset_name]
    batch_size = int(specification["batch_size"])
    runtime = config.section("runtime")
    loader = make_loader(
        dataset,
        batch_size,
        False,
        int(runtime["seed"]),
        int(config.section("training")["num_workers"]),
        device,
    )
    with h5py.File(source, "r") as clean_handle:
        shape = clean_handle["X"].shape
        chunks = clean_handle["X"].chunks
    temporary = output.with_suffix(".h5.tmp")
    if temporary.exists():
        temporary.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    attack_arguments = _attack_arguments(config)
    with h5py.File(temporary, "w") as handle:
        x_store = handle.create_dataset(
            "X", shape=shape, chunks=chunks, compression="lzf", dtype="float32"
        )
        handle.create_dataset("y", data=labels, chunks=True, dtype="int64")
        offset = 0
        for inputs, batch_labels in tqdm(loader, desc=f"Generating {scenario_slug(attacks)}"):
            inputs = inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            attacked = combine_attacks(
                model,
                inputs,
                batch_labels,
                list(attacks),
                configs=attack_arguments,
            )
            count = len(attacked)
            x_store[offset : offset + count] = attacked.squeeze(1).cpu().numpy().astype(np.float32)
            offset += count
        handle.attrs["dataset"] = dataset_name
        handle.attrs["scenario"] = scenario_slug(attacks)
        handle.attrs["combination"] = "independent_delta_sum"
        handle.attrs["generator_model"] = "clean_baseline"
        handle.attrs["complete"] = True
    dataset.close()
    os.replace(temporary, output)


def _save_attack_example(
    dataset_name: str,
    source: Path,
    attacked_source: Path,
    testing_indices: np.ndarray,
    output: Path,
) -> None:
    index = int(testing_indices[0])
    if dataset_name == "Nakanishi2015":
        metadata = pd.read_csv(metadata_path(source))
        subject_four = metadata.loc[metadata["subject"] == 4, "cache_index"]
        if len(subject_four):
            index = int(subject_four.iloc[0])
    with h5py.File(source, "r") as clean_handle, h5py.File(attacked_source, "r") as attacked_handle:
        clean = np.asarray(clean_handle["X"][index], dtype=np.float32)
        attacked = np.asarray(attacked_handle["X"][index], dtype=np.float32)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            clean=clean,
            attacked=attacked,
            perturbation=attacked - clean,
            sample_index=index,
            snr_db=signal_to_noise_ratio(clean, attacked),
        )
    os.replace(temporary, output)


def run_scenario(
    config: LoadedConfig,
    dataset_name: str,
    source: Path,
    training_indices: np.ndarray,
    testing_indices: np.ndarray,
    baseline_checkpoint: Path,
    attacks: Sequence[str],
    device: torch.device,
    force: bool = False,
) -> dict[str, Any]:
    slug = scenario_slug(attacks)
    output = run_directory(config) / dataset_name / "scenarios" / slug
    marker = output / "complete.json"
    metrics_path = output / "metrics.json"
    if not force and _is_complete(output, SCENARIO_ARTIFACTS, config.fingerprint):
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline = _load_model(config, dataset_name, baseline_checkpoint, device)
    adversarial_cache = output / "attacked.h5"
    _make_adversarial_cache(
        config, dataset_name, source, adversarial_cache, baseline, attacks, device
    )
    _save_attack_example(
        dataset_name,
        source,
        adversarial_cache,
        testing_indices,
        output / "attack_example.npz",
    )
    arguments = _training_arguments(config, dataset_name)
    attacked_testing = H5EEGDataset(adversarial_cache, testing_indices)
    labels_without, probabilities_without, metrics_without = predict_model(
        baseline,
        attacked_testing,
        device,
        arguments["batch_size"],
        arguments["seed"],
        arguments["num_workers"],
    )
    attacked_testing.close()
    _save_predictions(
        output / "predictions_without_annt.npz",
        testing_indices,
        labels_without,
        probabilities_without,
    )
    clean_training = H5EEGDataset(source, training_indices)
    attacked_training = H5EEGDataset(adversarial_cache, training_indices)
    robust_training = ConcatDataset([clean_training, attacked_training])
    robust = _model(config, dataset_name)
    history = train_model(robust, robust_training, device, **arguments)
    close_datasets([clean_training, attacked_training])
    robust_checkpoint = output / "model_annt.pt"
    atomic_torch_save(robust_checkpoint, robust.state_dict())
    _atomic_csv(output / "history_annt.csv", pd.DataFrame(history))
    attacked_testing = H5EEGDataset(adversarial_cache, testing_indices)
    labels_with, probabilities_with, metrics_with = predict_model(
        robust,
        attacked_testing,
        device,
        arguments["batch_size"],
        arguments["seed"],
        arguments["num_workers"],
    )
    attacked_testing.close()
    _save_predictions(
        output / "predictions_with_annt.npz",
        testing_indices,
        labels_with,
        probabilities_with,
    )
    clean_testing = H5EEGDataset(source, testing_indices)
    labels_clean, probabilities_clean, metrics_clean = predict_model(
        robust,
        clean_testing,
        device,
        arguments["batch_size"],
        arguments["seed"],
        arguments["num_workers"],
    )
    clean_testing.close()
    _save_predictions(
        output / "predictions_clean_with_annt.npz",
        testing_indices,
        labels_clean,
        probabilities_clean,
    )
    elapsed = time.perf_counter() - started
    result = {
        "dataset": dataset_name,
        "scenario": slug,
        "scenario_display": scenario_display(slug),
        "attack_count": len(attacks),
        "accuracy_without_annt": metrics_without["accuracy"],
        "accuracy_with_annt": metrics_with["accuracy"],
        "accuracy_improvement": metrics_with["accuracy"] - metrics_without["accuracy"],
        "cross_entropy_without_annt": metrics_without["cross_entropy"],
        "cross_entropy_with_annt": metrics_with["cross_entropy"],
        "auc_without_annt": metrics_without["auc_macro_ovr"],
        "auc_with_annt": metrics_with["auc_macro_ovr"],
        "auc_improvement": metrics_with["auc_macro_ovr"] - metrics_without["auc_macro_ovr"],
        "clean_accuracy_with_annt": metrics_clean["accuracy"],
        "clean_auc_with_annt": metrics_clean["auc_macro_ovr"],
        "clean_cross_entropy_with_annt": metrics_clean["cross_entropy"],
        "duration_seconds": elapsed,
    }
    atomic_json(metrics_path, result)
    atomic_json(marker, {"status": "complete", "config_fingerprint": config.fingerprint})
    del baseline, robust, robust_training
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not bool(config.section("training")["keep_attack_cache"]):
        adversarial_cache.unlink(missing_ok=True)
    return result


def completed_results(config: LoadedConfig) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    datasets = list(config.section("datasets"))
    scenarios = [list(item) for item in config.values["scenarios"]]
    for scenario in scenarios:
        slug = scenario_slug(scenario)
        for dataset in datasets:
            output = run_directory(config) / dataset / "scenarios" / slug
            if _is_complete(output, SCENARIO_ARTIFACTS, config.fingerprint):
                records.append(json.loads((output / "metrics.json").read_text(encoding="utf-8")))
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    minimum = float(frame["duration_seconds"].min())
    frame["normalized_computational_time"] = frame["duration_seconds"] / minimum * 100.0
    complete = len(frame) == len(datasets) * len(scenarios)
    frame["paper_run_complete"] = complete
    frame["timing_normalization_scope"] = (
        "complete_paper_table" if complete else "partial_completed_rows"
    )
    _atomic_csv(run_directory(config) / "metrics.csv", frame)
    return frame


def selected_scenarios(config: LoadedConfig, requested: Sequence[str] | None) -> list[list[str]]:
    scenarios = [list(item) for item in config.values["scenarios"]]
    if not requested:
        return scenarios
    available = {frozenset(item): item for item in scenarios}
    selected: list[list[str]] = []
    unknown: list[str] = []
    seen: set[frozenset[str]] = set()
    for value in requested:
        normalized = value.lower().replace("c&w", "cw").replace(" ", "")
        parts = normalized.split("+")
        key = frozenset(parts)
        if "" in parts or len(parts) != len(key) or key not in available:
            unknown.append(value)
        elif key not in seen:
            selected.append(available[key])
            seen.add(key)
    if unknown:
        raise ValueError(f"Unknown attack scenarios: {sorted(unknown)}")
    return selected


def run_experiments(
    config: LoadedConfig,
    datasets: Sequence[str] | None = None,
    attacks: Sequence[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    _write_run_manifest(config)
    runtime = config.section("runtime")
    seed_everything(int(runtime["seed"]), bool(runtime["deterministic"]))
    device = resolve_device(str(runtime["device"]))
    dataset_names = list(datasets) if datasets else list(config.section("datasets"))
    scenarios = selected_scenarios(config, attacks)
    for dataset_name in dataset_names:
        source = cache_dataset(config, dataset_name, force=False)
        labels = read_labels(source)
        training_indices, testing_indices = split_indices(config, labels)
        split_output = run_directory(config) / dataset_name / "split.npz"
        split_output.parent.mkdir(parents=True, exist_ok=True)
        with split_output.open("wb") as stream:
            np.savez_compressed(
                stream, training_indices=training_indices, testing_indices=testing_indices
            )
        baseline = ensure_baseline(
            config,
            dataset_name,
            source,
            training_indices,
            testing_indices,
            device,
            force,
        )
        for scenario in scenarios:
            run_scenario(
                config,
                dataset_name,
                source,
                training_indices,
                testing_indices,
                baseline,
                scenario,
                device,
                force,
            )
    frame = completed_results(config)
    if not frame.empty:
        from ssvep_b2b.comparison import compare_to_publication

        compare_to_publication(config, frame)
    return frame
