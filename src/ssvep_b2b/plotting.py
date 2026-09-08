from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ssvep_b2b.paper_results import load_table3

plt.switch_backend("Agg")


COLORS = {"Nakanishi2015": "#1565C0", "Lee2019_SSVEP": "#EF6C00"}
DISPLAY_NAMES = {"Nakanishi2015": "Nakanishi2015", "Lee2019_SSVEP": "Lee2019_SSVEP"}
PLOT_COLUMNS = (
    "scenario_order",
    "attack_count",
    "attacks",
    "dataset",
    "accuracy_improvement_reported",
    "auc_improvement_reported",
    "computational_time_normalized",
)


def _attack_display(value: str) -> str:
    names = {"fgsm": "FGSM", "bim": "BIM", "cw": "C&W", "mim": "MIM", "pgd": "PGD"}
    normalized = value.replace("C&W", "cw").replace(" ", "").lower()
    return " + ".join(names.get(item, item.upper()) for item in normalized.split("+"))


def _standardize_experiment_results(metrics_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(metrics_csv).expanduser().resolve())
    if set(PLOT_COLUMNS).issubset(frame.columns):
        result = frame.copy()
    else:
        required = {
            "dataset",
            "attack_count",
            "accuracy_without_annt",
            "accuracy_with_annt",
            "auc_without_annt",
            "auc_with_annt",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Experiment metrics are missing columns: {missing}")
        if "scenario_display" in frame:
            attack_series = frame["scenario_display"].astype(str)
        elif "scenario" in frame:
            attack_series = frame["scenario"].astype(str)
        else:
            raise ValueError("Experiment metrics require scenario or scenario_display")
        result = pd.DataFrame()
        result["dataset"] = frame["dataset"].astype(str)
        result["attack_count"] = frame["attack_count"].astype(int)
        result["attacks"] = attack_series.map(_attack_display)
        if "accuracy_improvement" in frame:
            result["accuracy_improvement_reported"] = frame["accuracy_improvement"]
        else:
            result["accuracy_improvement_reported"] = (
                frame["accuracy_with_annt"] - frame["accuracy_without_annt"]
            )
        if "auc_improvement" in frame:
            result["auc_improvement_reported"] = frame["auc_improvement"]
        else:
            result["auc_improvement_reported"] = (
                frame["auc_with_annt"] - frame["auc_without_annt"]
            )
        if "normalized_computational_time" in frame:
            result["computational_time_normalized"] = frame[
                "normalized_computational_time"
            ]
        elif "computational_time_normalized" in frame:
            result["computational_time_normalized"] = frame[
                "computational_time_normalized"
            ]
        elif "duration_seconds" in frame:
            minimum = float(frame["duration_seconds"].min())
            if minimum <= 0.0:
                raise ValueError("Experiment durations must be positive")
            result["computational_time_normalized"] = frame["duration_seconds"] / minimum * 100.0
        else:
            raise ValueError("Experiment metrics require normalized time or duration_seconds")
        order = {
            value: index + 1
            for index, value in enumerate(result["attacks"].drop_duplicates())
        }
        result["scenario_order"] = result["attacks"].map(order)
    _validate_plot_frame(result)
    return result.sort_values(["scenario_order", "dataset"]).reset_index(drop=True)


def _validate_plot_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(PLOT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Plot data are missing columns: {missing}")
    if frame[list(PLOT_COLUMNS)].isna().any().any():
        raise ValueError("Plot data contain missing values")
    if frame.duplicated(["scenario_order", "dataset"]).any():
        raise ValueError("Plot data contain duplicate scenario and dataset rows")
    counts = frame["attacks"].astype(str).str.count(r"\+") + 1
    if not np.array_equal(counts.to_numpy(), frame["attack_count"].astype(int).to_numpy()):
        raise ValueError("Plot-data attack counts do not match the attack labels")


def _dataset_order(frame: pd.DataFrame) -> list[str]:
    configured = [name for name in DISPLAY_NAMES if name in set(frame["dataset"])]
    additional = [name for name in frame["dataset"].drop_duplicates() if name not in configured]
    return configured + additional


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def _figure7(frame: pd.DataFrame, path: Path) -> None:
    scenario_frame = frame.drop_duplicates("scenario_order").sort_values("scenario_order")
    positions = scenario_frame["scenario_order"].to_numpy()
    labels = scenario_frame["attacks"].to_list()
    preferred = ["Lee2019_SSVEP", "Nakanishi2015"]
    datasets = [name for name in preferred if name in set(frame["dataset"])]
    datasets.extend(name for name in _dataset_order(frame) if name not in datasets)
    figure, accuracy_axis = plt.subplots(figsize=(15, 6.8))
    auc_axis = accuracy_axis.twinx()
    try:
        lines = []
        for index, dataset in enumerate(datasets):
            subset = frame.loc[frame["dataset"] == dataset].sort_values("scenario_order")
            color = COLORS.get(dataset, plt.colormaps["tab10"](index))
            display = DISPLAY_NAMES.get(dataset, dataset)
            accuracy_line = accuracy_axis.plot(
                subset["scenario_order"],
                subset["accuracy_improvement_reported"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4.2,
                label=f"{display} accuracy",
            )[0]
            auc_line = auc_axis.plot(
                subset["scenario_order"],
                subset["auc_improvement_reported"],
                color=color,
                linewidth=2.0,
                linestyle="--",
                marker="s",
                markersize=4.0,
                label=f"{display} AUC",
            )[0]
            lines.extend((accuracy_line, auc_line))
        for boundary in (5.5, 15.5, 25.5, 30.5):
            if positions.min() < boundary < positions.max():
                accuracy_axis.axvline(
                    boundary, color="#90A4AE", linewidth=0.8, linestyle=":"
                )
        accuracy_axis.set_title(
            "Accuracy and AUC improvements with adversarial neural network training"
        )
        accuracy_axis.set_xlabel("Adversarial attack scenario")
        accuracy_axis.set_ylabel("Accuracy improvement")
        auc_axis.set_ylabel("AUC improvement")
        accuracy_axis.set_xticks(positions)
        accuracy_axis.set_xticklabels(labels, rotation=62, ha="right", fontsize=7.5)
        accuracy_axis.legend(
            lines,
            [line.get_label() for line in lines],
            frameon=False,
            ncol=2,
            loc="upper left",
        )
        _style_axis(accuracy_axis)
        auc_axis.spines["top"].set_visible(False)
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _boxplot_panel(
    axis: plt.Axes,
    subset: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    color = "#4C78A8"
    attack_counts = sorted(int(value) for value in subset["attack_count"].unique())
    values = [
        subset.loc[subset["attack_count"] == attack_count, metric].to_numpy()
        for attack_count in attack_counts
    ]
    boxplot = axis.boxplot(
        values,
        positions=attack_counts,
        widths=0.58,
        patch_artist=True,
        showmeans=False,
        medianprops={"color": "#263238", "linewidth": 1.4},
        whiskerprops={"color": color},
        capprops={"color": color},
        flierprops={"markeredgecolor": color, "markerfacecolor": "white"},
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
    axis.set_title(title)
    axis.set_xlabel("Number of simultaneous attacks")
    axis.set_ylabel(ylabel)
    axis.set_xticks(attack_counts)
    _style_axis(axis)


def _figure8(frame: pd.DataFrame, path: Path) -> None:
    datasets = _dataset_order(frame)
    figure, axes = plt.subplots(
        len(datasets), 2, figsize=(12.6, 4.2 * len(datasets)), squeeze=False
    )
    try:
        for row, dataset in enumerate(datasets):
            subset = frame.loc[frame["dataset"] == dataset]
            display = DISPLAY_NAMES.get(dataset, dataset)
            _boxplot_panel(
                axes[row, 0],
                subset,
                "accuracy_improvement_reported",
                "Accuracy improvement" if row == 0 else "",
                display,
            )
            _boxplot_panel(
                axes[row, 1],
                subset,
                "auc_improvement_reported",
                "AUC improvement" if row == 0 else "",
                display,
            )
        figure.suptitle("Distribution of ANNT improvements by attack count", y=1.01)
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _figure9(frame: pd.DataFrame, path: Path) -> None:
    preferred = ["Lee2019_SSVEP", "Nakanishi2015"]
    datasets = [name for name in preferred if name in set(frame["dataset"])]
    datasets.extend(name for name in _dataset_order(frame) if name not in datasets)
    ordered = frame.sort_values("scenario_order")
    pivot = ordered.pivot(
        index="attacks", columns="dataset", values="computational_time_normalized"
    )
    labels = ordered.drop_duplicates("scenario_order")["attacks"].to_list()
    pivot = pivot.reindex(index=labels, columns=datasets)
    values = pivot.to_numpy(dtype=float)
    colors = LinearSegmentedColormap.from_list(
        "paper_time", ["#FFF8E1", "#80CBC4", "#1565C0", "#263238"]
    )
    figure, axis = plt.subplots(figsize=(8.5, 15.5))
    try:
        image = axis.imshow(values, aspect="auto", cmap=colors)
        axis.set_xticks(np.arange(len(datasets)))
        axis.set_xticklabels([DISPLAY_NAMES.get(name, name) for name in datasets])
        axis.set_yticks(np.arange(len(labels)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.set_xlabel("Dataset")
        axis.set_ylabel("Attack scenario")
        axis.set_title("Normalized computational time reported in Table 3")
        midpoint = float(np.nanmin(values) + np.nanmax(values)) / 2.0
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if value > midpoint else "#17202A"
                axis.text(
                    column,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
        colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.035)
        colorbar.set_label("Normalized computational time")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def _file_token(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip(
        "_"
    )


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=np.int64)
        probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) != len(probabilities):
        raise ValueError(f"Invalid prediction archive: {path}")
    return labels, probabilities


def _roc_figure(dataset: str, attack: str, scenario_dir: Path, path: Path) -> None:
    from sklearn.metrics import auc, roc_curve

    prediction_sets = (
        ("Without ANNT", scenario_dir / "predictions_without_annt.npz"),
        ("With ANNT", scenario_dir / "predictions_with_annt.npz"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.1), sharex=True, sharey=True)
    try:
        for axis, (condition, prediction_path) in zip(axes, prediction_sets, strict=True):
            labels, probabilities = _read_predictions(prediction_path)
            color_map = plt.colormaps["turbo"]
            plotted = 0
            for class_index in range(probabilities.shape[1]):
                binary_labels = labels == class_index
                if binary_labels.all() or not binary_labels.any():
                    continue
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    binary_labels.astype(np.int8), probabilities[:, class_index]
                )
                area = auc(false_positive_rate, true_positive_rate)
                color = color_map(class_index / max(probabilities.shape[1] - 1, 1))
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    color=color,
                    linewidth=1.5,
                    label=f"Class {class_index + 1} ({area:.3f})",
                )
                plotted += 1
            axis.plot([0.0, 1.0], [0.0, 1.0], color="#78909C", linestyle="--", linewidth=1.0)
            axis.set_title(condition)
            axis.set_xlabel("False-positive rate")
            axis.set_ylabel("True-positive rate")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.01)
            axis.grid(color="#E0E0E0", linewidth=0.6)
            if plotted:
                columns = 2 if probabilities.shape[1] > 8 else 1
                axis.legend(frameon=False, fontsize=7.2, ncol=columns, loc="lower right")
        figure.suptitle(f"{dataset}: class-wise ROC under {attack.upper()}")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def plot_roc_comparison(run_root: str | Path, output_dir: str | Path) -> list[Path]:
    root = Path(run_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for attack in ("fgsm", "pgd"):
            scenario_dir = dataset_dir / "scenarios" / attack
            required = (
                scenario_dir / "predictions_without_annt.npz",
                scenario_dir / "predictions_with_annt.npz",
            )
            if not all(path.is_file() for path in required):
                continue
            output_path = destination / (
                f"roc_comparison_{_file_token(dataset_dir.name)}_{attack}.png"
            )
            _roc_figure(dataset_dir.name, attack, scenario_dir, output_path)
            paths.append(output_path)
    return paths


def _read_attack_example(path: Path) -> dict[str, np.ndarray | float | int]:
    with np.load(path, allow_pickle=False) as archive:
        values: dict[str, np.ndarray | float | int] = {
            "clean": np.asarray(archive["clean"], dtype=np.float64),
            "attacked": np.asarray(archive["attacked"], dtype=np.float64),
            "perturbation": np.asarray(archive["perturbation"], dtype=np.float64),
            "sample_index": int(np.asarray(archive["sample_index"]).reshape(-1)[0]),
        }
        snr_key = next((key for key in ("snr", "snr_db", "SNR") if key in archive.files), None)
        values["snr"] = (
            float(np.asarray(archive[snr_key]).reshape(-1)[0])
            if snr_key is not None
            else float("nan")
        )
    return values


def _channel_series(value: np.ndarray) -> np.ndarray:
    squeezed = np.asarray(value, dtype=np.float64).squeeze()
    if squeezed.ndim == 0:
        return squeezed.reshape(1, 1)
    if squeezed.ndim == 1:
        return squeezed.reshape(1, -1)
    return squeezed.reshape(-1, squeezed.shape[-1])


def _attack_example_figure(dataset: str, files: dict[str, Path], path: Path) -> None:
    labels = {"fgsm": "FGSM", "pgd": "PGD", "fgsm+pgd": "FGSM + PGD"}
    figure, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex="col")
    try:
        for row, scenario in enumerate(("fgsm", "pgd", "fgsm+pgd")):
            example = _read_attack_example(files[scenario])
            clean = _channel_series(np.asarray(example["clean"]))
            attacked = _channel_series(np.asarray(example["attacked"]))
            perturbation = _channel_series(np.asarray(example["perturbation"]))
            channels = min(len(clean), len(attacked), len(perturbation))
            energies = np.mean(np.square(perturbation[:channels]), axis=1)
            channel = int(np.argmax(energies))
            series = (clean[channel], attacked[channel], perturbation[channel])
            colors = ("#1565C0", "#EF6C00", "#C62828")
            for column, (values, color) in enumerate(zip(series, colors, strict=True)):
                axis = axes[row, column]
                axis.plot(np.arange(len(values)), values, color=color, linewidth=1.0)
                axis.grid(color="#ECEFF1", linewidth=0.55)
                axis.spines[["top", "right"]].set_visible(False)
                if row == 2:
                    axis.set_xlabel("Sample")
                if column == 0:
                    axis.set_ylabel(f"{labels[scenario]}\nAmplitude")
            snr = float(example["snr"])
            snr_text = f", SNR {snr:.2f} dB" if np.isfinite(snr) else ""
            title = f"Sample {int(example['sample_index'])}, channel {channel + 1}{snr_text}"
            axes[row, 1].set_title(title, fontsize=9)
        axes[0, 0].set_title("Clean signal")
        axes[0, 2].set_title("Perturbation")
        figure.suptitle(f"{dataset}: representative adversarial EEG examples")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)


def plot_attack_examples(run_root: str | Path, output_dir: str | Path) -> list[Path]:
    root = Path(run_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    scenarios = ("fgsm", "pgd", "fgsm+pgd")
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = {
            scenario: dataset_dir / "scenarios" / scenario / "attack_example.npz"
            for scenario in scenarios
        }
        if not all(path.is_file() for path in files.values()):
            continue
        output_path = destination / f"figure1_attack_examples_{_file_token(dataset_dir.name)}.png"
        _attack_example_figure(dataset_dir.name, files, output_path)
        paths.append(output_path)
    return paths


def _plot_all(frame: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    _validate_plot_frame(frame)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = [
        destination / "figure7_improvements.png",
        destination / "figure8_improvement_distributions.png",
        destination / "figure9_computational_time.png",
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#263238",
            "axes.titlecolor": "#17202A",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
        }
    ):
        _figure7(frame, paths[0])
        _figure8(frame, paths[1])
        _figure9(frame, paths[2])
    return paths


def plot_paper_results(table3_path: str | Path, output_dir: str | Path) -> list[Path]:
    table3 = load_table3(Path(table3_path))
    return _plot_all(table3, output_dir)


def plot_experiment_results(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    metrics_path = Path(metrics_csv).expanduser().resolve()
    frame = _standardize_experiment_results(metrics_path)
    paths = _plot_all(frame, output_dir)
    paths.extend(plot_roc_comparison(metrics_path.parent, output_dir))
    paths.extend(plot_attack_examples(metrics_path.parent, output_dir))
    return paths
