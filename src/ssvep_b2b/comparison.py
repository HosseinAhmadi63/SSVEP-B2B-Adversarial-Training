from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ssvep_b2b.config import LoadedConfig
from ssvep_b2b.reproducibility import atomic_json


def _scenario_key(value: str) -> str:
    return value.replace("C&W", "cw").replace(" ", "").lower()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def compare_to_publication(
    config: LoadedConfig, generated: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paper = pd.read_csv(config.project_path("publication_source"))
    actual = generated.copy()
    actual["scenario_key"] = actual["scenario"].map(_scenario_key)
    paper["scenario_key"] = paper["attacks"].map(_scenario_key)
    paper_columns = {
        "accuracy_no_annt": "paper_accuracy_without_annt",
        "accuracy_annt": "paper_accuracy_with_annt",
        "accuracy_improvement_reported": "paper_accuracy_improvement_reported",
        "auc_no_annt": "paper_auc_without_annt",
        "auc_annt": "paper_auc_with_annt",
        "auc_improvement_reported": "paper_auc_improvement_reported",
        "computational_time_normalized": "paper_normalized_computational_time",
    }
    generated_columns = {
        "accuracy_without_annt": "generated_accuracy_without_annt",
        "accuracy_with_annt": "generated_accuracy_with_annt",
        "accuracy_improvement": "generated_accuracy_improvement",
        "auc_without_annt": "generated_auc_without_annt",
        "auc_with_annt": "generated_auc_with_annt",
        "auc_improvement": "generated_auc_improvement",
        "normalized_computational_time": "generated_normalized_computational_time",
    }
    paper_subset = paper[["dataset", "scenario_key", "attacks", *paper_columns]].rename(
        columns=paper_columns
    )
    actual_subset = actual[["dataset", "scenario_key", *generated_columns]].rename(
        columns=generated_columns
    )
    comparison = paper_subset.merge(
        actual_subset, on=["dataset", "scenario_key"], how="inner", validate="one_to_one"
    )
    pairs = (
        ("accuracy_without_annt", "paper_accuracy_without_annt", "generated_accuracy_without_annt"),
        ("accuracy_with_annt", "paper_accuracy_with_annt", "generated_accuracy_with_annt"),
        (
            "accuracy_improvement",
            "paper_accuracy_improvement_reported",
            "generated_accuracy_improvement",
        ),
        ("auc_without_annt", "paper_auc_without_annt", "generated_auc_without_annt"),
        ("auc_with_annt", "paper_auc_with_annt", "generated_auc_with_annt"),
        ("auc_improvement", "paper_auc_improvement_reported", "generated_auc_improvement"),
        (
            "normalized_computational_time",
            "paper_normalized_computational_time",
            "generated_normalized_computational_time",
        ),
    )
    summary: dict[str, Any] = {
        "matched_rows": int(len(comparison)),
        "paper_rows": int(len(paper)),
        "coverage": float(len(comparison) / len(paper)),
        "metrics": {},
    }
    for name, paper_column, generated_column in pairs:
        difference = comparison[generated_column] - comparison[paper_column]
        comparison[f"difference_{name}"] = difference
        summary["metrics"][name] = {
            "mean_absolute_error": float(difference.abs().mean()),
            "root_mean_squared_error": float((difference.square().mean()) ** 0.5),
            "mean_signed_error": float(difference.mean()),
        }
    output = config.project_path("results") / config.fingerprint
    _atomic_csv(output / "comparison_to_paper.csv", comparison)
    atomic_json(output / "comparison_summary.json", summary)
    return comparison, summary
