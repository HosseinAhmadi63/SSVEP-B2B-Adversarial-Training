from __future__ import annotations

import json
import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from ssvep_b2b.config import LoadedConfig, load_config

TABLE3_COLUMNS = (
    "scenario_order",
    "attack_count",
    "attacks",
    "dataset_code",
    "dataset",
    "accuracy_no_annt",
    "accuracy_annt",
    "accuracy_improvement_reported",
    "auc_no_annt",
    "auc_annt",
    "auc_improvement_reported",
    "computational_time_normalized",
)
TABLE4_COLUMNS = (
    "dataset",
    "metric",
    "attack_count",
    "least_effective_attack",
    "least_effective_value",
    "most_effective_attack",
    "most_effective_value",
)
DATASET_CODES = {"Nakanishi2015": "A", "Lee2019_SSVEP": "B"}
TABLE4_METRICS = {
    "accuracy_annt": ("accuracy_annt", "performance"),
    "auc_annt": ("auc_annt", "performance"),
    "accuracy_improvement": ("accuracy_improvement_reported", "improvement"),
    "auc_improvement": ("auc_improvement_reported", "improvement"),
    "computational_time": ("computational_time_normalized", "improvement"),
}
ConfigInput = LoadedConfig | str | Path | None


def _publication_paths(config_or_root: ConfigInput) -> tuple[Path, Path, Path]:
    if isinstance(config_or_root, LoadedConfig):
        table3 = config_or_root.project_path("publication_source")
        generated = config_or_root.project_path("publication_generated")
        return table3, table3.with_name("table4.csv"), generated
    if config_or_root is None:
        return _publication_paths(load_config())
    candidate = Path(config_or_root).expanduser().resolve()
    if candidate.suffix.lower() in {".yaml", ".yml"}:
        return _publication_paths(load_config(candidate))
    if candidate.suffix.lower() == ".csv":
        table3 = candidate
        generated = table3.parent.parent / "generated"
        return table3, table3.with_name("table4.csv"), generated
    table3 = candidate / "results" / "publication" / "source" / "table3.csv"
    generated = candidate / "results" / "publication" / "generated"
    return table3, table3.with_name("table4.csv"), generated


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(columns))
    if missing or extra:
        message = f"{table} columns differ from the publication schema"
        raise ValueError(f"{message}: missing={missing}, extra={extra}")
    if frame[list(columns)].isna().any().any():
        raise ValueError(f"{table} contains missing values")


def _validate_table3(frame: pd.DataFrame) -> None:
    _require_columns(frame, TABLE3_COLUMNS, "Table 3")
    if len(frame) != 62:
        raise ValueError(f"Table 3 must contain 62 rows, found {len(frame)}")
    orders = sorted(int(value) for value in frame["scenario_order"].unique())
    if orders != list(range(1, 32)):
        raise ValueError("Table 3 must contain scenario orders 1 through 31")
    expected_datasets = set(DATASET_CODES)
    for order, group in frame.groupby("scenario_order", sort=True):
        if len(group) != 2 or set(group["dataset"]) != expected_datasets:
            raise ValueError(f"Table 3 scenario {order} must contain both datasets exactly once")
        if group["attacks"].nunique() != 1 or group["attack_count"].nunique() != 1:
            raise ValueError(f"Table 3 scenario {order} has inconsistent attack metadata")
    if frame.duplicated(["scenario_order", "dataset"]).any():
        raise ValueError("Table 3 contains duplicate scenario and dataset rows")
    codes = dict(frame[["dataset", "dataset_code"]].drop_duplicates().to_numpy())
    if codes != DATASET_CODES:
        raise ValueError(f"Table 3 dataset codes are invalid: {codes}")
    declared_counts = frame["attacks"].str.count(r"\+") + 1
    if not declared_counts.equals(frame["attack_count"].astype(int)):
        raise ValueError("Table 3 attack counts do not match the attack labels")
    bounded = [
        "accuracy_no_annt",
        "accuracy_annt",
        "accuracy_improvement_reported",
        "auc_no_annt",
        "auc_annt",
        "auc_improvement_reported",
    ]
    if not frame[bounded].apply(lambda column: column.between(0.0, 1.0).all()).all():
        raise ValueError("Table 3 accuracy and AUC values must be within zero and one")
    if not frame["computational_time_normalized"].gt(0.0).all():
        raise ValueError("Table 3 computational times must be positive")


def _validate_table4(frame: pd.DataFrame) -> None:
    _require_columns(frame, TABLE4_COLUMNS, "Table 4")
    if len(frame) != 40:
        raise ValueError(f"Table 4 must contain 40 rows, found {len(frame)}")
    if set(frame["dataset"]) != set(DATASET_CODES):
        raise ValueError("Table 4 must contain both paper datasets")
    if set(frame["metric"]) != set(TABLE4_METRICS):
        raise ValueError("Table 4 does not contain the five published metrics")
    expected_counts = {1, 2, 3, 4}
    for key, group in frame.groupby(["dataset", "metric"], sort=False):
        if len(group) != 4 or set(group["attack_count"].astype(int)) != expected_counts:
            raise ValueError(f"Table 4 group {key} must contain attack counts one through four")
    if frame.duplicated(["dataset", "metric", "attack_count"]).any():
        raise ValueError("Table 4 contains duplicate dataset, metric, and attack-count rows")


def load_table3(config_or_root: ConfigInput = None) -> pd.DataFrame:
    table3_path, _, _ = _publication_paths(config_or_root)
    frame = pd.read_csv(table3_path)
    _validate_table3(frame)
    return frame


def load_table4(config_or_root: ConfigInput = None) -> pd.DataFrame:
    if isinstance(config_or_root, str | Path):
        candidate = Path(config_or_root).expanduser().resolve()
        if candidate.suffix.lower() == ".csv" and candidate.name == "table4.csv":
            frame = pd.read_csv(candidate)
            _validate_table4(frame)
            return frame
    _, table4_path, _ = _publication_paths(config_or_root)
    frame = pd.read_csv(table4_path)
    _validate_table4(frame)
    return frame


def _published_difference(after: float, before: float) -> Decimal:
    return (Decimal(str(after)) - Decimal(str(before))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def table3_delta_audit(table3: pd.DataFrame) -> pd.DataFrame:
    _validate_table3(table3)
    records: list[dict[str, Any]] = []
    specifications = (
        ("accuracy", "accuracy_no_annt", "accuracy_annt", "accuracy_improvement_reported"),
        ("auc", "auc_no_annt", "auc_annt", "auc_improvement_reported"),
    )
    for row in table3.itertuples(index=False):
        values = row._asdict()
        for metric, before_column, after_column, reported_column in specifications:
            before = float(values[before_column])
            after = float(values[after_column])
            reported = Decimal(str(values[reported_column])).quantize(Decimal("0.01"))
            recomputed = _published_difference(after, before)
            difference = (reported - recomputed).quantize(Decimal("0.01"))
            records.append(
                {
                    "scenario_order": int(values["scenario_order"]),
                    "attack_count": int(values["attack_count"]),
                    "attacks": str(values["attacks"]),
                    "dataset_code": str(values["dataset_code"]),
                    "dataset": str(values["dataset"]),
                    "metric": metric,
                    "without_annt": before,
                    "with_annt": after,
                    "reported_delta": float(reported),
                    "recomputed_delta": float(recomputed),
                    "reported_minus_recomputed": float(difference),
                    "is_consistent": reported == recomputed,
                }
            )
    return pd.DataFrame.from_records(records)


def table4_consistency_audit(table3: pd.DataFrame, table4: pd.DataFrame) -> pd.DataFrame:
    _validate_table3(table3)
    _validate_table4(table4)
    lookup = table3.set_index(["dataset", "attack_count", "attacks"])
    records: list[dict[str, Any]] = []
    for row in table4.itertuples(index=False):
        metric_column, metric_kind = TABLE4_METRICS[row.metric]
        group = table3.loc[
            (table3["dataset"] == row.dataset)
            & (table3["attack_count"] == row.attack_count)
        ]
        for role in ("least_effective", "most_effective"):
            selected_attack = str(getattr(row, f"{role}_attack"))
            printed_value = float(getattr(row, f"{role}_value"))
            key = (row.dataset, row.attack_count, selected_attack)
            if key not in lookup.index:
                table3_value = float("nan")
                value_match = False
                scenario_is_extreme = False
            else:
                selected = lookup.loc[key]
                if isinstance(selected, pd.DataFrame):
                    raise ValueError(f"Table 3 lookup is not unique for {key}")
                table3_value = float(selected[metric_column])
                value_match = Decimal(str(printed_value)) == Decimal(str(table3_value))
                if metric_kind == "performance":
                    expected_extreme = (
                        float(group[metric_column].max())
                        if role == "least_effective"
                        else float(group[metric_column].min())
                    )
                else:
                    expected_extreme = (
                        float(group[metric_column].min())
                        if role == "least_effective"
                        else float(group[metric_column].max())
                    )
                scenario_is_extreme = Decimal(str(table3_value)) == Decimal(
                    str(expected_extreme)
                )
            if metric_kind == "performance":
                expected_extreme = (
                    float(group[metric_column].max())
                    if role == "least_effective"
                    else float(group[metric_column].min())
                )
            else:
                expected_extreme = (
                    float(group[metric_column].min())
                    if role == "least_effective"
                    else float(group[metric_column].max())
                )
            expected_attacks = " | ".join(
                group.loc[group[metric_column] == expected_extreme, "attacks"].astype(str)
            )
            difference = (
                float((Decimal(str(printed_value)) - Decimal(str(table3_value))).normalize())
                if pd.notna(table3_value)
                else float("nan")
            )
            records.append(
                {
                    "dataset": str(row.dataset),
                    "metric": str(row.metric),
                    "attack_count": int(row.attack_count),
                    "claim_role": role,
                    "selected_attack": selected_attack,
                    "printed_value": printed_value,
                    "table3_value": table3_value,
                    "printed_minus_table3": difference,
                    "value_match": bool(value_match),
                    "expected_extreme_value": expected_extreme,
                    "expected_attacks": expected_attacks,
                    "scenario_is_extreme": bool(scenario_is_extreme),
                }
            )
    return pd.DataFrame.from_records(records)


def calculate_summaries(table3: pd.DataFrame) -> pd.DataFrame:
    _validate_table3(table3)
    working = table3.copy()
    working["accuracy_improvement_recomputed"] = (
        working["accuracy_annt"] - working["accuracy_no_annt"]
    )
    working["auc_improvement_recomputed"] = working["auc_annt"] - working["auc_no_annt"]
    records: list[dict[str, Any]] = []
    value_columns = (
        "accuracy_no_annt",
        "accuracy_annt",
        "accuracy_improvement_reported",
        "accuracy_improvement_recomputed",
        "auc_no_annt",
        "auc_annt",
        "auc_improvement_reported",
        "auc_improvement_recomputed",
        "computational_time_normalized",
    )
    for dataset, dataset_group in working.groupby("dataset", sort=False):
        groups: list[tuple[str, pd.DataFrame]] = [
            (str(int(count)), group)
            for count, group in dataset_group.groupby("attack_count", sort=True)
        ]
        groups.append(("all", dataset_group))
        for attack_count, group in groups:
            record: dict[str, Any] = {
                "dataset": str(dataset),
                "attack_count": attack_count,
                "scenario_count": int(len(group)),
            }
            for column in value_columns:
                record[f"mean_{column}"] = float(group[column].mean())
            record["minimum_computational_time_normalized"] = float(
                group["computational_time_normalized"].min()
            )
            record["maximum_computational_time_normalized"] = float(
                group["computational_time_normalized"].max()
            )
            records.append(record)
    return pd.DataFrame.from_records(records)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _verification_report(
    table3_path: Path,
    table4_path: Path,
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    delta_audit: pd.DataFrame,
    table4_audit: pd.DataFrame,
) -> dict[str, Any]:
    endpoint_discrepancies = delta_audit.loc[~delta_audit["is_consistent"]]
    table4_value_discrepancies = table4_audit.loc[~table4_audit["value_match"]]
    table4_extreme_discrepancies = table4_audit.loc[~table4_audit["scenario_is_extreme"]]
    return {
        "source_structure_valid": True,
        "published_values_internally_consistent": bool(
            endpoint_discrepancies.empty
            and table4_value_discrepancies.empty
            and table4_extreme_discrepancies.empty
        ),
        "table3_path": str(table3_path),
        "table4_path": str(table4_path),
        "table3_rows": int(len(table3)),
        "table3_scenarios": int(table3["scenario_order"].nunique()),
        "table3_dataset_rows_per_scenario": 2,
        "table3_endpoint_claims": int(len(delta_audit)),
        "table3_endpoint_discrepancy_count": int(len(endpoint_discrepancies)),
        "table3_endpoint_discrepancies": _json_records(endpoint_discrepancies),
        "table4_rows": int(len(table4)),
        "table4_claims": int(len(table4_audit)),
        "table4_value_discrepancy_count": int(len(table4_value_discrepancies)),
        "table4_value_discrepancies": _json_records(table4_value_discrepancies),
        "table4_extreme_discrepancy_count": int(len(table4_extreme_discrepancies)),
        "table4_extreme_discrepancies": _json_records(table4_extreme_discrepancies),
    }


def verify_paper_results(config_or_root: ConfigInput = None) -> dict[str, Any]:
    table3_path, table4_path, _ = _publication_paths(config_or_root)
    table3 = load_table3(table3_path)
    table4 = load_table4(table4_path)
    delta_audit = table3_delta_audit(table3)
    table4_audit = table4_consistency_audit(table3, table4)
    return _verification_report(
        table3_path, table4_path, table3, table4, delta_audit, table4_audit
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def reproduce_paper_analysis(config_or_root: ConfigInput = None) -> dict[str, Any]:
    table3_path, table4_path, output_dir = _publication_paths(config_or_root)
    table3 = load_table3(table3_path)
    table4 = load_table4(table4_path)
    delta_audit = table3_delta_audit(table3)
    table4_audit = table4_consistency_audit(table3, table4)
    summary = calculate_summaries(table3)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "table3_delta_audit": output_dir / "table3_delta_audit.csv",
        "table4_consistency_audit": output_dir / "table4_consistency_audit.csv",
        "derived_summary": output_dir / "derived_summary.csv",
        "publication_audit": output_dir / "publication_audit.json",
    }
    _atomic_csv(artifacts["table3_delta_audit"], delta_audit)
    _atomic_csv(artifacts["table4_consistency_audit"], table4_audit)
    _atomic_csv(artifacts["derived_summary"], summary)
    from ssvep_b2b.plotting import plot_paper_results

    figure_paths = plot_paper_results(table3_path, output_dir)
    report = _verification_report(
        table3_path, table4_path, table3, table4, delta_audit, table4_audit
    )
    report["derived_summary"] = _json_records(summary)
    report["artifacts"] = {name: str(path) for name, path in artifacts.items()}
    report["figures"] = [str(path) for path in figure_paths]
    _atomic_json(artifacts["publication_audit"], report)
    return report
