import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ssvep_b2b.paper_results import (
    calculate_summaries,
    load_table3,
    load_table4,
    reproduce_paper_analysis,
    table3_delta_audit,
    table4_consistency_audit,
    verify_paper_results,
)
from ssvep_b2b.plotting import plot_experiment_results, plot_paper_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "results" / "publication" / "source"
TABLE3 = SOURCE_ROOT / "table3.csv"
TABLE4 = SOURCE_ROOT / "table4.csv"


def test_publication_tables_have_the_complete_paper_shape() -> None:
    table3 = load_table3(PROJECT_ROOT)
    table4 = load_table4(PROJECT_ROOT)
    assert table3.shape == (62, 12)
    assert table4.shape == (40, 7)
    assert table3["scenario_order"].nunique() == 31
    assert table3.groupby("scenario_order")["dataset"].nunique().eq(2).all()
    assert table3.groupby("attack_count")["scenario_order"].nunique().to_dict() == {
        1: 5,
        2: 10,
        3: 10,
        4: 5,
        5: 1,
    }


def test_published_endpoint_discrepancies_are_preserved_and_audited() -> None:
    audit = table3_delta_audit(load_table3(TABLE3))
    discrepancies = audit.loc[~audit["is_consistent"]]
    observed = set(
        discrepancies[["scenario_order", "dataset", "metric"]].itertuples(index=False, name=None)
    )
    assert observed == {
        (1, "Lee2019_SSVEP", "auc"),
        (7, "Lee2019_SSVEP", "accuracy"),
        (11, "Lee2019_SSVEP", "auc"),
        (16, "Lee2019_SSVEP", "accuracy"),
        (18, "Nakanishi2015", "accuracy"),
    }
    assert len(audit) == 124


def test_table4_claims_are_compared_without_correcting_the_source() -> None:
    audit = table4_consistency_audit(load_table3(TABLE3), load_table4(TABLE4))
    assert len(audit) == 80
    assert int((~audit["value_match"]).sum()) == 15
    assert int((~audit["scenario_is_extreme"]).sum()) == 15
    claim = audit.loc[
        (audit["dataset"] == "Lee2019_SSVEP")
        & (audit["metric"] == "auc_improvement")
        & (audit["attack_count"] == 4)
        & (audit["claim_role"] == "most_effective")
    ].iloc[0]
    assert claim["printed_value"] == pytest.approx(0.45)
    assert claim["table3_value"] == pytest.approx(0.37)
    assert not claim["value_match"]


def test_derived_summary_reproduces_the_table_means() -> None:
    summary = calculate_summaries(load_table3(TABLE3))
    overall = summary.loc[summary["attack_count"] == "all"].set_index("dataset")
    assert overall.loc[
        "Nakanishi2015", "mean_accuracy_improvement_reported"
    ] == pytest.approx(0.09290322580645162)
    assert overall.loc[
        "Nakanishi2015", "mean_auc_improvement_reported"
    ] == pytest.approx(0.07096774193548387)
    assert overall.loc[
        "Lee2019_SSVEP", "mean_accuracy_improvement_reported"
    ] == pytest.approx(0.23548387096774193)
    assert overall.loc["Lee2019_SSVEP", "mean_auc_improvement_reported"] == pytest.approx(
        0.35
    )


def test_verification_and_reproduction_write_offline_audit_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "results" / "publication" / "source"
    source.mkdir(parents=True)
    shutil.copy2(TABLE3, source / "table3.csv")
    shutil.copy2(TABLE4, source / "table4.csv")
    verification = verify_paper_results(tmp_path)
    assert verification["source_structure_valid"]
    assert verification["table3_endpoint_discrepancy_count"] == 5
    assert verification["table4_value_discrepancy_count"] == 15
    report = reproduce_paper_analysis(tmp_path)
    artifact_paths = {name: Path(path) for name, path in report["artifacts"].items()}
    assert set(artifact_paths) == {
        "table3_delta_audit",
        "table4_consistency_audit",
        "derived_summary",
        "publication_audit",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifact_paths.values())
    saved = json.loads(artifact_paths["publication_audit"].read_text(encoding="utf-8"))
    assert saved["table3_rows"] == 62
    assert saved["table4_claims"] == 80


def test_publication_plots_are_complete_and_close_figures(tmp_path: Path) -> None:
    paths = plot_paper_results(TABLE3, tmp_path)
    assert [path.name for path in paths] == [
        "figure7_improvements.png",
        "figure8_improvement_distributions.png",
        "figure9_computational_time.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert plt.get_fignums() == []


def _prediction_archive(path: Path) -> None:
    labels = np.tile(np.arange(3), 4)
    probabilities = np.full((len(labels), 3), 0.1)
    probabilities[np.arange(len(labels)), labels] = 0.8
    np.savez_compressed(
        path,
        indices=np.arange(len(labels)),
        labels=labels,
        predictions=probabilities.argmax(axis=1),
        probabilities=probabilities,
    )


def test_experiment_plots_discover_predictions_and_attack_examples(tmp_path: Path) -> None:
    records = []
    scenarios = (("fgsm", "FGSM", 1), ("pgd", "PGD", 1), ("fgsm+pgd", "FGSM+PGD", 2))
    for index, (slug, display, attack_count) in enumerate(scenarios, start=1):
        scenario_dir = tmp_path / "Nakanishi2015" / "scenarios" / slug
        scenario_dir.mkdir(parents=True)
        if slug in {"fgsm", "pgd"}:
            _prediction_archive(scenario_dir / "predictions_without_annt.npz")
            _prediction_archive(scenario_dir / "predictions_with_annt.npz")
        clean = np.vstack((np.sin(np.linspace(0, 8, 64)), np.cos(np.linspace(0, 8, 64))))
        perturbation = np.full_like(clean, 0.01 * index)
        np.savez_compressed(
            scenario_dir / "attack_example.npz",
            clean=clean,
            attacked=clean + perturbation,
            perturbation=perturbation,
            sample_index=index,
            snr=30.0 - index,
        )
        records.append(
            {
                "dataset": "Nakanishi2015",
                "scenario": slug,
                "scenario_display": display,
                "attack_count": attack_count,
                "accuracy_without_annt": 0.70,
                "accuracy_with_annt": 0.80 + index * 0.01,
                "accuracy_improvement": 0.10 + index * 0.01,
                "auc_without_annt": 0.75,
                "auc_with_annt": 0.88 + index * 0.01,
                "auc_improvement": 0.13 + index * 0.01,
                "duration_seconds": float(index),
            }
        )
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame.from_records(records).to_csv(metrics_path, index=False)
    output = tmp_path / "figures"
    paths = plot_experiment_results(metrics_path, output)
    assert len(paths) == 6
    assert {path.name for path in paths}.issuperset(
        {
            "roc_comparison_nakanishi2015_fgsm.png",
            "roc_comparison_nakanishi2015_pgd.png",
            "figure1_attack_examples_nakanishi2015.png",
        }
    )
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in paths)
    assert plt.get_fignums() == []
