from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoadedConfig:
    values: dict[str, Any]
    path: Path
    root: Path
    fingerprint: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.values[name]
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section {name} is not a mapping")
        return value

    def project_path(self, key: str) -> Path:
        return self.root / self.section("paths")[key]


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "paper.yaml"


def load_config(path: str | Path | None = None) -> LoadedConfig:
    config_path = Path(path) if path is not None else default_config_path()
    config_path = config_path.expanduser().resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise TypeError("Configuration root must be a mapping")
    validate_config(values)
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return LoadedConfig(values, config_path, config_path.parent.parent, fingerprint)


def validate_config(values: dict[str, Any]) -> None:
    required = {
        "paper",
        "runtime",
        "paths",
        "preprocessing",
        "datasets",
        "split",
        "model",
        "training",
        "attacks",
        "evaluation",
        "scenarios",
    }
    missing = required.difference(values)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    datasets = values["datasets"]
    if set(datasets) != {"Nakanishi2015", "Lee2019_SSVEP"}:
        raise ValueError("The paper configuration must contain both paper datasets")
    scenarios = values["scenarios"]
    if len(scenarios) != 31 or len({tuple(item) for item in scenarios}) != 31:
        raise ValueError("The paper configuration must contain 31 unique attack scenarios")
    attacks = {"fgsm", "bim", "cw", "mim", "pgd"}
    if any(not scenario or not set(scenario).issubset(attacks) for scenario in scenarios):
        raise ValueError("Every scenario must be a nonempty subset of the five paper attacks")
    if float(values["split"]["test_size"]) != 0.2:
        raise ValueError("The paper split must reserve 20 percent for testing")
    for name, dataset in datasets.items():
        expected = dataset["classes"]
        if len(dataset["frequencies"]) != expected:
            raise ValueError(f"{name} frequency count does not equal its class count")
        if len(dataset["subjects"]) * expected * dataset["trials_per_class"] <= 0:
            raise ValueError(f"{name} has an invalid trial declaration")


def scenario_slug(attacks: list[str] | tuple[str, ...]) -> str:
    return "+".join(attacks)


def scenario_display(slug: str) -> str:
    names = {"fgsm": "FGSM", "bim": "BIM", "cw": "C&W", "mim": "MIM", "pgd": "PGD"}
    return "+".join(names[item] for item in slug.split("+"))
