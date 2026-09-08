from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ssvep_b2b.config import default_config_path, load_config

DATASETS = ("Nakanishi2015", "Lee2019_SSVEP")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ssvep-b2b")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", default=str(default_config_path()))
        return child

    download = command("download")
    download.add_argument("--dataset", nargs="+", choices=DATASETS)
    cache = command("cache")
    cache.add_argument("--dataset", nargs="+", choices=DATASETS)
    cache.add_argument("--force", action="store_true")
    run = command("run")
    run.add_argument("--dataset", nargs="+", choices=DATASETS)
    run.add_argument("--attack", action="append")
    run.add_argument("--force", action="store_true")
    command("verify-paper")
    command("figures")
    command("reproduce-paper-analysis")
    command("smoke")
    run_all = command("run-all")
    run_all.add_argument("--force", action="store_true")
    return parser


def _print(value) -> None:
    if isinstance(value, Path):
        print(value.resolve())
    else:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = load_config(arguments.config)
    if arguments.command == "download":
        from ssvep_b2b.data import download_dataset

        names = arguments.dataset or list(config.section("datasets"))
        for name in names:
            download_dataset(config, name)
        _print({"status": "complete", "datasets": names})
        return 0
    if arguments.command == "cache":
        from ssvep_b2b.data import cache_dataset

        names = arguments.dataset or list(config.section("datasets"))
        paths = [str(cache_dataset(config, name, arguments.force)) for name in names]
        _print({"status": "complete", "caches": paths})
        return 0
    if arguments.command == "run":
        from ssvep_b2b.experiment import run_experiments

        frame = run_experiments(config, arguments.dataset, arguments.attack, arguments.force)
        _print({"status": "complete", "rows": len(frame)})
        return 0
    if arguments.command == "verify-paper":
        from ssvep_b2b.paper_results import verify_paper_results

        _print(verify_paper_results(config))
        return 0
    if arguments.command == "figures":
        from ssvep_b2b.experiment import run_directory
        from ssvep_b2b.plotting import plot_experiment_results

        metrics = run_directory(config) / "metrics.csv"
        outputs = plot_experiment_results(metrics, run_directory(config) / "figures")
        _print({"status": "complete", "figures": [str(path) for path in outputs]})
        return 0
    if arguments.command == "reproduce-paper-analysis":
        from ssvep_b2b.paper_results import reproduce_paper_analysis

        _print(reproduce_paper_analysis(config))
        return 0
    if arguments.command == "smoke":
        from ssvep_b2b.smoke import run_smoke

        _print(
            run_smoke(
                config.project_path("smoke") / "summary.json",
                int(config.section("runtime")["seed"]),
            )
        )
        return 0
    if arguments.command == "run-all":
        from ssvep_b2b.experiment import run_directory, run_experiments
        from ssvep_b2b.paper_results import reproduce_paper_analysis
        from ssvep_b2b.plotting import plot_experiment_results

        reproduce_paper_analysis(config)
        frame = run_experiments(config, force=arguments.force)
        outputs = plot_experiment_results(
            run_directory(config) / "metrics.csv", run_directory(config) / "figures"
        )
        _print(
            {
                "status": "complete",
                "rows": len(frame),
                "figures": [str(path) for path in outputs],
            }
        )
        return 0
    raise RuntimeError(f"Unhandled command {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
