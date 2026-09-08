# PyCharm setup

## Open the project

Open `SSVEP-B2B-Adversarial-Training` as the project root. The root must contain `pyproject.toml`, `configs/`, `scripts/`, and `src/`.

Use Python 3.11. The shared run configurations resolve paths from `$PROJECT_DIR$` and use the project interpreter, so they remain portable across macOS, Linux, and Windows.

## Create the environment

Open the PyCharm terminal at the repository root.

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Select the interpreter in PyCharm:

- macOS or Linux: `.venv/bin/python`
- Windows: `.venv\Scripts\python.exe`

The editable installation makes `src/ssvep_b2b` importable by the terminal, tests, scripts, and debugger.

## Validate the installation

Paste these commands into the PyCharm terminal:

```bash
python -m pytest
python -m ssvep_b2b.cli verify-paper --config configs/paper.yaml
python -m ssvep_b2b.cli smoke --config configs/paper.yaml
```

The verification and smoke commands do not require the public EEG archives.

## Shared one-click configurations

PyCharm loads the XML files in `.run/` automatically. Select a configuration from the toolbar and press Run or Debug.

| Configuration | Operation |
|---|---|
| SSVEP verify paper | Validates publication-table structure and reports arithmetic and cross-table consistency |
| SSVEP synthetic smoke | Exercises model training, all five attacks, adversarial training, evaluation, and output writing on deterministic synthetic tensors |
| SSVEP publication analysis | Audits the frozen publication tables and recreates the source-paper figures |
| SSVEP complete reproduction | Downloads, caches, trains, attacks, evaluates, analyzes, and generates figures for both datasets |

Every shared configuration uses `main.py`, the project working directory, the selected project interpreter, and unbuffered Python output.

## Run individual stages

The same entry points can be pasted into the PyCharm terminal or placed in a temporary personal run configuration:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml
python -m ssvep_b2b.cli cache --config configs/paper.yaml
python -m ssvep_b2b.cli run --config configs/paper.yaml
python -m ssvep_b2b.cli reproduce-paper-analysis --config configs/paper.yaml
python -m ssvep_b2b.cli figures --config configs/paper.yaml
```

The thin files in `scripts/` invoke the same CLI stages:

```bash
python scripts/download_data.py --config configs/paper.yaml
python scripts/cache_data.py --config configs/paper.yaml
python scripts/run_experiments.py --config configs/paper.yaml
python scripts/verify_paper.py --config configs/paper.yaml
python scripts/reproduce_paper_analysis.py --config configs/paper.yaml
python scripts/make_figures.py --config configs/paper.yaml
python scripts/smoke.py --config configs/paper.yaml
python scripts/run_all.py --config configs/paper.yaml
```

Every stage defaults to `configs/paper.yaml`; the explicit option is shown so copied commands remain unambiguous.

## Focused debugging

Run the smaller dataset and one attack first:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml --dataset Nakanishi2015
python -m ssvep_b2b.cli cache --config configs/paper.yaml --dataset Nakanishi2015
python -m ssvep_b2b.cli run --config configs/paper.yaml --dataset Nakanishi2015 --attack fgsm --force
```

Set breakpoints inside `src/ssvep_b2b/`; the editable installation ensures the debugger executes those files. A focused execution is diagnostic and does not constitute a complete paper reproduction.

## Local data location

MOABB writes downloads below `data/raw/`. Processed HDF5 caches are written below `data/cache/`, and experiment artifacts are written below `results/runs/`. These generated directories are ignored by Git.

The Lee2019_SSVEP archive is large, and the complete 31-scenario run is computationally intensive. The synthetic smoke configuration is the appropriate one-click check for interpreter, package, attack, model, training, evaluation, and output integration without testing MOABB download or HDF5 caching.
