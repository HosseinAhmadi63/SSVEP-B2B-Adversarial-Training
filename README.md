# SSVEP B2B Adversarial Training

This repository is a paper-faithful executable reimplementation of:

> **Securing Brain-to-Brain Communication Channels Using Adversarial Training on SSVEP EEG**  
> Hossein Ahmadi, Ali Kuhestani, Mohammadreza Keshavarzi, and Luca Mesin  
> *IEEE Access*, volume 13, pages 14358–14378, 2025  
> [DOI 10.1109/ACCESS.2025.3528770](https://doi.org/10.1109/ACCESS.2025.3528770)

The pipeline downloads the two public SSVEP datasets used in the article, applies the stated common-average reference, 4–16 Hz filtering, normalization, and epoch construction, builds the reported CNN–TCN classifier, evaluates all 31 nonempty subsets of FGSM, BIM, Carlini–Wagner, MIM, and PGD, trains a separate adversarially robust model for every attack scenario, calculates accuracy and macro one-vs-rest ROC-AUC, reproduces the publication analysis from frozen source values, and generates paper-aligned figures and comparison tables.

If you use this repository, its code, or its results, cite the article above. Machine-readable citation metadata are in [CITATION.cff](CITATION.cff).

## Reimplementation status

This is an executable, deterministic reconstruction of the method described in the article. Exact numerical reproduction cannot be guaranteed because the paper omits the filter family and phase response, normalization axes, data-split grouping and seed, several epoch-boundary details, CNN padding, the executable form of the TCN, initialization, optimizer and training hyperparameters, the adversarial-training schedule, the precise construction of combined attacks, the multiclass AUC convention, and timing boundaries. Dataset-loader and numerical-library versions can also affect samples and floating-point results.

Every choice required to turn the paper into runnable code is frozen in [configs/paper.yaml](configs/paper.yaml) and explained in [docs/IMPLEMENTATION_DETAILS.md](docs/IMPLEMENTATION_DETAILS.md). The pipeline joins generated and transcribed values only after evaluation, reports their signed and aggregate errors, and never tunes models toward publication targets.

## Frozen pipeline

| Stage | Executable implementation |
|---|---|
| Datasets | MOABB `Nakanishi2015` and `Lee2019_SSVEP` |
| Nakanishi2015 | 9 subjects, 8 EEG channels, 12 classes, 15 trials per class, one session, 256 Hz, 4.15 s interval |
| Lee2019_SSVEP | 54 subjects, 62 EEG channels, 4 classes, 50 trials per class, two sessions, 1000 Hz, 4.0 s interval, labelled training runs |
| Preprocessing | Standard 10–20 montage, continuous-run common-average reference, zero-phase MNE FIR 4–16 Hz filtering with a Hamming-windowed `firwin` design, exact-length epoching, then full-cohort per-channel z-score and `float32` storage |
| Split | Pooled subjects and sessions, stratified 80/20 epoch split, seed 2025 |
| CNN | Conv2D(32, 3×3, same, ReLU), MaxPool2D(2×2), Conv2D(64, 3×3, same, ReLU) |
| TCN | Flattened CNN positions, one causal residual Conv1D layer, 64 filters, kernel 3, dilation 1, ReLU, global mean |
| Classifier | Dense output with 12 classes for Nakanishi2015 or 4 classes for Lee2019_SSVEP; cross-entropy on logits |
| Training | Adam, learning rate 0.001, zero weight decay, 100 fixed epochs, no early stopping, batch 8 or 2 by dataset |
| ANNT | Fresh robust model trained on a 1:1 concatenation of clean data and static same-scenario adversarial data |
| Attack combinations | Each constituent perturbation is generated independently against the clean baseline and clean input; deltas are summed without a final joint projection |
| Evaluation | Clean baseline, attacked baseline, clean robust model, and attacked robust model; accuracy and macro one-vs-rest ROC-AUC |
| Timing | Raw elapsed seconds plus normalization over the complete two-dataset scenario table, with the global minimum defined as 100 |

Attack parameters are frozen as follows:

| Attack | Parameters |
|---|---|
| FGSM | epsilon 0.01 |
| BIM | epsilon 0.01, step 0.001, 50 iterations, zero start |
| C&W | L2 objective, `c=0.001`, confidence 10, 1000 iterations, Adam learning rate 0.01, elementwise delta limit 0.01 |
| MIM | epsilon 0.01, step 0.001, momentum 0.5, 50 iterations, zero start |
| PGD | epsilon 0.01, step 0.001, 50 iterations, zero start |

Attacks are untargeted and operate on normalized tensors without input-domain clipping. BIM, MIM, and PGD project each individual perturbation into its own L-infinity ball. Combined scenarios intentionally do not receive another projection after summation.

## Repository structure

```text
configs/paper.yaml                    Frozen paper reimplementation
scripts/                              PyCharm-friendly stage entry points
src/ssvep_b2b/                        Installable implementation package
tests/                                Scientific, configuration, and integration checks
data/README.md                        Dataset provenance and license notes
results/publication/source/           Values transcribed from the article
results/publication/generated/        Recreated analyses, figures, and comparisons
results/runs/{config_hash}/            Models, predictions, metrics, split indices, and manifests
docs/PAPER_TO_CODE.md                 Article-to-implementation map
docs/IMPLEMENTATION_DETAILS.md        Paper statements and frozen executable choices
docs/REPRODUCIBILITY.md               Determinism and scope contract
docs/RESULT_SCHEMA.md                 Output paths, columns, and units
```

Start with the [paper-to-code map](docs/PAPER_TO_CODE.md), then use the [reproducibility contract](docs/REPRODUCIBILITY.md) to interpret the scope of generated results.

## Installation

Python 3.11 is the reference interpreter.

macOS or Linux:

```bash
git clone https://github.com/HosseinAhmadi63/SSVEP-B2B-Adversarial-Training.git
cd SSVEP-B2B-Adversarial-Training
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Windows PowerShell:

```powershell
git clone https://github.com/HosseinAhmadi63/SSVEP-B2B-Adversarial-Training.git
cd SSVEP-B2B-Adversarial-Training
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Verify the installation without downloading EEG archives:

```bash
python -m pytest
python -m ssvep_b2b.cli verify-paper --config configs/paper.yaml
python -m ssvep_b2b.cli smoke --config configs/paper.yaml
```

`verify-paper` validates the frozen table structure and reports arithmetic or cross-table discrepancies without altering the transcription.

An equivalent Conda environment is defined in [environment.yml](environment.yml).

## PyCharm

Open the repository root as a PyCharm project, select Python 3.11, create the interpreter at `.venv`, and run the installation command above in the project terminal. Shared one-click configurations are included for paper verification, the synthetic smoke run, publication analysis, and the complete reproduction. Detailed setup and paste-ready commands are in [docs/PYCHARM.md](docs/PYCHARM.md).

## Complete reproduction

The full Lee2019_SSVEP archive is large and model training includes 31 attack scenarios for each dataset. Ensure adequate disk space and expect the complete CPU run to take substantial time. MOABB downloads source recordings automatically during the download stage.

Run the complete ordered pipeline:

```bash
python -m ssvep_b2b.cli run-all --config configs/paper.yaml
```

Run stages separately:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml
python -m ssvep_b2b.cli cache --config configs/paper.yaml
python -m ssvep_b2b.cli run --config configs/paper.yaml
python -m ssvep_b2b.cli reproduce-paper-analysis --config configs/paper.yaml
python -m ssvep_b2b.cli figures --config configs/paper.yaml
```

`configs/paper.yaml` is the default for every command, so its `--config` option may be omitted. `main.py` delegates to the same CLI, and the thin files in `scripts/` mirror its individual stages for IDE use.

Run a focused diagnostic experiment:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml --dataset Nakanishi2015
python -m ssvep_b2b.cli cache --config configs/paper.yaml --dataset Nakanishi2015
python -m ssvep_b2b.cli run --config configs/paper.yaml --dataset Nakanishi2015 --attack fgsm
```

Omitting `--dataset` selects both configured datasets. Omitting `--attack` selects all 31 configured scenarios. Completed artifacts are reused; `--force` intentionally replaces outputs for the same configuration and selection.

## Outputs

Each effective configuration receives a deterministic SHA-256-derived run key. Baseline artifacts are written below:

```text
results/runs/{config_hash}/{dataset}/baseline/
```

Attack-specific baseline and robust-model artifacts are written below:

```text
results/runs/{config_hash}/{dataset}/scenarios/{attack_slug}/
```

The combined scenario table is `results/runs/{config_hash}/metrics.csv`. Its matched publication comparison and error summary are `comparison_to_paper.csv` and `comparison_summary.json` in the same run root. Frozen article values remain in `results/publication/source/table3.csv` and `table4.csv`; regenerated tables, integrity reports, and figures are written to `results/publication/generated/`. Exact file and column semantics are documented in [docs/RESULT_SCHEMA.md](docs/RESULT_SCHEMA.md).

## Data and licenses

No EEG recordings are committed. MOABB downloads data into the configured local data directory, and generated caches are excluded from Git. The source datasets remain governed by their providers' terms; installing or using this MIT-licensed software does not grant additional rights to the recordings. See [data/README.md](data/README.md) before downloading.

MOABB is BSD-3-Clause software, while its dataset catalog records the Nakanishi2015 license as unknown and the Lee2019_SSVEP license as GPL 3.0. Always retain and review the notices delivered by the upstream dataset providers.

The IEEE Access article is available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The software in this repository is released under the [MIT License](LICENSE). These licenses do not replace the independent terms of either dataset.

## Citation

```bibtex
@article{ahmadi2025securing,
  author  = {Ahmadi, Hossein and Kuhestani, Ali and Keshavarzi, Mohammadreza and Mesin, Luca},
  title   = {Securing Brain-to-Brain Communication Channels Using Adversarial Training on SSVEP EEG},
  journal = {IEEE Access},
  year    = {2025},
  volume  = {13},
  pages   = {14358--14378},
  doi     = {10.1109/ACCESS.2025.3528770}
}
```

## License

The source code is released under the MIT License. The article's CC BY 4.0 license and the source datasets' terms remain separate and controlling.
