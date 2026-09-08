# Paper-to-code map

## Purpose

This document maps the method in *Securing Brain-to-Brain Communication Channels Using Adversarial Training on SSVEP EEG* to the executable repository. It separates values stated in the article from deterministic choices added here where the article does not uniquely specify an implementation.

The primary paper-mode specification is `configs/paper.yaml`. The command below validates its fixed datasets, scenarios, attack values, and publication-source structure without downloading EEG data:

```bash
python -m ssvep_b2b.cli verify-paper --config configs/paper.yaml
```

## End-to-end map

| Article component | Constraint carried from the article | Executable realization | Primary implementation or artifact |
|---|---|---|---|
| Data sources | Nakanishi2015 and Lee2019_SSVEP cohorts | Pinned MOABB dataset classes; recordings download automatically to `data/raw/` | `ssvep_b2b.data`, `configs/paper.yaml`, `data/README.md` |
| Cohorts | Nakanishi: 9 subjects, 8 channels, 12 classes, 15 trials per class; Lee: 54 subjects, 62 channels, 4 classes, 50 trials per class, 2 sessions | Explicit subject lists, dimensions, event intervals, native rates, and labelled Lee training runs | `configs/paper.yaml`, `ssvep_b2b.data.validate_cache` |
| Montage | Standard 10–20 channel locations | MNE `standard_1020`, case-insensitive matching, missing auxiliary locations ignored | `ssvep_b2b.data._prepare_subject` |
| Referencing | Common-average reference | Applied to each continuous EEG run with projection disabled | `ssvep_b2b.data._prepare_subject` |
| Filtering | 4–16 Hz band-pass | Zero-phase MNE FIR, Hamming window, `firwin` design, applied to continuous EEG before epoching | `ssvep_b2b.data._prepare_subject`, `configs/paper.yaml` |
| Epochs | Full 4.15 s or 4.0 s trial interval | Event-locked MNE epochs with no baseline correction and an inclusive endpoint chosen for exactly 1,062 or 4,000 samples | `ssvep_b2b.data._prepare_subject` |
| Normalization | Z-score normalization | One mean and standard deviation per channel over the entire configured dataset, fitted after epoch construction and before splitting | `ssvep_b2b.data.cache_dataset` |
| Labels | Frequency-indexed classes and one-hot description | Stable zero-based integer labels express the equivalent hard targets for cross-entropy; full softmax probability vectors support multiclass evaluation | `ssvep_b2b.data._label_indices`, `ssvep_b2b.training` |
| Reference signals | Sine/cosine SSVEP references appear in the method flow | Two harmonics per frequency are retained in the cache for provenance but are not CNN–TCN inputs | `ssvep_b2b.data.reference_signals` |
| Train/evaluation division | 80/20 | One pooled, shuffled, class-stratified epoch split with seed 2025; subjects and sessions are not grouping variables | `ssvep_b2b.data.split_indices` |
| CNN block | 32-filter 3×3 convolution, ReLU, 2×2 max pooling, 64-filter 3×3 convolution, ReLU | Same padding and unit stride for both convolutions | `ssvep_b2b.model.CNNTCN` |
| TCN block | Reshape to a 64-filter TCN described with a 3×3 kernel | Pooled spatial positions form a sequence; one causal residual Conv1D layer uses 64 channels, kernel 3, dilation 1, and ReLU | `ssvep_b2b.model.CausalResidualBlock` |
| Classifier | Dataset-sized softmax output | Global sequence mean and a linear 12- or 4-class logits layer; softmax is applied during prediction | `ssvep_b2b.model`, `ssvep_b2b.training.predict_model` |
| Baseline training | A clean CNN–TCN baseline | Adam, cross-entropy, learning rate 0.001, weight decay 0, 100 epochs, no early stopping, dataset batch size 8 or 2 | `ssvep_b2b.training.train_model`, `ssvep_b2b.experiment.ensure_baseline` |
| Adversarial examples | FGSM, BIM, C&W, MIM, and PGD | Untargeted white-box attacks against the completed clean baseline on normalized tensors | `ssvep_b2b.attacks`, `ssvep_b2b.experiment._make_adversarial_cache` |
| Attack combinations | Every combination from one through five attacks | The 31 nonempty subsets are explicitly ordered in the YAML specification | `configs/paper.yaml`, `ssvep_b2b.experiment.selected_scenarios` |
| Combined perturbation | Figure 1 presents an additive combination | Constituent attacks start independently from the same clean input; their deltas are summed without a final joint projection or input clipping | `ssvep_b2b.attacks.combine_attacks` |
| ANNT | Train with clean and matching adversarial examples | A fresh model per scenario is trained on a 1:1 concatenation of clean training epochs and static same-scenario adversarial epochs | `ssvep_b2b.experiment.run_scenario` |
| Matched testing | Compare attacks without and with ANNT | The clean baseline and scenario-specific robust model are evaluated on the same attacked held-out indices; the robust model is also evaluated on clean held-out data | `ssvep_b2b.experiment.run_scenario` |
| Accuracy | Multiclass classification accuracy | Argmax accuracy over all held-out epochs | `ssvep_b2b.metrics.classification_metrics` |
| ROC-AUC | Multiclass ROC/AUC | Macro-averaged one-vs-rest ROC-AUC from the complete softmax matrix | `ssvep_b2b.metrics.classification_metrics` |
| Computational time | Relative time with the minimum equal to 100 | Scenario wall time in seconds, followed by `100 × time / global minimum` over the complete two-dataset table | `ssvep_b2b.experiment.run_scenario`, `ssvep_b2b.experiment.completed_results` |
| Paper table analysis | Accuracy, AUC, improvement, and timing values in Tables 3 and 4 | Frozen transcription, arithmetic audit, extreme-value audit, derived summaries, and recreated figures | `results/publication/source/`, `ssvep_b2b.paper_results`, `ssvep_b2b.plotting` |
| New-run comparison | Generated execution values must remain distinguishable from the publication transcription | Post-evaluation join by dataset and canonical scenario, with per-row differences, coverage, MAE, RMSE, and mean signed error | `ssvep_b2b.comparison`, `comparison_to_paper.csv`, `comparison_summary.json` |

## Dataset mapping

| Property | Nakanishi2015 | Lee2019_SSVEP |
|---|---:|---:|
| MOABB class | `Nakanishi2015` | `Lee2019_SSVEP(train_run=True, test_run=None)` |
| Subjects | 1–9 | 1–54 |
| Sessions used | 1 | 2 |
| EEG channels | 8 | 62 |
| Classes | 12 | 4 |
| Trials per class and subject | 15 | 50 |
| Native sampling rate | 256 Hz | 1000 Hz |
| Frozen event interval | 0.15–4.30 s | 0.00–4.00 s |
| Samples per stored epoch | 1,062 | 4,000 |
| Expected epochs | 1,620 | 10,800 |
| Training batch size | 8 | 2 |

Nakanishi class indices follow `9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75` Hz. Lee class indices follow the pinned MOABB event-code order `12.00, 8.57, 6.67, 5.45` Hz. The article lists the Lee frequencies but does not define an executable index order, so the latter is a repository-defined choice.

## Attack mapping

| Attack | Article values retained | Frozen executable details |
|---|---|---|
| FGSM | epsilon 0.01 | One untargeted cross-entropy sign step |
| BIM | epsilon 0.01, alpha 0.001, 50 iterations | Clean start and per-step L-infinity projection |
| C&W | perturbation limit 0.01, `c=0.001`, `kappa=10`, 1000 iterations, L2 | Adam learning rate 0.01, squared-L2 objective, per-coordinate delta cap, least-L2 successful candidate |
| MIM | epsilon 0.01, alpha 0.001, momentum 0.5, 50 iterations | Clean start, per-sample L1 gradient normalization, per-step L-infinity projection |
| PGD | epsilon 0.01, alpha 0.001, 50 iterations | Zero random start and per-step L-infinity projection |

BIM and the frozen zero-start PGD have the same update equation. Both names remain because the article reports them as distinct attacks.

## What is paper-stated and what is repository-defined

The datasets, cohort dimensions, passband, 80/20 division, high-level CNN–TCN structure, five attacks and their reported numeric parameters, attack subsets, ANNT comparison, accuracy, ROC/AUC, and normalized-time presentation come from the article.

The following are repository-defined deterministic choices needed to execute that description: MOABB/MNE loader versions; precise event bounds; FIR family, phase, and window; normalization axes and pre-split fitting; pooled stratification and seed; CNN padding; causal residual TCN interpretation; global-mean reduction; initialization; optimizer and all training hyperparameters; static 1:1 adversarial concatenation; fresh robust-model initialization; C&W optimizer details; independent delta summation; no combined projection or input clipping; macro one-vs-rest AUC; and timing boundaries.

These choices are documented rather than inferred to be the authors' unpublished settings. The implementation therefore supports a transparent, paper-faithful experiment, but exact numerical recovery from the article alone cannot be guaranteed. See `docs/IMPLEMENTATION_DETAILS.md` for the full rationale and `docs/REPRODUCIBILITY.md` for the execution contract.

## Executable stages

```text
download -> cache -> split -> clean baseline -> 31 scenario runs -> aggregate -> compare -> figures
                      publication source -> arithmetic and consistency audits -> source-paper figures
```

The full experiment is started with:

```bash
python -m ssvep_b2b.cli run-all --config configs/paper.yaml
```

Output paths and field definitions are specified in `docs/RESULT_SCHEMA.md`.
