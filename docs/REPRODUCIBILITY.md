# Reproducibility contract

## What this repository guarantees

The repository guarantees that every executable choice in its paper-mode pipeline is visible in `configs/paper.yaml`, validated before a run, and associated with a deterministic configuration fingerprint. It preserves generated outputs even when they differ from the article.

The repository does not guarantee bit-identical recovery of the authors' historical experiment. The article leaves implementation details unresolved, the original split and trained weights are not published here, upstream dataset loaders can evolve, floating-point kernels vary by hardware, and computational-time values depend on the machine and measurement boundary.

## Reference environment

- Python 3.11.
- Direct scientific dependencies pinned in `requirements.txt`.
- Setuptools package installed with `pip install -e '.[dev]'`.
- Global seed 2025.
- Deterministic mode enabled.
- Data-loader workers set to zero.
- Fixed 100-epoch training without early stopping.

The runtime seeds Python, NumPy, and PyTorch together, sets `PYTHONHASHSEED`, seeds every available CUDA device, requests deterministic PyTorch algorithms, disables cuDNN benchmarking, and enables deterministic cuDNN behavior.

CPU execution provides the strongest cross-run determinism. CUDA and Apple MPS can produce small numerical differences even with identical seeds. Timing results are inherently machine-specific on every device.

## Configuration identity

The loaded YAML mapping is serialized with sorted keys and compact separators, hashed with SHA-256, and truncated to 12 hexadecimal characters. This fingerprint identifies caches and experiment outputs.

```text
data/cache/Nakanishi2015_{config_hash}.h5
data/cache/Lee2019_SSVEP_{config_hash}.h5
results/runs/{config_hash}/
```

Changing a paper parameter creates a different fingerprint. Dataset and attack selections are execution filters and do not change that fingerprint. A filtered run is therefore diagnostic until the directory contains both datasets and all 31 configured scenarios for each; the expected 62-row aggregate table is the completeness criterion.

The fingerprint covers configuration values but not source-code or dependency changes. Every run records a compact software environment, and a publication archive should additionally retain the Git commit identifier and installed package inventory used to produce it.

## Dataset identity

- MOABB is pinned to version 1.4.3.
- MNE is pinned to version 1.10.1.
- PyRiemann is pinned to version 0.12 because it is part of MOABB's scientific runtime.
- The Nakanishi2015 paper cohort is subjects 1–9.
- Lee2019_SSVEP uses subjects 1–54, both sessions, and labelled training runs.
- Expected sample counts, channels, class counts, trial counts, and metadata row counts are validated before training.
- Original labels and source lineage are retained beside every cache row.
- Raw downloads and generated caches are never treated as repository source files.

Upstream servers remain outside this reproducibility boundary. If a provider replaces or removes source bytes without a versioned identifier, a later download can differ despite the same local configuration.

## Frozen preprocessing order

The paper-mode implementation applies the following order to each continuous run:

1. Attach the standard 10–20 montage with MNE's `standard_1020` definition.
2. Apply common-average reference over EEG channels.
3. Apply a 4–16 Hz zero-phase FIR filter using a Hamming window and `firwin` design through the pinned MNE implementation.
4. Extract the configured full-length event epochs and trim to the exact integer sample count.
5. Concatenate the complete configured cohort.
6. Fit one mean and standard deviation per channel over every cohort epoch and time sample.
7. Store normalized `float32` epochs.
8. Create the stratified 80/20 epoch split with seed 2025.

Normalization is intentionally fitted before the split to reproduce the order described by the article. This exposes evaluation-distribution summary statistics to training and is not a leakage-free protocol. The choice is documented, tested, and must not be confused with a train-only normalization benchmark.

## Split identity

The split pools subjects and sessions, stratifies by frequency class, shuffles with seed 2025, and reserves 20 percent of epochs for evaluation. The saved indices are shared by the baseline and every attack scenario.

This design tests random held-out epochs from the same combined cohort. It does not estimate performance on unseen subjects or unseen sessions. LOSO, group shuffle, and cross-session evaluation are scientifically useful but constitute different protocols.

## Matched model comparisons

- The same split is used across all scenarios.
- The clean baseline is trained once per dataset.
- Every robust scenario creates a fresh CNN–TCN model under the frozen seed.
- Every robust model uses the same clean training epochs and a 1:1 same-scenario adversarial extension.
- Static adversarial samples are created against the completed clean baseline.
- Clean and attacked evaluations use identical held-out indices.
- The transcribed Table 3 values never enter loss calculation, early stopping, attack generation, or model selection.

## Attack determinism

FGSM, BIM, MIM, and zero-start PGD are deterministic for a fixed baseline, input tensor, and numerical environment. C&W begins from a zero perturbation and uses a fixed Adam update sequence. Attacks disable model training behavior while calculating gradients and restore the previous model mode afterward.

Every constituent of a combined scenario is generated independently from the same clean tensor. Perturbations are summed in the scenario order stored in `configs/paper.yaml`. There is no final joint projection or input-domain clipping. The scenario slug preserves that canonical order.

## Safe resumption

Experiment stages write completion records after their required artifacts are produced. Generated JSON, CSV, HDF5, NPZ, and model files use temporary paths followed by replacement. Cache, experiment, and complete-run commands reuse existing artifacts by default and accept `--force` for their applicable generated outputs. Publication analysis deterministically replaces its derived audit and figure files on each invocation.

A complete paper run is:

```bash
python -m ssvep_b2b.cli run-all --config configs/paper.yaml
```

A download-free integrity check is:

```bash
python -m ssvep_b2b.cli verify-paper --config configs/paper.yaml
```

A deterministic synthetic model, attack, training, evaluation, and output check is:

```bash
python -m ssvep_b2b.cli smoke --config configs/paper.yaml
```

## Timing interpretation

Raw elapsed times are recorded in seconds. For the complete reproduction, normalized time is calculated after both datasets and all 31 paper scenarios have been collected:

```text
normalized_time = 100 × elapsed_seconds / global_minimum_elapsed_seconds
```

The normalization makes the fastest row equal to 100; it does not make timing hardware-independent. A focused run also receives a provisional normalization over its currently completed rows, but that value is not comparable with the 62-row paper protocol. Published and complete-run times should be compared as relative trends, not as evidence of identical execution cost.

## Archival record

A defensible result archive contains:

- The exact YAML configuration.
- Its 12-character fingerprint.
- The source Git commit identifier.
- The environment manifest.
- Dataset cache metadata and split indices.
- Baseline and robust checkpoints.
- Training histories.
- Clean and adversarial prediction probabilities.
- Raw and normalized timing records.
- Generated publication comparisons and integrity reports.

Together these artifacts establish what this implementation executed. They do not substitute for unavailable provenance from the original unpublished codebase.
