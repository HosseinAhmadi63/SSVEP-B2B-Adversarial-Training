# Result schema

## Path conventions

`config-hash` denotes the first 12 hexadecimal characters of the SHA-256 digest of the canonical loaded YAML mapping. `dataset` is `Nakanishi2015` or `Lee2019_SSVEP`. `attack-slug` joins the canonical lowercase attack names with `+`, such as `fgsm`, `fgsm+cw`, or `fgsm+cw+bim+mim+pgd`.

The paper configuration contains 31 scenario slugs per dataset. A complete run therefore contributes 62 rows to the aggregate experiment table.

## Data caches

```text
data/cache/{dataset}_{config-hash}.h5
data/cache/{dataset}_{config-hash}.metadata.csv
```

The HDF5 cache contains:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `X` | epochs × channels × samples | `float32` | Preprocessed and normalized EEG epochs |
| `y` | epochs | `int64` | Zero-based class indices |
| `channel_mean` | channels | floating point | Full-cohort pre-normalization mean for each channel |
| `channel_scale` | channels | floating point | Full-cohort pre-normalization standard deviation for each channel |
| `channel_names` | channels | UTF-8 text | EEG channel names in the exact tensor-axis order |
| `reference_signals` | classes × 4 × samples | `float32` | Sine/cosine pairs for two harmonics of each stimulus frequency |

HDF5 attributes record `dataset`, `config_fingerprint`, and the canonical JSON `configuration`.

The metadata CSV contains:

| Column | Meaning |
|---|---|
| `cache_index` | Unique row index into `X` and `y` |
| `subject` | MOABB subject identifier |
| `session` | Upstream session identifier as text |
| `run` | Upstream run identifier as text |
| `run_trial` | Zero-based retained epoch position within that run |
| `event_sample` | Event sample index in the upstream continuous recording |
| `original_label` | Label exposed by the dataset loader |
| `class_index` | Zero-based class index stored in `y` |
| `source_row` | Zero-based retained row position within the subject preparation batch |

## Experiment tree

```text
results/runs/{config-hash}/
├── configuration.yaml
├── environment.json
├── metrics.csv
├── comparison_to_paper.csv
├── comparison_summary.json
├── figures/
└── {dataset}/
    ├── split.npz
    ├── baseline/
    │   ├── model.pt
    │   ├── history.csv
    │   ├── predictions_clean.npz
    │   ├── metrics_clean.json
    │   └── complete.json
    └── scenarios/{attack-slug}/
        ├── attack_example.npz
        ├── model_annt.pt
        ├── history_annt.csv
        ├── predictions_without_annt.npz
        ├── predictions_with_annt.npz
        ├── predictions_clean_with_annt.npz
        ├── metrics.json
        └── complete.json
```

`configuration.yaml` is the effective configuration for the hash. `environment.json` records Python and platform strings; PyTorch, CUDA, and NumPy versions; CUDA and MPS availability; CUDA device count; and the installed version of every direct scientific dependency. `split.npz` contains the one-dimensional `training_indices` and `testing_indices` arrays used by the baseline and every scenario for that dataset.

`complete.json` is written only for a completed baseline or scenario and contains `status: complete` and `config_fingerprint`. Existing completed artifacts are reused unless the applicable command receives `--force`.

## Models and training histories

`model.pt` and `model_annt.pt` are PyTorch state dictionaries for the clean baseline and scenario-specific robust model. Loading also requires the model definition and the dataset class count from the matching configuration.

Both history CSV files use:

| Column | Unit | Meaning |
|---|---|---|
| `epoch` | one-based count | Completed training epoch |
| `loss` | mean cross-entropy | Sample-weighted training loss for that epoch |
| `accuracy` | fraction in [0, 1] | Training-set accuracy for that epoch |

`history.csv` covers clean baseline training. `history_annt.csv` covers the fresh robust model trained on the 1:1 clean and adversarial concatenation.

## Prediction archives

Every prediction NPZ contains:

| Array | Shape | Meaning |
|---|---|---|
| `indices` | samples | Cache row indices in the held-out evaluation partition |
| `labels` | samples | True zero-based class indices |
| `predictions` | samples | Argmax predicted class indices |
| `probabilities` | samples × classes | Softmax probabilities in configured class order |

The files differ only by evaluated model and input:

| File | Model | Input |
|---|---|---|
| `baseline/predictions_clean.npz` | Clean baseline | Clean held-out epochs |
| `predictions_without_annt.npz` | Clean baseline | Same-scenario attacked held-out epochs |
| `predictions_with_annt.npz` | Scenario robust model | Same-scenario attacked held-out epochs |
| `predictions_clean_with_annt.npz` | Scenario robust model | Clean held-out epochs |

Every scenario also retains `attack_example.npz`, containing `clean`, `attacked`, and `perturbation` channel-by-sample arrays, scalar `sample_index`, and scalar `snr_db`. Nakanishi2015 selects the first cached epoch from subject 4 when available, matching the article's representative subject; other datasets select the first held-out index. This artifact supports signal and perturbation visualization and is not used for training or metric calculation.

## Metric JSON files

`baseline/metrics_clean.json` contains:

| Field | Meaning |
|---|---|
| `samples` | Number of evaluated epochs |
| `classes` | Number of probability columns |
| `accuracy` | Argmax accuracy as a fraction |
| `auc_macro_ovr` | Macro one-vs-rest ROC-AUC |
| `cross_entropy` | Mean multiclass cross-entropy |

Each scenario `metrics.json` contains:

| Field | Meaning |
|---|---|
| `dataset` | Dataset name |
| `scenario` | Canonical attack slug |
| `scenario_display` | Display label with uppercase names and `C&W` |
| `attack_count` | Number of constituent attacks |
| `accuracy_without_annt` | Attacked-test accuracy of the clean baseline |
| `accuracy_with_annt` | Attacked-test accuracy of the robust model |
| `accuracy_improvement` | `accuracy_with_annt - accuracy_without_annt` |
| `cross_entropy_without_annt` | Attacked-test cross-entropy of the clean baseline |
| `cross_entropy_with_annt` | Attacked-test cross-entropy of the robust model |
| `auc_without_annt` | Attacked-test macro OvR AUC of the clean baseline |
| `auc_with_annt` | Attacked-test macro OvR AUC of the robust model |
| `auc_improvement` | `auc_with_annt - auc_without_annt` |
| `clean_accuracy_with_annt` | Clean-test accuracy of the robust model |
| `clean_auc_with_annt` | Clean-test macro OvR AUC of the robust model |
| `clean_cross_entropy_with_annt` | Clean-test cross-entropy of the robust model |
| `duration_seconds` | Scenario wall-clock duration in seconds |

Scenario timing begins before loading the clean baseline and generating the scenario attack cache. It includes attack generation, attacked-baseline evaluation, robust-model training, attacked-robust evaluation, clean-robust evaluation, and prediction/checkpoint/history writes. It excludes data download, preprocessing cache construction, split creation, clean-baseline training, aggregate-table creation, plotting, and post-scenario cleanup.

The temporary `attacked.h5` scenario cache stores attacked `X` and labels `y`, plus dataset, scenario, combination, generator-model, and completion attributes. With the paper setting `keep_attack_cache: false`, it is removed after the durable scenario artifacts are written. Setting that option to true retains it for inspection.

## Aggregate experiment table

`results/runs/{config-hash}/metrics.csv` contains the scenario JSON fields in configured scenario order, with datasets ordered `Nakanishi2015` then `Lee2019_SSVEP`, plus:

| Column | Meaning |
|---|---|
| `normalized_computational_time` | `100 × duration_seconds / global_minimum_duration_seconds` |
| `paper_run_complete` | `true` only when all 62 configured dataset-scenario artifacts are complete |
| `timing_normalization_scope` | `complete_paper_table` for 62 rows or `partial_completed_rows` for a diagnostic subset |

The global minimum is taken across the complete two-dataset scenario table. The fastest row is therefore 100. A partial diagnostic table is not a complete paper reproduction and its provisional normalization must not be compared with the publication values.

`python -m ssvep_b2b.cli figures --config configs/paper.yaml` reads this aggregate table and writes experiment figures below `results/runs/{config-hash}/figures/`. A complete run produces:

```text
figure7_improvements.png
figure8_improvement_distributions.png
figure9_computational_time.png
roc_comparison_nakanishi2015_fgsm.png
roc_comparison_nakanishi2015_pgd.png
roc_comparison_lee2019_ssvep_fgsm.png
roc_comparison_lee2019_ssvep_pgd.png
figure1_attack_examples_nakanishi2015.png
figure1_attack_examples_lee2019_ssvep.png
```

## Generated-to-paper comparison

Every nonempty `run` result is joined to `results/publication/source/table3.csv` by dataset and canonical scenario. `comparison_to_paper.csv` contains these columns in order:

```text
dataset
scenario_key
attacks
paper_accuracy_without_annt
paper_accuracy_with_annt
paper_accuracy_improvement_reported
paper_auc_without_annt
paper_auc_with_annt
paper_auc_improvement_reported
paper_normalized_computational_time
generated_accuracy_without_annt
generated_accuracy_with_annt
generated_accuracy_improvement
generated_auc_without_annt
generated_auc_with_annt
generated_auc_improvement
generated_normalized_computational_time
difference_accuracy_without_annt
difference_accuracy_with_annt
difference_accuracy_improvement
difference_auc_without_annt
difference_auc_with_annt
difference_auc_improvement
difference_normalized_computational_time
```

Every `difference_` value is generated minus paper.

`comparison_summary.json` contains `matched_rows`, `paper_rows`, `coverage`, and `metrics`. `paper_rows` is 62; `coverage` is `matched_rows / paper_rows`. Each metric entry contains `mean_absolute_error`, `root_mean_squared_error`, and `mean_signed_error` across the matched rows. A focused run produces an explicit partial-coverage comparison; coverage reaches 1.0 only after all 62 dataset-scenario rows are present.

## Publication source tables

`results/publication/source/table3.csv` is a frozen transcription with these columns:

| Column | Meaning |
|---|---|
| `scenario_order` | One-based order of the reported scenario |
| `attack_count` | Number of constituent attacks |
| `attacks` | Publication display label |
| `dataset_code` | Paper shorthand `A` or `B` |
| `dataset` | Repository dataset name |
| `accuracy_no_annt` | Reported attacked accuracy without adversarial training |
| `accuracy_annt` | Reported attacked accuracy with adversarial training |
| `accuracy_improvement_reported` | Improvement printed by the article |
| `auc_no_annt` | Reported attacked AUC without adversarial training |
| `auc_annt` | Reported attacked AUC with adversarial training |
| `auc_improvement_reported` | AUC improvement printed by the article |
| `computational_time_normalized` | Reported relative time with the minimum defined as 100 |

`results/publication/source/table4.csv` captures the article's least- and most-effective claims. Its columns are `dataset`, `metric`, `attack_count`, `least_effective_attack`, `least_effective_value`, `most_effective_attack`, and `most_effective_value`.

The frozen source tables are evidence inputs. Experiment training, attack generation, and model selection never read them.

## Generated publication analysis

`python -m ssvep_b2b.cli reproduce-paper-analysis --config configs/paper.yaml` writes:

| Artifact | Contents |
|---|---|
| `results/publication/generated/table3_delta_audit.csv` | Printed improvements, recomputed differences, residuals, and consistency flags for accuracy and AUC |
| `results/publication/generated/table4_consistency_audit.csv` | Table 4 claim values joined to Table 3, numeric agreement, expected extremes, tied attack sets, and extreme-selection flags |
| `results/publication/generated/derived_summary.csv` | Scenario counts plus grouped means and normalized-time ranges by dataset and attack count |
| `results/publication/generated/publication_audit.json` | Structure counts, discrepancy counts and records, derived summary, generated artifact paths, and a separate `figures` path list |

`table3_delta_audit.csv` uses `scenario_order`, `attack_count`, `attacks`, `dataset_code`, `dataset`, `metric`, `without_annt`, `with_annt`, `reported_delta`, `recomputed_delta`, `reported_minus_recomputed`, and `is_consistent`.

`table4_consistency_audit.csv` uses `dataset`, `metric`, `attack_count`, `claim_role`, `selected_attack`, `printed_value`, `table3_value`, `printed_minus_table3`, `value_match`, `expected_extreme_value`, `expected_attacks`, and `scenario_is_extreme`.

`derived_summary.csv` uses `dataset`, `attack_count`, `scenario_count`, `mean_accuracy_no_annt`, `mean_accuracy_annt`, `mean_accuracy_improvement_reported`, `mean_accuracy_improvement_recomputed`, `mean_auc_no_annt`, `mean_auc_annt`, `mean_auc_improvement_reported`, `mean_auc_improvement_recomputed`, `mean_computational_time_normalized`, `minimum_computational_time_normalized`, and `maximum_computational_time_normalized`. It contains attack-count groups 1 through 5 and an `all` group for each dataset.

`publication_audit.json` records source validity and overall internal consistency; source paths and row, scenario, endpoint, and claim counts; the derived-summary records; the four audit artifact paths; and the three generated figure paths. 

Publication plotting writes `figure7_improvements.png`, `figure8_improvement_distributions.png`, and `figure9_computational_time.png` below `results/publication/generated/`.

## Synthetic smoke result

`python -m ssvep_b2b.cli smoke --config configs/paper.yaml` writes `results/smoke/summary.json`. It records completion status, sample and class counts, the five attacks, the maximum combined absolute perturbation, and nested clean-baseline, attacked-without-ANNT, and attacked-with-ANNT metrics. It uses deterministic synthetic SSVEP-like tensors and does not download either study dataset.
