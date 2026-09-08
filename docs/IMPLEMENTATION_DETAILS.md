# Implementation details

## Scope

This repository turns the method in *Securing Brain-to-Brain Communication Channels Using Adversarial Training on SSVEP EEG* into a complete Python 3.11 pipeline. It implements the two reported datasets, the CNN–TCN model, five white-box attacks, every nonempty attack combination, scenario-specific adversarial training, multiclass evaluation, timing analysis, and publication comparisons.


## Choices stated by the article

The article specifies:

- Nakanishi2015 and Lee2019_SSVEP.
- Nine subjects, 8 channels, 12 classes, 15 trials per class, one session, 4.15 s trials, and 256 Hz for Nakanishi2015.
- Fifty-four subjects, 62 channels, 4 classes, 50 trials per class, two sessions, 4.0 s trials, and 1000 Hz for Lee2019_SSVEP.
- Common-average reference, 4–16 Hz band-pass filtering, normalization, epoch creation, frequency-label indexing, one-hot representation, and an 80/20 train/evaluation division.
- Conv2D with 32 filters and a 3×3 kernel, ReLU, 2×2 max pooling, Conv2D with 64 filters and a 3×3 kernel, ReLU, reshape, a 64-filter TCN with a reported 3×3 kernel, and a softmax classifier sized to the dataset.
- FGSM, BIM, C&W, MIM, and PGD, including the values reproduced in `configs/paper.yaml`.
- Individual attacks and combinations of two through five attacks.
- Adversarial training with both clean and same-scenario perturbed examples.
- Evaluation of clean data, attacked data without ANNT, and attacked data with ANNT.
- Accuracy, multiclass ROC/AUC, and computational-time comparisons.

## Frozen data acquisition

MOABB 1.4.3 supplies both dataset loaders and their continuous recordings. MNE 1.10.1 applies the montage, reference, FIR filter, and event epoching directly. Direct scientific dependencies are pinned so this behavior does not drift silently.

### Nakanishi2015

- Subjects: 1 through 9.
- Channels: 8 EEG channels.
- Class indices follow `9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75` Hz.
- Trial interval: 0.15 through 4.30 seconds.
- Stored samples: `round(4.15 × 256) = 1062` per epoch.
- Expected epochs: `9 × 12 × 15 = 1620`.

### Lee2019_SSVEP

- Subjects: 1 through 54.
- Channels: the 62 EEG channels, excluding auxiliary channels.
- Both sessions are used.
- The labelled training/offline run is requested; an unlabelled online test run is not added.
- Class indices follow the MOABB event-code order `12.00, 8.57, 6.67, 5.45` Hz.
- Trial interval: 0.00 through 4.00 seconds.
- Stored samples: `4.0 × 1000 = 4000` per epoch.
- Expected epochs: `54 × 4 × 50 = 10800`.

The loader validates the full expected tensor shape and equal class counts. Every metadata row receives a stable cache index and retains the original label, subject, session, and run lineage.

## Frozen preprocessing

The executable protocol uses:

1. Load each configured subject's continuous runs through the pinned MOABB dataset class.
2. Attach the MNE `standard_1020` montage, matching channel names without case sensitivity and retaining a run when an auxiliary name has no montage location.
3. Apply common-average reference to the continuous EEG run with projection disabled.
4. Apply MNE's zero-phase FIR band-pass from 4 through 16 Hz to the continuous EEG channels, using a Hamming window and `firwin` design. Filtering does not split at annotations.
5. Extract event-locked epochs with no baseline correction, reject annotated spans, and choose the inclusive endpoint that produces exactly the configured integer sample count.
6. Concatenate all configured subjects, sessions, runs, and epochs.
7. Fit one population mean and population standard deviation per EEG channel over every cohort epoch and time sample, using `float64` accumulators.
8. Replace a channel scale below `1e-6` with one, normalize, and store tensors as `float32`.
9. Map labels to zero-based integer class indices for cross-entropy; probability outputs preserve that class order. The executable pipeline does not materialize the article's one-hot label representation because hard-label cross-entropy accepts the equivalent class indices directly.
10. Store two sine/cosine harmonics per stimulus frequency for traceability; the CNN–TCN consumes EEG epochs rather than those reference signals.

## Frozen split

- Subjects and sessions are pooled within each dataset.
- The unit of splitting is an epoch.
- Twenty percent of epochs form the held-out evaluation set.
- Class stratification is enabled.
- The random seed is 2025.
- Split membership is generated once and reused by the clean baseline and every adversarial scenario.

This is not leave-one-subject-out, cross-session, or grouped cross-validation. Those protocols answer different generalization questions and are not substituted for the paper-mode split.

## Frozen CNN–TCN

The model accepts `[batch, 1, EEG channel, time]` and applies:

```text
Conv2D(1, 32, kernel=3×3, stride=1, padding=1)
ReLU
MaxPool2D(kernel=2×2, stride=2)
Conv2D(32, 64, kernel=3×3, stride=1, padding=1)
ReLU
Flatten pooled channel and time positions into one sequence
Causal left padding of 2 samples
Conv1D(64, 64, kernel=3, dilation=1)
Residual addition and ReLU
Global mean over the sequence
Linear(64, number_of_classes)
```

## Frozen model training

- Global and model seed: 2025.
- Optimizer: Adam.
- Learning rate: 0.001.
- Weight decay: 0.
- Loss: multiclass cross-entropy.
- Epochs: 100.
- Early stopping: disabled.
- Nakanishi2015 batch size: 8.
- Lee2019_SSVEP batch size: 2.
- Data-loader workers: 0.
- Device: CUDA, then Apple MPS, then CPU when `device: auto` is selected.

The clean baseline is trained once for each dataset. Each attack scenario starts a fresh robust model rather than continuing from the baseline weights. The robust training set concatenates clean training epochs with the corresponding static adversarial epochs at a 1:1 ratio. Adversarial epochs are generated against the completed clean baseline and are not regenerated after each robust-model update.

## Frozen attacks

All attacks are untargeted white-box attacks using multiclass cross-entropy. They temporarily place the baseline model in evaluation mode, preserve its parameters, and return detached tensors. Input-domain clipping is disabled because normalized EEG has no paper-defined physical bound.

### FGSM

- Epsilon: 0.01.
- One sign-gradient update.
- L-infinity projection around the clean normalized input.

### BIM

- Epsilon: 0.01.
- Step size: 0.001.
- Iterations: 50.
- Initial point: the clean input.
- L-infinity projection after every update.

### PGD

- Epsilon: 0.01.
- Step size: 0.001.
- Iterations: 50.
- Random start: disabled.
- L-infinity projection after every update.

With random start disabled, the frozen BIM and PGD update equations are operationally identical. They remain separate named scenarios because the article reports both.

### MIM

- Epsilon: 0.01.
- Step size: 0.001.
- Momentum decay: 0.5.
- Iterations: 50.
- Initial point: the clean input.
- Each gradient is divided by its per-sample L1 norm before momentum accumulation.
- L-infinity projection after every update.

### Carlini–Wagner

- Untargeted multiclass L2 objective.
- Trade-off constant `c`: 0.001.
- Confidence `kappa`: 10.
- Iterations: 1000.
- Adam learning rate: 0.01.
- Elementwise perturbation limit: 0.01.
- The least-L2 successful adversarial candidate observed during optimization is retained.

The optimizer, learning rate, and elementwise constraint are frozen executable choices around the article's stated C&W values.

## Combined attack scenarios

The five base attacks produce 31 nonempty subsets. For a combination, every constituent delta is generated independently from the same clean input against the same clean baseline:

```text
delta_combined = delta_attack_1 + ... + delta_attack_n
adversarial = clean + delta_combined
```

The combined result receives no final joint projection and no amplitude clipping. Consequently, an `n`-attack scenario can have an elementwise displacement as large as `n × 0.01`. This interpretation follows the additive comparison shown in Figure 1 and preserves each constituent attack's identity. It is not asserted to be the only interpretation compatible with the article.

## Frozen evaluation and analysis

Each dataset records:

- Clean-test performance of the clean baseline.
- Same-scenario attacked-test performance of the clean baseline.
- Clean-test performance of the freshly trained robust model.
- Same-scenario attacked-test performance of the robust model.
- Accuracy.
- Macro one-vs-rest ROC-AUC from the complete probability matrix.
- Cross-entropy where available.
- One scenario-level wall-clock duration covering attack generation, robust training, matched evaluation, and durable artifact writes.

The complete two-dataset metric table is assembled before computational time is normalized. The smallest measured paper-scenario time across both datasets is assigned 100, and every other value is reported as `100 × elapsed / global_minimum`.
