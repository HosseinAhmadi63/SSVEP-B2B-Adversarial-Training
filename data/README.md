# Data

The repository uses the public `Nakanishi2015` and `Lee2019_SSVEP` loaders supplied by MOABB 1.4.3. No EEG recording is distributed with this source tree.

## Automatic download

After installing the project, download both configured datasets with:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml
```

Download one dataset with:

```bash
python -m ssvep_b2b.cli download --config configs/paper.yaml --dataset Nakanishi2015
python -m ssvep_b2b.cli download --config configs/paper.yaml --dataset Lee2019_SSVEP
```

MOABB downloads source files into `data/raw/`. The command is resumable through MOABB's dataset cache and does not replace existing source archives unless the loader determines that an update is required.

Create the validated local HDF5 caches with:

```bash
python -m ssvep_b2b.cli cache --config configs/paper.yaml
```

The cache files are named `data/cache/Nakanishi2015_{config_hash}.h5` and `data/cache/Lee2019_SSVEP_{config_hash}.h5`; a matching `.metadata.csv` file preserves source-row, participant, session, run, original-label, class-index, and cache-index lineage.

## Frozen dataset records

| Dataset | Paper cohort | EEG channels | Classes | Trials per class | Sessions | Sampling rate | Frozen interval | Frequencies in class-index order |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Nakanishi2015 | 9 subjects | 8 | 12 | 15 | 1 | 256 Hz | 0.15–4.30 s | 9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75 Hz |
| Lee2019_SSVEP | 54 subjects | 62 | 4 | 50 | 2 | 1000 Hz | 0.00–4.00 s | 12.00, 8.57, 6.67, 5.45 Hz |

The resulting validated tensor declarations are:

```text
Nakanishi2015   1,620 epochs × 8 channels × 1,062 samples
Lee2019_SSVEP  10,800 epochs × 62 channels × 4,000 samples
```

For Lee2019_SSVEP, the repository requests labelled training runs from both sessions; the online test runs exposed by the upstream study do not provide labels for this classification workflow. For Nakanishi2015, the paper and frozen configuration use the nine-subject cohort exposed by the pinned MOABB loader.

Each HDF5 cache contains normalized `X`, integer labels `y`, full-cohort `channel_mean` and `channel_scale`, the canonical configuration, and two-harmonic sine/cosine reference signals for every class. Cache generation validates tensor dimensions, balanced class counts, and finite values before the file is accepted.

Cache construction loads continuous MOABB runs, attaches the MNE `standard_1020` montage, applies continuous-run common-average reference, applies MNE's zero-phase Hamming-windowed `firwin` FIR filter from 4 through 16 Hz, creates exact-length event epochs, and finally fits the full-cohort per-channel z-score before the 80/20 split.

## Dataset citations

For Nakanishi2015, cite:

```bibtex
@article{nakanishi2015comparison,
  author  = {Nakanishi, Masaki and Wang, Yijun and Wang, Yu-Te and Jung, Tzyy-Ping},
  title   = {A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials},
  journal = {PLOS ONE},
  year    = {2015},
  volume  = {10},
  number  = {10},
  pages   = {e0140703},
  doi     = {10.1371/journal.pone.0140703}
}
```

For Lee2019_SSVEP, cite the dataset article and supporting-data record:

```bibtex
@article{lee2019eeg,
  author  = {Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
  title   = {EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy},
  journal = {GigaScience},
  year    = {2019},
  volume  = {8},
  number  = {5},
  pages   = {giz002},
  doi     = {10.1093/gigascience/giz002}
}

@dataset{lee2019supporting,
  author    = {Lee, Min-Ho and Kwon, O-Yeon and Kim, Yong-Jeong and Kim, Hong-Kyung and Lee, Young-Eun and Williamson, John and Fazli, Siamac and Lee, Seong-Whan},
  title     = {Supporting Data for EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy},
  publisher = {GigaScience Database},
  year      = {2019},
  doi       = {10.5524/100542}
}
```

## License and access terms

The repository's MIT License covers only the software written for this reimplementation.

- [MOABB](https://github.com/NeuroTechX/moabb) is distributed under the BSD 3-Clause License; that software license does not relicense datasets obtained through its loaders.
- The [MOABB Nakanishi2015 catalog](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Nakanishi2015.html) records the dataset license as unknown. Consult the original UC San Diego distribution and publication before downloading or reusing it.
- The [MOABB Lee2019_SSVEP catalog](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_SSVEP.html) records the dataset license as GPL 3.0 and links the [GigaDB supporting-data record](https://doi.org/10.5524/100542). Retain the notices delivered with the files and verify the current source-record terms before redistribution.

Downloading data signifies neither acceptance by this repository nor a transfer of rights. The original provider terms, privacy statements, attribution requirements, and usage restrictions remain controlling. Raw downloads, processed caches, and generated training artifacts are excluded from Git.
