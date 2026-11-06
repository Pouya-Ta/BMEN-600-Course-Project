# BMEN 600 EEG preprocessing

Read `PREPROCESSING_PLAN.md` for scientific choices and limitations. This workflow uses
the verified author derivatives of OpenNeuro ds004504 v1.0.9; original recordings are
audited and used for QC comparisons. It matches the proposal's eight-second epochs.
Source files are read only. Default paths resolve relative to this code directory,
independent of the current shell directory:

- Inputs: `../Dataset Completed`
- Outputs: `../Outputs/preprocessed data`

## Reproduce

Use Python with `requirements_preprocessing.txt`. These packages were already installed
in the environment used for this run. An optional virtual environment can be used on
another machine. No download is needed for the existing local dataset and stored snapshot.

```bash
python3 -m pip install -r requirements_preprocessing.txt
python3 -m unittest -v test_preprocessing.py
python3 01_data_qc.py
python3 preprocess_eeg.py
python3 03_validate_preprocessed.py
MPLCONFIGDIR=/tmp/bmen600-mpl MPLBACKEND=Agg python3 04_create_report.py
```

All scripts accept `--data-dir` and `--output-dir`. For the audit, `--output-dir` is the
QC subfolder, while the remaining scripts use the root output folder. For example:

```bash
python3 01_data_qc.py --data-dir '/path/to/dataset' --output-dir '/path/to/new-run/qc'
python3 preprocess_eeg.py --data-dir '/path/to/dataset' --output-dir '/path/to/new-run'
python3 03_validate_preprocessed.py --data-dir '/path/to/dataset' --output-dir '/path/to/new-run'
python3 04_create_report.py --data-dir '/path/to/dataset' --output-dir '/path/to/new-run'
```

`preprocess_eeg.py --config custom.json` supports the same keys as the checked-in config.
Different settings, source code, inputs or package versions require a new output directory;
existing scientific results cannot silently change identity. Repeat the audit into the
new run's `qc` folder first. Identical reruns deterministically regenerate signal arrays.
Recorded processing timestamps and ZIP/container metadata need not be byte-identical.

The snapshot archive in `reference/` contains metadata and Git-annex links, not the EEG
samples. It is read without extracting symlinks and provides published version checksums.
Downloaded upstream project scripts are reference material and are never imported or run.

## Load data

```python
from load_preprocessed import load_subject
epochs_v, epoch_metadata, sfreq = load_subject('sub-001')
# epochs_v.shape == (74, 19, 2048) for the default completed run
# Arrays support memory mapping; multiply by 1e6 only when microvolts are required.
# epoch_metadata rows correspond exactly to array rows.
```

For MNE, use `mne.EpochsArray` with `mne.create_info(CHANNELS, sfreq, 'eeg')` and a
standard 10-20 montage if needed. These are physical-amplitude epochs in volts, already
filtered and cleaned upstream. Model-specific rate/channel adapters belong downstream.
Do not infer chronology from an MNE-generated event grid; use the saved original timestamps.

The final output has subject arrays, inclusion TSVs, group/threshold summaries, QC figures,
a report and a Section 3 draft. Review sub-086 before modelling. Keep identical epochs and
participant partitions for both feature branches and fit statistical transforms within folds.

## Files

`eeg_common.py` validates channels, units, participant labels, histories and original masks.
`01_data_qc.py` checks every recording and published input checksum.
`preprocess_eeg.py` creates guarded, anti-aliased, QC-filtered shared epochs.
`03_validate_preprocessed.py` independently checks all outputs and source correspondence.
`04_create_report.py` creates figures and text from actual validated results.
`load_preprocessed.py` supplies aligned arrays and metadata.
`test_preprocessing.py` tests artifact handling, boundaries and resampling on synthetic signals.

Reference project inspected: https://github.com/Pouya-Ta/BMEN-600-Course-Project
(main resolved to commit 7fa3c0491839cd2157bf9bec1be4370b28cafecf on October 14, 2026).
Dataset: https://openneuro.org/datasets/ds004504/versions/1.0.9 .
