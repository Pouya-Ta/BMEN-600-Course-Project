# BMEN 600 preliminary EEG workflow

This additive workflow implements dataset QC, participant-weighted spectral summaries,
and a fixed handcrafted classification baseline. Code and local outputs live in
`preliminary_analysis/`; the existing README, project plans, and analysis scripts are preserved.
The optional frozen BENDR convolutional encoder uses the same intervals and saved participant folds.
No research report, fine-tuning, feature search, connectivity, complexity, or feature fusion is included.

## Reproduce steps 1–3 on CPU

Run from the repository root. Tested with Python 3.12.3; the requirements file pins
the actual baseline environment, including NumPy, SciPy, pandas, scikit-learn, Matplotlib,
requests, and MNE. MNE provides an independent physical-unit check.

```bash
python3.12 -m venv preliminary_analysis/.venv
source preliminary_analysis/.venv/bin/activate
python -m pip install -r preliminary_analysis/requirements.txt
python preliminary_analysis/download.py --data-dir preliminary_analysis/data/ds004504
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/run.py \
  --data-dir preliminary_analysis/data/ds004504 \
  --output-dir preliminary_analysis/results \
  --config preliminary_analysis/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/validate.py \
  --output-dir preliminary_analysis/results
python preliminary_analysis/summarize.py --output-dir preliminary_analysis/results
```

The default download is approximately 2.9 GB: all 88 provided derivative `.set` files,
raw JSON/channel metadata for all participants, and three raw recordings for matching QC examples.
It requires public HTTPS access to GitHub's API/raw content and the OpenNeuro S3 bucket;
no dataset login was required in this run. `--workers` controls download concurrency.
`--no-raw-examples` omits the optional raw trace comparisons. The main classifier always
uses derivatives by default. The download retries transient network failures and uses atomic
replacement only after file-size/SHA256 checks.

Use `--data-dir` and `--output-dir` to choose other locations. For an existing local copy,
point `download.py` at its root: files matching the pinned snapshot are reused; missing or
incorrect files are fetched. A `download_manifest.json` is required for analysis. The output
directory cannot silently switch dataset versions or scientific parameters: choose a new
output directory for a changed configuration. Raw EEG, weights, environments, and large caches
are ignored inside `preliminary_analysis/`; small tables/figures remain available to git.
The large epoch embedding CSV is also ignored, with its participant summary and all comparison
metrics retained. For custom locations inside the repository, arrange equivalent local ignores.

## Exact dataset and provenance

This run used [OpenNeuro ds004504 version 1.0.9](https://openneuro.org/datasets/ds004504/versions/1.0.9),
DOI `10.18112/openneuro.ds004504.v1.0.9`, accessed **October 10, 2026** in America/Edmonton.
The downloader resolves tag `1.0.9` to commit
`63b2ec152bc3cd4487d131124f10f767ebe543bc`, inventories that tree, and obtains metadata
from that commit. Each S3 recording must match its snapshot's git-annex SHA256 key;
the unversioned S3 path alone is insufficient evidence of version identity.
`dataset_provenance.json` records source URLs, checksums, file sizes, version, commit, and access date.

Diagnostic levels come from the dataset's
[participants.json](https://github.com/OpenNeuroDatasets/ds004504/blob/1.0.9/participants.json):
`A` = AD, `F` = FTD, `C` = HC. They are never inferred from numbering. The actual participant
table and downloaded derivatives confirm **36 AD, 23 FTD, 29 HC (88 total)**, with one
derivative recording per participant. Metadata predictors are excluded from both EEG classifiers.
The source field is named `Gender`; F/M summaries report the dataset's recorded sex/gender values.

The [dataset README](https://github.com/OpenNeuroDatasets/ds004504/blob/1.0.9/README)
and embedded EEGLAB histories describe filtering to 0.5–45 Hz, re-reference to A1/A2,
ASR (`BurstCriterion=17`), RunICA, and ICLabel-based component removal. The histories show
`BurstRejection='off'` for ASR reconstruction plus a window rejection criterion, and
ICLabel eye/muscle rejection probabilities 0.9–1. They are retained in `recording_histories.json`
and linked by hash from the manifest. The actual masks show removed periods.

There are documentation discrepancies:

- The README calls the filter Butterworth; **all supplied derivative histories use
  `pop_eegfiltnew`**, an EEGLAB FIR filtering function. We retain the existing filtering.
- Raw acquisition documentation says Cz reference/70 Hz amplifier filtering; raw sidecars
  say A1/A2 and software filters `0.4 - 50`. Derivative headers say `A1 A2`.
  No additional reference change is made by default.
- FTD MMSE SD in `participants.tsv` is **2.64**, whereas the source README says 8.22.
  Tables compute all demographic summaries directly from downloaded metadata.

Raw automatic artifact annotations were omitted by the dataset authors. Supplied cleaning
does not establish artifact freedom; QC is practical and records this limitation.

## Shared preparation and spectral features

`shared.py` loads native EEGLAB microvolt samples, validates them against channel metadata,
checks the complete unique 19-channel montage, and orders channels consistently. The actual
supplied order is Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz.
An independent MNE read checks its volts-to-microvolts conversion against the native samples.
Original files remain unchanged.

The default shared signal retains **500 Hz, supplied A1/A2 reference, and physical amplitude**.
There is no new filtering, ICA, normalization, or detrending of the shared time series.
The derivative `clean_sample_mask` reconstructs original recording indices and is validated
against internal EEGLAB boundary events. Each uninterrupted stretch starts its own grid of
**8-second, non-overlapping epochs**; short tails are omitted. Epochs never span removed data.
Both derivative-source and original-recording start/end times are saved; end times are exclusive.

Candidate epochs containing nonfinite values or any numerically flat channel
(peak-to-peak ≤ 10⁻⁶ µV) are rejected. This is a numerical integrity tolerance, not an
aggressive physiological threshold. A 1,000 µV absolute-amplitude threshold is **only a review
flag**, without automatic amplitude rejection. The manifest and per-channel table record
finite counts, ranges, dispersion, flatness, provider removal, tails, and usable epochs.
Only three matching raw/derivative trace/spectrum examples were visually reviewed; residual
artifacts across the cohort remain a limitation.

Edit `config.json` for epoch duration, shared target sample rate, reference (`preserve` or
`average`), Welch subwindow duration, numerical tolerances, seed, and predeclared classifier
settings. `sampling_hz: null` retains native rate; an explicit target ≥100 Hz uses SciPy
polyphase anti-aliased resampling within each selected interval. The shared normalization
parameter must remain `none` to preserve absolute-power information. Encoder normalization
lives in its adapter. The 8-second default is compatible with the variable-length BENDR CNN,
so no change or subdivision was needed.

`features.py` computes Welch PSD using 2-second Hann subwindows, 50% overlap, constant
subwindow detrending, `nfft=nperseg`, mean averaging, and density scaling (µV²/Hz).
At 500 Hz each subwindow contains 1,000 samples, with 500-sample overlap and 0.5 Hz resolution;
an 8-second epoch contains seven Welch subwindows. Bands are fixed:

| Band | Integration interval |
|---|---|
| Delta | 1–4 Hz |
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |

Power is the trapezoidal integral of the linearly interpolated PSD at exact band endpoints.
Adjacent bands share only a zero-measure endpoint; no frequency bin is double counted.
Log absolute features are `log10(max(power, 10^-12 µV²) / 1 µV²)`.
Relative features divide by the **1–30 Hz integrated power in the same channel and epoch**,
with the same numerical floor. The four relative powers partition that denominator.
The classifier has **19 × 4 × 2 = 152 features**, ordered by channel, feature kind, band;
`feature_schema.json` lists exact column names. Each participant's feature vector is the
arithmetic mean of its epoch vectors, including mean log powers (not log of mean powers).

PSD curves first average epochs within each participant, then channels, then participants
within each group. Lines show arithmetic group means; shading shows participant 25th–75th
percentiles, not confidence intervals. The saved curve table additionally includes participant
SD. Theta/alpha box-and-point plots likewise display one channel-averaged value per participant.
Longer recordings do not carry more weight in group summaries or classifier fitting.

## Shared evaluation

`evaluation.py` saves five stratified participant folds with seed **600**, then reuses them.
Participants, rather than epochs, are the unit of fitting and evaluation. There is one feature
row per participant. Every epoch belonging to a participant inherits that participant's fold.
Multiple recordings are explicitly detected; this snapshot has one per participant, and
unexpected duplicates are excluded pending an explicit aggregation decision.

The fixed baseline is StandardScaler plus **L2 multinomial logistic regression, C=1.0,
lbfgs, intercept enabled, no class weighting**, with training-only median imputation and
scaling inside a scikit-learn pipeline. There is no PCA, selection, tuning, or use of age,
sex/gender, or MMSE. The same-fold dummy uses training class priors and predicts the most
frequent training class. Convergence warnings are treated as errors.

Metrics include balanced accuracy (primary), macro-F1, per-class recall, each fold, their
mean and sample SD (`ddof=1`), pooled OOF metrics, confusion matrices, and participant-level
OOF class probabilities. **Fold SD is descriptive split variability, not a confidence
interval.** Three-class balanced-accuracy chance is 1/3. No interval or hypothesis test is
estimated from folds/epochs as though they were independent samples.

## Optional frozen encoder

The candidate was inspected before settling preparation defaults. It is the original
[BENDR official pretrained release](https://github.com/SPOClab-ca/BENDR/releases/tag/v0.1-alpha),
published February 25, 2021. This workflow uses only `encoder.pt`, strictly loaded into the
official `ConvEncoderBENDR(20, encoder_h=512)`; it does not extract class logits or initialize
substitute random weights. Checkpoint, source commits, downloaded source hashes, and release
asset identity are recorded. The original release has no independent published digest;
the locally computed SHA256 identifies the downloaded asset and protects cache reuse.

Official [pretraining configuration](https://github.com/SPOClab-ca/BENDR/blob/v0.1-alpha/configs/pretraining.yml)
uses TUEG v1.1/v1.2 and BCI IV 2a validation; documented downstream tasks include motor imagery,
ERN, P300, and sleep staging. No ds004504 pretraining/evaluation overlap is documented, and
the release predates this dataset's 2023 publication. This is evidence for no known overlap,
not an exhaustive participant identity audit. No ds004504-fine-tuned checkpoint is used.

```bash
# Optional, after the baseline outputs exist; actual run used CPU.
python -m pip install -r preliminary_analysis/requirements-encoder.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/encoder.py \
  --data-dir preliminary_analysis/data/ds004504 \
  --output-dir preliminary_analysis/results --device cpu
python preliminary_analysis/validate.py --output-dir preliminary_analysis/results
python preliminary_analysis/summarize.py --output-dir preliminary_analysis/results
```

The optional requirement pins a CPU PyTorch wheel. A compatible CUDA PyTorch installation
can use `--device cuda`, but the current computer's `nvidia-smi` reports
`Failed to initialize NVML: Driver/library version mismatch` (NVML 595.99).
No drivers or unrelated environments were changed. CPU extraction was practical.

`encoder.py` downloads pinned original BENDR/DN3 source and selects **unmodified official
model/transform definitions** using Python AST. This avoids importing the obsolete DN3
training framework; the legacy MNE constant imports are supplied from current MNE.
The model loads every checkpoint key strictly. All weights are frozen, evaluation mode is
enabled, and inference uses no gradients and deterministic algorithms.

The adapter follows the official DN3 transformation order: microvolts → volts, Deep1010
mapping and global per-window EEG minmax to [-1,1], nearest temporal interpolation to
**256 Hz**, then official `To1020` ordering. The source already has a 45 Hz low-pass;
nearest interpolation matches the official implementation. T3/T4 are explicitly renamed
to their T7/T8 sensor equivalents before mapping, so both sensors are retained. Input order:
Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, T5, P3, Pz, P4, T6, O1, O2, SCALE.
SCALE is **zero**, following the official behavior when dataset amplitude bounds are
unspecified; no extrema are estimated from the held-out cohort. This loses encoder amplitude
scale information and is a documented transfer limitation; the shared handcrafted data retain it.

Each matching shared 8-second epoch becomes 20 × 2,048 samples. The final CNN output is
512 × 22; its temporal mean is a **512-dimensional representation**, followed by the
participant's arithmetic epoch mean. The transformer is not used. The pretraining window
was 60 seconds, but the CNN accepts variable-length input and official downstream tasks
use shorter windows. The smoke test checks finite shapes, strict loading, frozen evaluation,
repeat inference, variation between inputs, and representation rather than logits.
Embedding caches bind to checkpoint/source identity, adapter settings, baseline fingerprint,
and the shared interval table. Both feature sets use corresponding source time intervals.
The same saved folds and classifier protocol produce paired scores and a comparison table.

## Output map and validation

All paths below are relative to `preliminary_analysis/results/`.

| Output | Meaning |
|---|---|
| `analysis_config.json`, `software_versions.json`, `dataset_provenance.json` | Scientific settings, code/version fingerprints, dataset identity and verified downloads |
| `participant_manifest.csv`, `participant_definitions.json` | Demographics, labels, source paths/hash, channels/units, lengths, retained epochs, missingness, exclusions |
| `channel_qc.csv`, `qc_summary.json`, `recording_histories.json` | Numeric channel audit, matching/QC outcomes, supplied preprocessing histories |
| `group_summary.csv`, `group_summary_compact.csv` | Numeric report-ready group summary and compact presentation table; durations refer to included participants, demographics to enrolled participants |
| `shared_intervals.csv` | Candidate intervals, rejection status/reasons, source/original time mapping; usable rows define both representations |
| `epoch_features.csv.gz`, `participant_features.csv`, `feature_schema.json` | Channel-level spectral vectors and their participant means |
| `participant_psd.npz`, `group_psd_summary.csv` | Within-participant PSD by channel/frequency, group mean/SD/percentile curves |
| `group_psd.*`, `relative_theta_alpha.*`, `relative_theta_alpha_points.csv` | Participant-weighted spectra and relative-power distributions (PNG and PDF) |
| `qc_representative_traces.*`, `qc_representative_spectra.*` | Matching raw/cleaned examples for one participant per group |
| `fold_assignments.csv` | Shared participant ID/group/fold assignments (folds 0–4) |
| `spectral_fold_scores.csv`, `spectral_metrics.csv`, `spectral_oof_predictions.csv` | Fixed baseline/dummy fold metrics, summaries, one OOF row per participant per classifier |
| `spectral_*_confusion_matrix.csv`, `spectral_confusion_matrix.*` | Pooled OOF counts with explicit class labels |
| `encoder_config.json`, `encoder_smoke_test.json` | Checkpoint identity, exact adaptation/pooling, successful frozen inference checks |
| `encoder_epoch_embeddings.csv.gz`, `encoder_participant_embeddings.csv` | Matching interval embeddings and participant means; large epoch CSV is local/ignored |
| `bendr_*`, `paired_fold_scores.csv`, `representation_comparison.csv` | Encoder evaluation, paired per-fold comparisons, descriptive comparison table |
| `validation_checks.json`, `spectral_validation.json`, `bendr_validation.json` | Numerical and real-data integrity checks; runtime training-only preprocessing checks |
| `reproducibility_check.json` | Hash comparisons from a repeated run; scientific tables/predictions/figures reproduce exactly |
| `RESULTS_NOTES.md` | Short factual results/limitations handoff note, generated from actual output tables |
| `cache/` | Ignored configuration/source-bound per-recording feature/embedding caches |

`validate.py` checks real subject coverage/folds, finite dimensions, equal-epoch averaging,
original-time nonoverlap, preserved units, excluded metadata predictors, and complete OOF
probability coverage. The classifier checks its fitted imputer/scaler against the training
participants on every fold. Numerical smoke checks use a labelled synthetic signal/data
only to verify integration, amplitude scaling, gap handling, rejection, and fitting mechanics.
**Synthetic checks are not preliminary scientific results.**

To repeat the saved configuration and check identical scientific tables, predictions,
and figure files in the tested environment (using checked caches), run:

```bash
python preliminary_analysis/check_reproducibility.py \
  --data-dir preliminary_analysis/data/ds004504 \
  --output-dir preliminary_analysis/results
```

The repeat check regenerates baseline outputs and, when present, encoder outputs. Encoder
inference is freshly checked on the first participant; remaining valid embedding caches
are reused. Execution date/cache-hit counts may change; scientific QC fields must agree.

Steps 1–3 are complete on the actual 88-participant snapshot, with all requested baseline
artifacts saved. The optional frozen encoder was also run; see `RESULTS_NOTES.md` for measured
results. This supports a preliminary feasibility discussion for October 16, 2026, with
single-site and small-sample limitations, weak FTD recall, residual artifacts, and provenance
uncertainties. It does not establish clinical validation or superiority to other study protocols.
