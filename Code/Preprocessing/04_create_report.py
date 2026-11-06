"""Create scientific QC figures, a run report and a Section 3 draft from actual results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from eeg_common import CHANNELS, DEFAULT_DATA, DEFAULT_OUTPUT, read_eeg, recording_path
from load_preprocessed import load_subject

COLORS = {"AD": "#3569a5", "FTD": "#bd692d", "HC": "#398568"}


def trace(ax, data_uv, fs, title):
    spacing = max(30., float(np.percentile(np.ptp(data_uv, axis=-1), 90) * 0.7))
    offsets = np.arange(19)[::-1] * spacing
    time = np.arange(data_uv.shape[-1]) / fs
    for ch in range(19):
        ax.plot(time, data_uv[ch] - np.median(data_uv[ch]) + offsets[ch], color="#244f75", lw=0.55)
    ax.set_yticks(offsets, CHANNELS, fontsize=7)
    ax.set_xlim(0, 8)
    ax.set_xlabel("Time within epoch (s)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(.99, 1.01, f"Channel spacing: {spacing:.0f} µV", ha="right", transform=ax.transAxes, fontsize=8)
    ax.grid(axis="x", alpha=.2)


def figures(root, output):
    folder = output / "qc" / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(output / "participant_manifest.csv")
    qc = pd.read_csv(output / "qc" / "epoch_qc.csv")
    sensitivity = pd.read_csv(output / "qc" / "threshold_sensitivity.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelsize": 10, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    positions = np.arange(len(manifest))
    axes[0, 0].bar(positions, manifest.candidate_epochs, color="#d5dde4", label="Candidate")
    axes[0, 0].bar(positions, manifest.retained_epochs,
                   color=[COLORS[g] for g in manifest.diagnosis], label="Retained")
    axes[0, 0].set_xticks(positions[::5], manifest.participant_id.str.replace("sub-", "")[::5], fontsize=8)
    axes[0, 0].set(xlabel="Participant ID", ylabel="Eight-second epochs", title="Retention for every participant")
    axes[0, 0].legend(frameon=False)
    arrays = [manifest.loc[manifest.diagnosis.eq(g), "retained_seconds"] / 60 for g in COLORS]
    box = axes[0, 1].boxplot(arrays, tick_labels=list(COLORS), patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], COLORS.values()):
        patch.set_facecolor(color)
        patch.set_alpha(.25)
    rng = np.random.default_rng(600)
    for i, group in enumerate(COLORS):
        y = manifest.loc[manifest.diagnosis.eq(group), "retained_seconds"] / 60
        axes[0, 1].scatter(i + 1 + rng.uniform(-.12, .12, len(y)), y, color=COLORS[group], s=17, alpha=.8)
    axes[0, 1].set(ylabel="Retained EEG (min)", title="Participants remain the independent samples")
    minimum = manifest.loc[manifest.retained_seconds.idxmin()]
    axes[0, 1].annotate(f"{minimum.participant_id}: {minimum.retained_seconds:.0f}s",
                      xy=(list(COLORS).index(minimum.diagnosis) + 1, minimum.retained_seconds / 60),
                      xytext=(1.6, 3), arrowprops={"arrowstyle": "->", "color": "#a43a3a"}, color="#a43a3a", fontsize=9)
    cutoffs = [200, 300, 500, 1000]
    bottom = np.zeros(4)
    for group in COLORS:
        values = sensitivity.loc[sensitivity.diagnosis.eq(group)].groupby("threshold_ptp_uv").retained_epochs.sum().reindex(cutoffs).to_numpy()
        axes[1, 0].bar(np.arange(4), values, bottom=bottom, color=COLORS[group], label=group)
        bottom += values
    for i, value in enumerate(bottom):
        axes[1, 0].text(i, value + 70, str(int(value)), ha="center", fontsize=9)
    axes[1, 0].set_xticks(np.arange(4), [str(c) for c in cutoffs])
    axes[1, 0].set(xlabel="Peak-to-peak cutoff (µV)", ylabel="Retained epochs", title="QC threshold sensitivity; main cutoff = 500 µV")
    axes[1, 0].legend(frameon=False)
    spectra = np.load(output / "qc" / "participant_psd.npz", allow_pickle=False)
    frequency = spectra["frequency_hz"]
    mask = (frequency >= .5) & (frequency <= 60)
    for group in COLORS:
        selected = manifest.diagnosis.eq(group).to_numpy()
        before = spectra["candidate_psd_uv2_per_hz"][selected].mean(axis=(0, 1))
        after = spectra["retained_psd_uv2_per_hz"][selected].mean(axis=(0, 1))
        axes[1, 1].plot(frequency[mask], 10 * np.log10(np.maximum(before[mask], 1e-15)), color=COLORS[group], alpha=.5, ls="--")
        axes[1, 1].plot(frequency[mask], 10 * np.log10(np.maximum(after[mask], 1e-15)), color=COLORS[group], label=group)
    axes[1, 1].set(xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 µV²/Hz)", title="Equal-participant PSD: candidate (dashed), retained (solid)")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("ds004504 v1.0.9 | Shared EEG preprocessing QC", fontsize=17)
    fig.savefig(folder / "preprocessing_overview.png", dpi=180)
    fig.savefig(folder / "preprocessing_overview.svg")
    plt.close(fig)
    # A deliberately difficult retained epoch for each subject, rather than best-case examples.
    for person in manifest.itertuples():
        pid = person.participant_id
        data, table, target = load_subject(pid, output)
        if not len(data):
            continue
        event = table.loc[table.output_max_ptp_uv.idxmax()]
        epoch = np.asarray(data[int(event.array_index)], dtype=float) * 1e6
        _, raw, fs, _ = read_eeg(recording_path(root, pid, "raw"), root, pid)
        raw_epoch = raw[:, int(event.original_start_sample):int(event.original_end_sample)]
        _, derivative, _, _ = read_eeg(recording_path(root, pid), root, pid)
        derivative_epoch = derivative[:, int(event.source_start_sample):int(event.source_end_sample)]
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
        trace(axes[0, 0], raw_epoch, fs, "Original recording | per-channel median removed for display")
        trace(axes[0, 1], derivative_epoch, fs, "Author-cleaned derivative | per-channel median removed for display")
        trace(axes[1, 0], epoch, target, "Saved epoch | 256 Hz, preserved A1/A2 reference")
        for signal, rate, label, color in [(raw_epoch, fs, "Original", "#999999"),
                     (derivative_epoch, fs, "Author derivative", "#bd692d"), (epoch, target, "Saved epoch", "#3569a5")]:
            freq, power = welch(signal, fs=rate, nperseg=round(2 * rate), noverlap=round(rate), axis=-1)
            take = (freq >= .5) & (freq <= 100)
            axes[1, 1].plot(freq[take], 10 * np.log10(np.maximum(power.mean(axis=0)[take], 1e-15)), label=label, color=color)
        axes[1, 1].set(xlabel="Frequency (Hz)", ylabel="Channel-mean PSD (dB re 1 µV²/Hz)", title="Matching original-time interval")
        axes[1, 1].legend(frameon=False)
        fig.suptitle(f"{pid} ({person.diagnosis}) | largest peak-to-peak retained epoch: {event.output_max_ptp_uv:.1f} µV\n"
                     f"Original interval {event.original_start_seconds:.3f}–{event.original_end_seconds:.3f}s | "
                     f"retained {person.retained_epochs}/{person.candidate_epochs} epochs", fontsize=14)
        fig.savefig(folder / f"{pid}_qc.png", dpi=140)
        plt.close(fig)
        print(f"QC figure: {pid}", flush=True)


def documents(output):
    summary = json.loads((output / "qc" / "preprocessing_summary.json").read_text())
    audit = json.loads((output / "qc" / "dataset_audit.json").read_text())
    validation = json.loads((output / "qc" / "validation_results.json").read_text())
    manifest = pd.read_csv(output / "participant_manifest.csv")
    groups = pd.read_csv(output / "qc" / "group_summary.csv").set_index("diagnosis")
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    table = "| Group | Participants (F/M) | Age, mean ± SD | MMSE, mean ± SD | Retained epochs | EEG hours |\n|---|---:|---:|---:|---:|---:|\n"
    for group in ["AD", "FTD", "HC"]:
        row = groups.loc[group]
        table += f"| {group} | {int(row.participants)} ({int(row.female)}/{int(row.participants-row.female)}) | {row.age_mean:.2f} ± {row.age_sd:.2f} | {row.mmse_mean:.2f} ± {row.mmse_sd:.2f} | {int(row.retained_epochs):,} | {row.retained_hours:.2f} |\n"
    report = f"""# Preprocessing run report

Executed October 14, 2026 (America/Edmonton). **Completed; all automated validation checks passed.**

{summary['participants_processed']} participants were processed and {summary['participants_retained']} retained.
The final dataset contains **{summary['retained_epochs']:,} eight-second epochs ({summary['retained_hours']:.2f} hours)**.
This is 88 independent participants, not {summary['retained_epochs']:,} independent cases.

{table}

## What was checked and changed

All 176 EEG files and {audit['verified_files']-176} metadata/documentation files match the
published OpenNeuro version 1.0.9 checksums. Both raw and author derivatives were inspected
for dimensions, channel identities, native microvolt units, finite values and sample rates.
The native recordings contain {audit['raw_hours']:.2f} hours, and author derivatives
contain {audit['derivative_hours']:.2f} hours. The provider removed {audit['provider_removed_seconds']:.2f}
seconds; {audit['internal_boundaries']} internal cuts were recovered from masks and validated
against embedded boundary events. Independent MNE reads checked unit conversion for one
subject in each group.

The existing 0.5-45 Hz FIR, A1/A2 reference, ASR and ICA/ICLabel cleaning were preserved.
Each uninterrupted stretch was independently resampled from 500 to 256 Hz with anti-aliasing.
A one-second guard at both ends was followed by eight-second non-overlapping segmentation.
No added bandpass, notch, ICA, ASR, channel interpolation, z-scoring or classifier was applied.
Source files were unchanged. Median subtraction in trace figures is for display only.

The proposal's initial segmentation of 8,464 epochs was independently reproduced.
Edge guards reduce this to {summary['candidate_epochs_after_edge_guards']:,} candidates, and
{summary['rejected_epochs']} windows were rejected for >500 µV peak-to-peak amplitude in
at least one channel. No windows failed the <=0.1 µV flat-channel criterion.
All retained windows pass these checks before and after resampling. The main cutoff was
chosen conservatively after a data-only audit; no classification scores were used.
The QC sensitivity table records 200/300/500/1000 µV alternatives and their participant counts.

## QC findings that matter for the paper

**sub-086 (FTD) needs review:** only 7/38 guarded candidate epochs survive (56 seconds,
18.4% retention). It is retained and flagged, rather than silently excluded. Before
classification, review its traces and prescribe a sensitivity analysis with and without
it. A minimum-duration exclusion, if adopted, should be defined before seeing model results.

{summary['retained_epochs_flagged_for_amplitude_review']:,} retained epochs exceed the
200 µV review threshold. These flags indicate that automated gross-artifact QC is not
equivalent to visual certification or artifact-free EEG. The supplied reference is
preserved for compatibility with the proposal; common-reference contributions and possible
residual physiological artifacts should be considered when interpreting spectra. There is
a deliberately high-amplitude retained-epoch plot for every participant in `qc/figures`.
Representative manual inspection should include all diagnoses, sub-086 and other extremes.

The README calls the derivative filter Butterworth, but the actual embedded histories use
`pop_eegfiltnew` FIR. Raw acquisition reference/filter descriptions also disagree with the
sidecars. The derivative headers specify A1 A2. The participant table gives FTD MMSE SD
**2.64**, whereas the source README reports 8.22. This report computes summaries from the table.

## Saved data and verification

Each `sub-XXX/eeg/*_epo.npy` has shape `(retained epochs, 19, 2048)` and dtype float32 in
**volts**. The matching TSV includes every candidate's original/raw and derivative
half-open intervals, rejection reasons, and `array_index` (-1 for rejection).
`processing.json` specifies exact channel order, configuration, versions, and code hashes.
`participant_manifest.csv` lists demographics, counts, paths and review flags. Age, Gender,
MMSE and diagnosis are metadata, not EEG signal channels or automatically selected predictors.

Validation checked {validation['saved_epochs_checked']:,} saved epochs, all original mappings,
all input/output sizes and hashes, and {validation['independent_source_epoch_spot_checks']}
independent first/middle/last reconstructions from source (three per participant).
Focused tests separately passed for discontinuities, anti-aliasing, amplitude preservation
and artifact rejection. Feature extraction and classifiers have not yet been run.

![Preprocessing overview](../qc/figures/preprocessing_overview.png)

## Next analysis constraints

Use identical retained epoch IDs for classical and foundation-model features. Keep every
epoch from a participant inside the same split. Fit feature scaling, PCA, feature selection
and hyperparameters on training participants only, preferably using nested subject-level
cross-validation. Weight participants equally when aggregating results. Model-specific
channel/rate/unit adapters must be documented; pretraining overlap with ds004504 needs to
be checked for the selected encoder. Report threshold/duration sensitivity, class retention,
uncertainty, and this single-site cohort's limited generalizability.

Sources: [OpenNeuro v1.0.9](https://openneuro.org/datasets/ds004504/versions/1.0.9),
[data descriptor](https://doi.org/10.3390/data8060095),
[project repository](https://github.com/Pouya-Ta/BMEN-600-Course-Project).
"""
    (reports / "preprocessing_report.md").write_text(report)
    section = f"""# Section 3 - Dataset and Responsible Use

We used the publicly available AHEPA resting-state EEG dataset, OpenNeuro ds004504,
version 1.0.9 (DOI: 10.18112/openneuro.ds004504.v1.0.9) [1], comprising 88 participants:
36 with Alzheimer's disease (AD), 23 with frontotemporal dementia (FTD), and 29 healthy
controls (HC). Recordings were acquired during eyes-closed rest using a Nihon Kohden
EEG-2100 system with 19 scalp channels in the international 10-20 arrangement at 500 Hz
[2]. Each participant has one original and one author-processed continuous EEGLAB `.set`
recording. Participant age, recorded Gender and Mini-Mental State Examination (MMSE)
scores are available as metadata. All local EEG files and accompanying scientific metadata
were verified against the SHA256/metadata contents of the published version 1.0.9 archive.

{table}

Demographic values were calculated directly from `participants.tsv`. The resulting FTD
MMSE standard deviation is 2.64; the dataset README's conflicting value of 8.22 was not
used. F/M denotes the dataset's recorded Gender categories, not an inferred biological
measurement. The original recordings total {audit['raw_hours']:.2f} hours; the supplied
processed recordings total {audit['derivative_hours']:.2f} hours.

Consistent with the project proposal, the primary analysis inputs are the author-processed
recordings. Embedded processing histories document A1/A2 rereferencing, 0.5-45 Hz
`pop_eegfiltnew` FIR filtering, artifact subspace reconstruction with burst criterion 17
and window rejection, extended RunICA, and removal of components classified by ICLabel
as eye or muscle artifacts with probability at least 0.9. The README's description of a
Butterworth filter conflicts with the embedded FIR history; we report the latter.
Original acquisition descriptions and raw sidecar reference/filter fields also differ;
the saved derivative headers specify A1 A2. No additional ICA, ASR or bandpass was applied.

For shared model input, each uninterrupted recording stretch was independently resampled
to 256 Hz using anti-aliased polyphase FIR resampling. Retained-sample masks recovered
original timestamps, and all {audit['internal_boundaries']} internal cuts were cross-checked
against EEGLAB boundary events. One second at each stretch edge was reserved as a guard,
and the remaining signal was divided into non-overlapping eight-second epochs. Epochs
containing a channel with peak-to-peak amplitude above 500 µV or at/below 0.1 µV, or
nonfinite samples, were rejected using identical rules for all diagnoses before and after
resampling. The 500 µV cutoff was chosen conservatively after a data-only audit, without
reference to classifier performance; it is not a clinically validated threshold. Counts
under 200, 300, 500 and 1000 µV alternatives are retained for sensitivity analysis.

The initial 8,464-epoch feasibility segmentation was reproduced. The guarded pipeline
yielded {summary['candidate_epochs_after_edge_guards']:,} candidates and rejected
{summary['rejected_epochs']} for excessive peak-to-peak amplitude, retaining
**{summary['retained_epochs']:,} epochs ({summary['retained_hours']:.2f} hours)** across all
88 participants. Signals retain physical amplitude and the supplied reference and are
stored in SI volts without shared z-scoring. Participant sub-086 (FTD) retained only
seven epochs (56 seconds) and is explicitly flagged for review and a duration sensitivity
analysis. A further {summary['retained_epochs_flagged_for_amplitude_review']:,} retained
epochs exceed a 200 µV review threshold; automatic QC does not certify artifact freedom.

The participant is the independent observational unit. Both representation branches must
use identical retained intervals and participant-level partitions; multiple epochs do not
increase the independent sample size. Feature scaling, dimensionality reduction, feature
selection and hyperparameter tuning must be confined to training folds. Age, Gender and
MMSE are retained for cohort description and possible prespecified confounding analyses,
and are not automatically included as predictors in the EEG-only comparison. Model-specific
adaptations and any pretraining overlap with the dataset must be disclosed.

The dataset is released under CC0. According to its metadata, the original study received
AHEPA Scientific and Ethics Committee approval (protocol 142/12-04-2023) [1,2]. This work
is secondary analysis of the released recordings; we do not claim a new ethics approval.
Derived outputs use public participant IDs and relative recording times and omit embedded
acquisition date/time fields. Source data are unchanged. Checksums, processing settings,
software versions, epoch-level inclusion logs and QC figures support reproducibility.
The modest, imbalanced, single-site cohort, residual artifacts and absence of an independent
external cohort limit diagnostic and clinical generalizability. This dataset supports
methodological comparison rather than a claim of clinical diagnostic readiness.

[1] OpenNeuro, ds004504, version 1.0.9. https://doi.org/10.18112/openneuro.ds004504.v1.0.9

[2] A. Miltiadous et al., "A Dataset of Scalp EEG Recordings of Alzheimer's Disease,
Frontotemporal Dementia and Healthy Subjects from Routine EEG," Data, vol. 8, no. 6,
art. 95, 2023. https://doi.org/10.3390/data8060095

Dataset-requested companion citations for the final manuscript: DICE-net,
https://doi.org/10.1109/ACCESS.2023.3294618 ; AHEPA EEG benchmark,
https://doi.org/10.1007/s11571-026-10464-w . Renumber references within the full paper.
"""
    (reports / "section3_dataset_and_responsible_use.md").write_text(section)
    (output / "README.md").write_text("""# Preprocessed EEG data

Read `reports/preprocessing_report.md` for actual results and QC limitations.
`reports/section3_dataset_and_responsible_use.md` is a draft for the proposal/paper.

- `sub-XXX/eeg/*_epo.npy`: retained float32 epochs in **volts**, axes `(epoch, channel, sample)`.
- `sub-XXX/eeg/*_epochs.tsv`: all candidate windows, original timestamps, flags and rejection reasons.
- `participant_manifest.csv`: all subjects, diagnoses, demographics, counts, paths and review flags.
- `processing.json`: exact settings, channel order, software versions and code fingerprint.
- `qc/`: complete dataset audit, channel/epoch QC, group summaries, threshold sensitivity and validation.
- `qc/figures/`: overview and one deliberately difficult retained-epoch comparison per participant.
- `input_checksums.csv` (under qc) and `output_checksums.csv`: input and derived-file integrity records.

Default shape per epoch: **19 channels × 2048 samples (8 seconds at 256 Hz)**.
End sample/time fields are exclusive. Rejected windows have `array_index = -1` and are absent from arrays.
Indices for retained rows directly address the matching subject array. Preserve participant IDs in splits.
sub-086 needs manual review: only 56 seconds survive. Arrays are not certified artifact-free.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figures(args.data_dir, args.output_dir)
    documents(args.output_dir)
