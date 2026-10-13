"""Participant-weighted descriptive figures (Agg; PNG and PDF)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import CLASSES
from features import extract
from shared import CHANNELS, read_recording

COLORS = {"AD": "#bd533f", "FTD": "#8964a9", "HC": "#25897a"}


def save(fig, output, name):
    fig.savefig(Path(output) / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(Path(output) / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def summaries(subjects, spectra, frequency, output, seed):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    curves = []
    # Input spectra already contain the within-participant mean over all epochs.
    for group in CLASSES:
        ids = subjects.loc[subjects.group == group, "participant_id"]
        stack = np.stack([spectra[pid].mean(axis=0) for pid in ids])
        mean, sd = stack.mean(axis=0), stack.std(axis=0, ddof=1)
        lower, upper = np.percentile(stack, [25, 75], axis=0)
        ax.plot(frequency, mean, label=f"{group} (n={len(ids)})", color=COLORS[group], linewidth=2)
        ax.fill_between(frequency, lower, upper, color=COLORS[group], alpha=.13)
        for f, m, s, lo, hi in zip(frequency, mean, sd, lower, upper):
            curves.append({"group": group, "n_participants": len(ids), "frequency_hz": f,
                           "mean_psd_uv2_per_hz": m, "participant_sd_uv2_per_hz": s,
                           "participant_q25_psd_uv2_per_hz": lo, "participant_q75_psd_uv2_per_hz": hi})
    ax.set(xlabel="Frequency (Hz)", ylabel="Power spectral density (µV²/Hz)", xlim=(1, 30), yscale="log",
           title="Eyes-closed EEG: participant-weighted spectra")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.supxlabel("Mean over epochs, then 19 channels, then participants. Lines: arithmetic group means.\n"
                  "Shading: 25th–75th percentiles across participants; descriptive variability, not confidence intervals.", fontsize=9)
    save(fig, output, "group_psd")
    pd.DataFrame(curves).to_csv(Path(output) / "group_psd_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True, layout="constrained")
    rng = np.random.default_rng(seed)
    points = []
    for ax, band in zip(axes, ["theta", "alpha"]):
        for i, group in enumerate(CLASSES):
            selected = subjects.loc[subjects.group == group]
            values = selected[[f"{ch}__relative__{band}" for ch in CHANNELS]].mean(axis=1).to_numpy()
            ax.boxplot([values], positions=[i], widths=.48, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": COLORS[group], "alpha": .16},
                       medianprops={"color": "black"})
            ax.scatter(i + rng.uniform(-.15, .15, len(values)), values, color=COLORS[group], s=24, alpha=.8)
            points.extend({"participant_id": pid, "group": group, "band": band, "relative_power": val}
                          for pid, val in zip(selected.participant_id, values))
        ax.set(xticks=range(3), xticklabels=CLASSES, title=f"{band.capitalize()} ({'4–8' if band == 'theta' else '8–13'} Hz)",
               xlabel="Diagnostic group")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Relative power (fraction of 1–30 Hz power)")
    fig.suptitle("Participant-level relative band power")
    fig.supxlabel("Each point is one participant: epoch mean, then mean across 19 channels.\n"
                  "Boxes: median and interquartile range; whiskers: 1.5 × IQR. Descriptive only.", fontsize=9)
    save(fig, output, "relative_theta_alpha")
    pd.DataFrame(points).to_csv(Path(output) / "relative_theta_alpha_points.csv", index=False)


def representative_qc(root, manifest, intervals, output, config):
    """Show exact matching original time intervals for raw/derivative representatives."""
    root = Path(root)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    psdfig, psdaxes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for col, group in enumerate(CLASSES):
        subject = manifest.loc[(manifest.group == group) & ~manifest.excluded].iloc[0]
        pid = subject.participant_id
        segment_table = intervals.loc[(intervals.participant_id == pid) & intervals.usable]
        later = segment_table.loc[segment_table.original_start_seconds >= 60]
        segment = later.iloc[0] if len(later) else segment_table.iloc[0]
        channel_file = root / pid / "eeg" / f"{pid}_task-eyesclosed_channels.tsv"
        sidecar = root / pid / "eeg" / f"{pid}_task-eyesclosed_eeg.json"
        offset = np.arange(len(CHANNELS))[::-1] * 150.
        for row, source in enumerate(["raw", "derivatives"]):
            path = root / ("" if source == "raw" else "derivatives") / pid / "eeg" / f"{pid}_task-eyesclosed_eeg.set"
            ax = axes[row, col]
            if not path.exists():
                ax.text(.5, .5, f"Raw example unavailable: {pid}", transform=ax.transAxes, ha="center")
                continue
            _, x, fs, _, _ = read_recording(path, channel_file, sidecar)
            start = int(round((segment.original_start_seconds if source == "raw" else segment.source_start_seconds) * fs))
            end = int(round((segment.original_end_seconds if source == "raw" else segment.source_end_seconds) * fs))
            epoch = x[:, start:end]
            time = segment.original_start_seconds + np.arange(epoch.shape[1]) / fs
            for ch, shift in zip(epoch, offset):
                ax.plot(time, ch + shift, linewidth=.5, color=COLORS[group])
            ax.set(yticks=offset, yticklabels=CHANNELS, xlabel="Original-recording time (s)",
                   title=f"{group}: {pid} · {'raw' if source == 'raw' else 'provided derivative'}")
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylim(-300, offset[0] + 300)
            ax.plot([time[-1] - .3, time[-1] - .3], [-150, -100], color="black", linewidth=2)
            ax.text(time[-1] - .5, -125, "50 µV", ha="right", fontsize=8)
            _, frequency, psd, _ = extract(epoch, fs, config)
            psdaxes[col].plot(frequency, psd.mean(axis=0), label=source, linewidth=1.5)
        psdaxes[col].set(title=f"{group}: {pid}", xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)",
                         yscale="log", xlim=(.5, 45))
        psdaxes[col].legend(frameon=False)
    fig.suptitle("Representative matching raw and cleaned intervals")
    fig.supxlabel("Microvolt samples; 150 µV vertical display offsets; fixed display limits may clip large excursions.\n"
                  "Provided derivatives retain ASR/ICA cleaning; traces shown in their supplied reference, without further processing.", fontsize=9)
    save(fig, output, "qc_representative_traces")
    psdfig.supxlabel("Welch PSD of the same source interval; mean across channels. These examples do not establish cohort-wide artifact freedom.", fontsize=9)
    save(psdfig, output, "qc_representative_spectra")


def confusion(oof, output, representation="spectral"):
    from sklearn.metrics import confusion_matrix
    names = list(oof.classifier.unique())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False, layout="constrained")
    for ax, name in zip(axes.ravel(), names):
        selected = oof.loc[oof.classifier == name]
        matrix = confusion_matrix(selected.group, selected.prediction, labels=CLASSES)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=selected.group.value_counts().max())
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel="Predicted group", ylabel="True group", title=name.capitalize())
    fig.suptitle(f"{representation.capitalize()}: pooled out-of-fold participants")
    save(fig, output, f"{representation}_confusion_matrix")
