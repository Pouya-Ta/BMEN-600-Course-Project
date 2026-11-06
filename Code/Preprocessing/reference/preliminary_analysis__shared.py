"""Amplitude-preserving EEGLAB loading and contiguous, shared interval selection."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]


def participant_metadata(root):
    root = Path(root)
    frame = pd.read_csv(root / "participants.tsv", sep="\t", na_values=["n/a"])
    definitions = json.loads((root / "participants.json").read_text())
    # Decode documented levels, never infer diagnoses from participant numbering.
    levels = definitions["Group"]["Levels"]
    diagnoses = {}
    for code, description in levels.items():
        lower = description.lower()
        if "alzheimer" in lower:
            diagnoses[code] = "AD"
        elif "frontotemporal" in lower:
            diagnoses[code] = "FTD"
        elif "healthy" in lower:
            diagnoses[code] = "HC"
        else:
            raise ValueError(f"Unrecognized documented diagnosis: {description}")
    if not frame.participant_id.is_unique or frame.participant_id.isna().any():
        raise ValueError("Missing/duplicate participant identifiers")
    frame["group"] = frame.Group.map(diagnoses)
    if frame.group.isna().any():
        raise ValueError("Missing/undocumented diagnostic labels")
    for col in ["Age", "MMSE"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    allowed_sex = definitions["Gender"]["Levels"]
    if not frame.Gender.dropna().isin(allowed_sex).all():
        raise ValueError("Undocumented sex/gender values")
    return frame.sort_values("participant_id").reset_index(drop=True), definitions


def read_recording(path, channels_path, sidecar_path):
    """EEGLAB native samples are microvolts; retain physical amplitudes as float64."""
    eeg = loadmat(path, simplify_cells=True)
    eeg = eeg.get("EEG", eeg)
    x = np.asarray(eeg["data"], dtype=np.float64)
    fs = float(eeg["srate"])
    labels = [str(ch["labels"]) for ch in eeg["chanlocs"]]
    if x.shape != (int(eeg["nbchan"]), int(eeg["pnts"])) or int(eeg["trials"]) != 1:
        raise ValueError("Expected continuous channels × samples EEGLAB data")
    table = pd.read_csv(channels_path, sep="\t")
    if labels != table.name.tolist():
        raise ValueError("EEGLAB channel order disagrees with channels.tsv")
    if set(labels) != set(CHANNELS) or len(labels) != len(set(labels)):
        raise ValueError("Expected the complete, unique 19-channel montage")
    if not table.type.eq("EEG").all() or not table.units.isin(["microV", "uV", "µV"]).all():
        raise ValueError("EEG microvolt physical units must be documented")
    sidecar = json.loads(Path(sidecar_path).read_text())
    if fs != float(sidecar["SamplingFrequency"]) or len(labels) != int(sidecar["EEGChannelCount"]):
        raise ValueError("Header and raw sidecar sampling rate/channel count disagree")
    # Canonical order is independent of source ordering and of any encoder order.
    x = x[[labels.index(ch) for ch in CHANNELS]]
    return eeg, x, fs, labels, sidecar


def original_sample_indices(eeg, n_samples, source):
    """Recover raw timeline from derivative clean_sample_mask and check boundaries."""
    events = eeg.get("event", [])
    if isinstance(events, dict):
        events = [events]
    boundaries = sorted(int(np.floor(float(event["latency"]))) for event in events
                        if str(event.get("type", "")).lower() == "boundary")
    if source == "raw":
        if boundaries:
            raise ValueError("Raw recording has unexplained discontinuities")
        return np.arange(n_samples), "raw_recording", 0
    mask = np.asarray(eeg.get("etc", {}).get("clean_sample_mask", []), dtype=bool).ravel()
    if not mask.size or mask.sum() != n_samples:
        raise ValueError("Derivative lacks a consistent original-sample mapping")
    indices = np.flatnonzero(mask)
    breaks = (np.flatnonzero(np.diff(indices) != 1) + 1).tolist()
    if breaks != [b for b in boundaries if 0 < b < n_samples]:
        raise ValueError("EEGLAB boundary events disagree with retained sample mask")
    return indices, "raw_recording_via_clean_sample_mask", int(mask.size - n_samples)


def prepare(x, fs, config):
    if config["normalization"] != "none":
        raise ValueError("Shared signal normalization must preserve physical amplitudes (none)")
    if config["reference"] == "average":
        x = x - x.mean(axis=0, keepdims=True)
    elif config["reference"] != "preserve":
        raise ValueError("Unknown reference")
    return x


def select_intervals(x, fs, original_indices, config):
    """Restart the non-overlapping epoch grid at each gap; never bridge a boundary."""
    samples = int(round(config["epoch_seconds"] * fs))
    if samples < 1 or not np.isclose(samples / fs, config["epoch_seconds"]):
        raise ValueError("Epoch length must be an integer number of source samples")
    edges = np.r_[0, np.flatnonzero(np.diff(original_indices) != 1) + 1, x.shape[1]]
    rows = []
    for run, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        for start in range(int(left), int(right) - samples + 1, samples):
            stop = start + samples
            segment = x[:, start:stop]
            reasons = []
            if not np.isfinite(segment).all():
                reasons.append("nonfinite_samples")
            elif (np.ptp(segment, axis=1) <= config["flat_ptp_uv"]).any():
                reasons.append("numerically_flat_channel")
            # Amplitude alerts are review flags; they do not reject data.
            amplitude = float(np.max(np.abs(segment))) if np.isfinite(segment).all() else np.nan
            rows.append({"run": run, "source_start_sample": start, "source_end_sample": stop,
                         "source_start_seconds": start / fs, "source_end_seconds": stop / fs,
                         "original_start_seconds": original_indices[start] / fs,
                         "original_end_seconds": (original_indices[stop - 1] + 1) / fs,
                         "usable": not reasons, "rejection_reason": ";".join(reasons),
                         "max_abs_uv": amplitude,
                         "amplitude_review_flag": amplitude > config["amplitude_warning_uv"]})
    columns = ["run", "source_start_sample", "source_end_sample", "source_start_seconds", "source_end_seconds",
               "original_start_seconds", "original_end_seconds", "usable", "rejection_reason", "max_abs_uv",
               "amplitude_review_flag"]
    return pd.DataFrame(rows, columns=columns)


def iter_epochs(x, fs, intervals, config):
    """Shared epochs in µV; optional anti-aliased resampling preserves amplitude scale."""
    target = config["sampling_hz"] or fs
    ratio = Fraction(target / fs).limit_denominator(10000)
    for row in intervals.loc[intervals.usable].itertuples():
        epoch = x[:, row.source_start_sample:row.source_end_sample]
        if target != fs:
            epoch = resample_poly(epoch, ratio.numerator, ratio.denominator, axis=-1)
        yield row, epoch, target
