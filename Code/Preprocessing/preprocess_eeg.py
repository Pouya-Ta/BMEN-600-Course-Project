"""Prepare shared, physical-amplitude EEG epochs from verified author derivatives.

Resampling is performed independently within each uninterrupted stretch. No sample
from another participant or across an ASR cut enters a resampling convolution.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import resample_poly, welch

from eeg_common import (CHANNELS, DEFAULT_DATA, DEFAULT_OUTPUT, HERE, original_indices,
                        participants, read_eeg, recording_path, sha256,
                        verify_provider_history, write_json)


def config_from(path):
    config = json.loads(Path(path).read_text())
    expected = set(json.loads((HERE / "preprocessing_config.json").read_text()))
    if set(config) != expected:
        raise ValueError("Configuration keys must match preprocessing_config.json")
    for key in ["target_sfreq_hz", "epoch_seconds", "boundary_guard_seconds",
                "reject_peak_to_peak_uv", "reject_flat_peak_to_peak_uv", "review_peak_to_peak_uv",
                "minimum_retained_seconds_review", "minimum_retention_fraction_review"]:
        if not isinstance(config[key], (float, int)) or not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid numeric parameter {key}")
    if config["source"] != "derivatives" or config["dataset_id"] != "ds004504" or config["dataset_version"] != "1.0.9":
        raise ValueError("Only the audited ds004504 v1.0.9 author derivatives are supported")
    if config["reference"] not in ["preserve", "average"] or config["normalization"] != "none":
        raise ValueError("Reference must be preserve/average; shared normalization must be none")
    if not 100 <= config["target_sfreq_hz"] <= 500 or config["epoch_seconds"] <= 0:
        raise ValueError("Require 100 <= target frequency <= 500 Hz and positive epoch duration")
    if not 0 <= config["minimum_retention_fraction_review"] <= 1:
        raise ValueError("Invalid retention review fraction")
    if not 0 < config["reject_flat_peak_to_peak_uv"] < config["review_peak_to_peak_uv"] < config["reject_peak_to_peak_uv"]:
        raise ValueError("Flat < review < rejection thresholds are required")
    for value in [config["epoch_seconds"], config["boundary_guard_seconds"]]:
        for rate in [500, config["target_sfreq_hz"]]:
            if not np.isclose(value * rate, round(value * rate)):
                raise ValueError("Epoch duration and guard must align to source and target samples")
    return config


def window_starts(left, right, fs, config):
    """Half-open windows with an edge guard at both ends of a contiguous run."""
    guard = round(config["boundary_guard_seconds"] * fs)
    width = round(config["epoch_seconds"] * fs)
    return range(int(left + guard), int(right - guard - width + 1), width)


def quality(epoch_uv, config):
    if not np.isfinite(epoch_uv).all():
        return ["nonfinite"], float("nan"), float("nan")
    ptp = np.ptp(epoch_uv, axis=-1)
    reasons = []
    if ptp.max() > config["reject_peak_to_peak_uv"]:
        reasons.append("high_peak_to_peak")
    if ptp.min() <= config["reject_flat_peak_to_peak_uv"]:
        reasons.append("flat_channel")
    return reasons, float(ptp.max()), float(ptp.min())


def resample_run(data_uv, fs, target):
    if fs == target:
        return data_uv.copy()
    ratio = Fraction(target / fs).limit_denominator(10000)
    # scipy's Kaiser FIR provides anti-aliasing, with centered output and line padding.
    return resample_poly(data_uv, ratio.numerator, ratio.denominator,
                         axis=-1, window=("kaiser", 5.0), padtype="line")


def run(root, output, config_path):
    started = datetime.now(ZoneInfo("America/Edmonton"))
    root, output = Path(root).resolve(), Path(output).resolve()
    config = config_from(config_path)
    qc = output / "qc"
    audit_path = qc / "dataset_audit.json"
    if not audit_path.exists():
        raise ValueError("Run 01_data_qc.py first, with its output under <output-dir>/qc")
    audit = json.loads(audit_path.read_text())
    if not audit["all_checksums_match"] or audit["version"] != config["dataset_version"]:
        raise ValueError("A successful matching dataset audit is required")
    for row in pd.read_csv(qc / "input_checksums.csv").itertuples():
        path = root / row.path
        if path.stat().st_size != row.size_bytes or sha256(path) != row.sha256:
            raise ValueError(f"Input changed since audit: {row.path}")
    inventory = pd.read_csv(qc / "dataset_inventory.csv").set_index("participant_id")
    people = participants(root)
    versions = {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "pandas", "mne", "matplotlib"]}
    code_names = ["eeg_common.py", "01_data_qc.py", "preprocess_eeg.py", "03_validate_preprocessed.py"]
    specification = {"config": config, "software_versions": versions, "python_version": platform.python_version(),
                     "code_sha256": {name: sha256(HERE / name) for name in code_names},
                     "input_checksums_sha256": sha256(qc / "input_checksums.csv"),
                     "units": "V", "dtype": "float32", "axes": ["epoch", "channel", "sample"],
                     "channel_order": CHANNELS,
                     "resampling": "scipy.signal.resample_poly; Kaiser beta=5.0; line padding; each contiguous run independently",
                     "provider_cleaning": "0.5-45 Hz pop_eegfiltnew FIR; A1 A2; ASR burst=17 and window rejection; extended RunICA; ICLabel eye/muscle >=0.9 removal",
                     "added_bandpass": None, "added_notch": None, "added_ica_or_asr": None,
                     "metadata_used_as_predictors": [], "training_dependent_transforms": [],
                     "input_root": str(root), "executable": sys.executable}
    identity = __import__("hashlib").sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    processing_path = output / "processing.json"
    if processing_path.exists():
        previous = json.loads(processing_path.read_text())
        if previous["fingerprint"] != identity:
            raise ValueError("Existing output has different inputs, code or settings; choose a new --output-dir and audit there")
    write_json(processing_path, {"fingerprint": identity, "status": "running", "started_at": started.isoformat(), **specification})
    manifest_rows, all_epochs, sensitivity, output_hashes = [], [], [], []
    spectra_before, spectra_after = [], []
    for person in people.to_dict("records"):
        pid = person["participant_id"]
        eeg, data, fs, info = read_eeg(recording_path(root, pid), root, pid)
        verify_provider_history(eeg)
        original, edges = original_indices(eeg, data.shape[1], int(inventory.loc[pid, "raw_samples"]))
        if config["reference"] == "average":
            data -= data.mean(axis=0, keepdims=True)
        target = config["target_sfreq_hz"]
        width = round(config["epoch_seconds"] * fs)
        out_width = round(config["epoch_seconds"] * target)
        arrays, rows, psds_before, psds_after = [], [], [], []
        for run_id, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            starts = window_starts(left, right, fs, config)
            if not len(starts):
                continue
            transformed = resample_run(data[:, left:right], fs, target)
            for start in starts:
                end = start + width
                offset = (start - left) / fs * target
                if not np.isclose(offset, round(offset)):
                    raise ValueError("Epoch grid did not align with target samples")
                out_start = round(offset)
                source_epoch = data[:, start:end]
                output_epoch = transformed[:, out_start:out_start + out_width]
                if output_epoch.shape != (19, out_width):
                    raise ValueError("Incomplete resampled epoch")
                reasons, max_ptp, min_ptp = quality(source_epoch, config)
                out_reasons, out_max, out_min = quality(output_epoch, config)
                reasons = sorted(set(reasons + out_reasons))
                retained = not reasons
                slot = len(arrays) if retained else -1
                if retained:
                    arrays.append((output_epoch * 1e-6).astype(np.float32))
                freq, psd = welch(output_epoch, fs=target, window="hann", nperseg=round(2 * target),
                                  noverlap=round(target), detrend="constant", scaling="density", axis=-1)
                psds_before.append(psd)
                if retained:
                    psds_after.append(psd)
                if original[end - 1] - original[start] != width - 1:
                    raise ValueError("Epoch unexpectedly bridges a removed interval")
                rows.append({"participant_id": pid, "diagnosis": person["diagnosis"],
                             "epoch_id": f"{pid}_run-{run_id:03d}_start-{start:07d}",
                             "run_id": run_id, "source_start_sample": start, "source_end_sample": end,
                             "original_start_sample": int(original[start]),
                             "original_end_sample": int(original[end - 1] + 1),
                             "original_start_seconds": original[start] / fs,
                             "original_end_seconds": (original[end - 1] + 1) / fs,
                             "run_source_start_sample": left, "run_source_end_sample": right,
                             "source_max_ptp_uv": max_ptp, "source_min_ptp_uv": min_ptp,
                             "output_max_ptp_uv": out_max, "output_min_ptp_uv": out_min,
                             "review_high_amplitude": max(max_ptp, out_max) > config["review_peak_to_peak_uv"],
                             "retained": retained, "rejection_reason": ";".join(reasons),
                             "array_index": slot})
        if not rows:
            raise ValueError(f"No full guarded epochs: {pid}")
        epochs = pd.DataFrame(rows)
        folder = output / pid / "eeg"
        folder.mkdir(parents=True, exist_ok=True)
        array_path = folder / f"{pid}_task-eyesclosed_desc-clean_epo.npy"
        table_path = folder / f"{pid}_task-eyesclosed_desc-epochs.tsv"
        array = np.stack(arrays) if arrays else np.empty((0, 19, out_width), dtype=np.float32)
        temporary = array_path.with_suffix(".partial.npy")
        np.save(temporary, array, allow_pickle=False)
        temporary.replace(array_path)
        epochs.to_csv(table_path, sep="\t", index=False)
        n = len(array)
        fraction = n / len(epochs)
        reasons = []
        if n * config["epoch_seconds"] < config["minimum_retained_seconds_review"]:
            reasons.append("short_retained_duration")
        if fraction < config["minimum_retention_fraction_review"]:
            reasons.append("low_retention")
        if epochs.review_high_amplitude.any():
            reasons.append("residual_amplitude_review")
        manifest_rows.append({**person, "included": n > 0, "raw_seconds": inventory.loc[pid, "raw_seconds"],
                              "provider_derivative_seconds": data.shape[1] / fs,
                              "provider_removed_seconds": inventory.loc[pid, "provider_removed_seconds"],
                              "baseline_8s_epochs": inventory.loc[pid, "baseline_8s_epochs"],
                              "contiguous_runs": len(edges) - 1, "candidate_epochs": len(epochs),
                              "retained_epochs": n, "rejected_epochs": len(epochs) - n,
                              "retained_seconds": n * config["epoch_seconds"], "retention_fraction": fraction,
                              "edge_and_tail_seconds": data.shape[1] / fs - len(epochs) * config["epoch_seconds"],
                              "review_epoch_count": int((epochs.retained & epochs.review_high_amplitude).sum()),
                              "review_reasons": ";".join(reasons),
                              "output_sfreq_hz": target, "array_path": str(array_path.relative_to(output)),
                              "epoch_table_path": str(table_path.relative_to(output)),
                              "remaining_provider_ica_components": int(np.asarray(eeg["icaweights"]).shape[0])})
        for cutoff in [200, 300, 500, 1000]:
            valid = (epochs.source_max_ptp_uv <= cutoff) & (epochs.output_max_ptp_uv <= cutoff)
            valid &= (epochs.source_min_ptp_uv > config["reject_flat_peak_to_peak_uv"])
            valid &= (epochs.output_min_ptp_uv > config["reject_flat_peak_to_peak_uv"])
            sensitivity.append({"participant_id": pid, "diagnosis": person["diagnosis"],
                                "threshold_ptp_uv": cutoff, "candidate_epochs": len(epochs),
                                "retained_epochs": int(valid.sum()), "retained_seconds": float(valid.sum() * config["epoch_seconds"])})
        spectra_before.append(np.mean(psds_before, axis=0))
        spectra_after.append(np.mean(psds_after, axis=0) if psds_after else np.full((19, len(freq)), np.nan))
        for path in [array_path, table_path]:
            output_hashes.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "size_bytes": path.stat().st_size})
        all_epochs.append(epochs)
        print(f"{pid} {person['diagnosis']}: {n}/{len(epochs)} retained, {n*config['epoch_seconds']:.0f}s", flush=True)
    manifest = pd.DataFrame(manifest_rows)
    epochs = pd.concat(all_epochs, ignore_index=True)
    manifest.to_csv(output / "participant_manifest.csv", index=False)
    people.to_csv(output / "participants.tsv", sep="\t", index=False)
    epochs.to_csv(qc / "epoch_qc.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(qc / "threshold_sensitivity.csv", index=False)
    np.savez_compressed(qc / "participant_psd.npz", frequency_hz=freq, channels=np.array(CHANNELS),
                        participant_ids=people.participant_id.to_numpy(dtype=str),
                        candidate_psd_uv2_per_hz=np.stack(spectra_before),
                        retained_psd_uv2_per_hz=np.stack(spectra_after))
    groups = manifest.groupby("diagnosis").agg(participants=("participant_id", "size"),
                included_participants=("included", "sum"), female=("Gender", lambda x: int(x.eq("F").sum())),
                age_mean=("Age", "mean"), age_sd=("Age", "std"), mmse_mean=("MMSE", "mean"), mmse_sd=("MMSE", "std"),
                candidate_epochs=("candidate_epochs", "sum"), retained_epochs=("retained_epochs", "sum"),
                retained_seconds=("retained_seconds", "sum"), rejected_epochs=("rejected_epochs", "sum"))
    groups["retained_hours"] = groups.retained_seconds / 3600
    groups["epoch_retention_fraction"] = groups.retained_epochs / groups.candidate_epochs
    groups.to_csv(qc / "group_summary.csv")
    summary = {"participants_processed": len(manifest), "participants_retained": int(manifest.included.sum()),
               "baseline_epochs_in_proposal": int(manifest.baseline_8s_epochs.sum()),
               "candidate_epochs_after_edge_guards": len(epochs), "retained_epochs": int(epochs.retained.sum()),
               "rejected_epochs": int((~epochs.retained).sum()),
               "retained_hours": float(epochs.retained.sum() * config["epoch_seconds"] / 3600),
               "rejected_high_amplitude_epochs": int(epochs.rejection_reason.str.contains("high_peak_to_peak").sum()),
               "rejected_flat_epochs": int(epochs.rejection_reason.str.contains("flat_channel").sum()),
               "retained_epochs_flagged_for_amplitude_review": int((epochs.retained & epochs.review_high_amplitude).sum()),
               "participants_flagged_short_or_low_retention": manifest.loc[manifest.review_reasons.str.contains("short_retained|low_retention"), "participant_id"].tolist(),
               "group_counts_retained": manifest.loc[manifest.included, "diagnosis"].value_counts().to_dict(),
               "original_hours": audit["raw_hours"], "provider_derivative_hours": audit["derivative_hours"],
               "output_array_total_bytes": int(sum(x["size_bytes"] for x in output_hashes if x["path"].endswith(".npy"))),
               "fingerprint": identity}
    write_json(qc / "preprocessing_summary.json", summary)
    for path in [output / "participant_manifest.csv", output / "participants.tsv", qc / "epoch_qc.csv",
                 qc / "threshold_sensitivity.csv", qc / "participant_psd.npz", qc / "group_summary.csv", qc / "preprocessing_summary.json"]:
        output_hashes.append({"path": str(path.relative_to(output)), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    pd.DataFrame(output_hashes).to_csv(output / "output_checksums.csv", index=False)
    write_json(output / "dataset_description.json", {
        "Name": "BMEN 600 shared EEG epochs derived from OpenNeuro ds004504 v1.0.9",
        "DatasetType": "derivative", "License": "CC0", "SourceDatasetDOI": "10.18112/openneuro.ds004504.v1.0.9",
        "FormatNote": "Project-specific NumPy derivatives; this folder is not claimed to be BIDS compliant.",
        "GeneratedBy": [{"Name": "BMEN600 preprocessing", "Version": "1.0", "CodeFingerprint": identity}]})
    write_json(processing_path, {"fingerprint": identity, "status": "complete", "started_at": started.isoformat(),
                                "finished_at": datetime.now(ZoneInfo("America/Edmonton")).isoformat(), **specification})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=HERE / "preprocessing_config.json")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.config)
