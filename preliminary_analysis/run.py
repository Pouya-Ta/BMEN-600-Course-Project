"""Run the real-data QC audit, spectral summaries, and participant-only baseline."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from download import sha256
from evaluation import CLASSES, evaluate, saved_folds
from features import BANDS, extract, feature_names
from plots import confusion, representative_qc, summaries
from shared import CHANNELS, iter_epochs, original_sample_indices, participant_metadata, prepare, read_recording, select_intervals

HERE = Path(__file__).resolve().parent
DEFAULTS = {"epoch_seconds": 8., "sampling_hz": None, "reference": "preserve", "normalization": "none",
            "welch_seconds": 2., "power_floor_uv2": 1e-12, "flat_ptp_uv": 1e-6,
            "amplitude_warning_uv": 1000., "seed": 600, "logistic_C": 1., "class_weight": None,
            "source": "derivatives"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def group_summary(manifest):
    records = []
    for group in CLASSES:
        table = manifest.loc[manifest.group == group]
        used = table.loc[~table.excluded]
        row = {"group": group, "n_enrolled": len(table), "n_included": len(used),
               "n_excluded": int(table.excluded.sum()), "sex_F": int(table.Gender.eq("F").sum()),
               "sex_M": int(table.Gender.eq("M").sum()), "sex_missing": int(table.Gender.isna().sum()),
               "exclusion_reasons": "; ".join(sorted(set(table.loc[table.excluded, "exclusion_reason"]))) or "none"}
        for col, label in [("Age", "age_years"), ("MMSE", "mmse")]:
            values = table[col].dropna()
            row.update({f"{label}_mean": values.mean(), f"{label}_sd": values.std(ddof=1),
                        f"{label}_min": values.min(), f"{label}_max": values.max(),
                        f"{label}_missing": int(table[col].isna().sum())})
        for col in ["original_duration_seconds", "recording_duration_seconds", "usable_duration_seconds", "usable_epochs"]:
            values = pd.to_numeric(used[col], errors="coerce")
            row.update({f"{col}_mean": values.mean(), f"{col}_sd": values.std(ddof=1),
                        f"{col}_min": values.min(), f"{col}_max": values.max(),
                        f"{col}_total": values.sum(), f"{col}_missing": int(values.isna().sum())})
        records.append(row)
    return pd.DataFrame(records)


def run(data_dir, output_dir, config):
    root, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    if config["epoch_seconds"] <= 0 or config["welch_seconds"] <= 0 or config["welch_seconds"] > config["epoch_seconds"]:
        raise ValueError("Require 0 < Welch duration <= epoch duration")
    if config["sampling_hz"] is not None and config["sampling_hz"] < 100:
        raise ValueError("Use at least 100 Hz to retain the derivative passband")
    if config["power_floor_uv2"] <= 0 or config["flat_ptp_uv"] < 0 or config["amplitude_warning_uv"] <= 0:
        raise ValueError("Invalid numerical floor/QC parameters")
    np.random.seed(config["seed"])
    provenance_path = root / "download_manifest.json"
    if not provenance_path.exists():
        raise ValueError("Run download.py to generate verified snapshot provenance, including local file checks")
    provenance = json.loads(provenance_path.read_text())
    expected = {f["path"]: f for f in provenance["files"]}
    software = {p: version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "mne", "requests"]}
    software["python"] = platform.python_version()
    software["platform"] = platform.platform()
    code = {name: sha256(HERE / name) for name in
            ["download.py", "shared.py", "features.py", "evaluation.py", "plots.py", "run.py"]}
    specification = {**config, "dataset": provenance["dataset"], "dataset_version": provenance["version"],
                     "dataset_commit": provenance["git_commit"], "channels": CHANNELS,
                     "bands_hz": BANDS, "relative_denominator_hz": [1, 30], "signal_units": "microvolts",
                     "psd_units": "microvolts^2/Hz", "absolute_power_units": "microvolts^2",
                     "log_power": "log10(power / 1 microvolt^2)",
                     "welch_window": "hann", "welch_overlap_fraction": .5, "welch_detrend": "constant",
                     "welch_average": "mean", "welch_scaling": "density", "welch_nfft": "nperseg",
                     "integration": "trapezoid over interpolated PSD, exact endpoints; zero-measure shared boundaries",
                     "software": software, "source_code_sha256": code}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    specification["analysis_fingerprint"] = fingerprint
    if (output / "analysis_config.json").exists():
        old = json.loads((output / "analysis_config.json").read_text())
        if any(old.get(k) != v for k, v in config.items()) or old.get("dataset_commit") != provenance["git_commit"]:
            raise ValueError("Analysis parameters changed: choose a new output directory to preserve results")
    write_json(output / "analysis_config.json", specification)
    write_json(output / "software_versions.json", software)
    write_json(output / "dataset_provenance.json", provenance)
    participants, definitions = participant_metadata(root)
    write_json(output / "participant_definitions.json", definitions)
    # Metadata caches are verified too; modified diagnostic labels cannot silently enter the analysis.
    for relative, record in expected.items():
        if not relative.endswith(".set") and (root / relative).exists() and sha256(root / relative) != record["sha256"]:
            raise ValueError(f"Snapshot metadata checksum changed: {relative}")
    inventory = provenance["recording_inventory"]
    known_ids = set(participants.participant_id)
    unmatched = [p for p in inventory if (p.split("/")[1] if p.startswith("derivatives/") else p.split("/")[0]) not in known_ids]
    if unmatched:
        raise ValueError(f"Recordings without participant metadata: {unmatched}")
    manifests, segments, epoch_tables, channel_qc, histories = [], [], [], [], {}
    subject_rows, spectra = [], {}
    frequency = None
    cached_count, units_checked = 0, False
    for participant in participants.to_dict("records"):
        pid = participant["participant_id"]
        prefix = "derivatives/" if config["source"] == "derivatives" else ""
        paths = [p for p in inventory if p.startswith(f"{prefix}{pid}/") and p.endswith("_eeg.set")]
        row = {**participant, "source_type": config["source"], "recording_count": len(paths),
               "recording_path": "", "signal_units": "microvolts", "sampling_hz": np.nan,
               "analysis_sampling_hz": np.nan, "channel_names": "", "channel_count": np.nan,
               "original_duration_seconds": np.nan, "recording_duration_seconds": np.nan,
               "usable_duration_seconds": 0., "usable_epochs": 0, "excluded": True,
               "exclusion_reason": "", "metadata_missing": ";".join(k for k in ["Age", "Gender", "MMSE"] if pd.isna(participant[k]))}
        if len(paths) != 1:
            row["exclusion_reason"] = "missing_recording" if not paths else "multiple_recordings_require_explicit_aggregation"
            manifests.append(row)
            continue
        relative, path = paths[0], root / paths[0]
        row["recording_path"] = relative
        sidecar = root / pid / "eeg" / f"{pid}_task-eyesclosed_eeg.json"
        channels_path = root / pid / "eeg" / f"{pid}_task-eyesclosed_channels.tsv"
        if not all(p.exists() for p in [path, sidecar, channels_path]):
            row["exclusion_reason"] = "missing_recording_or_metadata_file"
            manifests.append(row)
            continue
        digest = sha256(path)
        if relative not in expected or digest != expected[relative]["sha256"]:
            raise ValueError(f"Unverified/changed recording: {relative}")
        try:
            eeg, x, fs, labels, info = read_recording(path, channels_path, sidecar)
            original, timeline, removed = original_sample_indices(eeg, x.shape[1], config["source"])
        except (ValueError, KeyError) as error:
            row["exclusion_reason"] = f"loader_validation: {error}"
            manifests.append(row)
            continue
        history = str(eeg.get("history", ""))
        history_hash = hashlib.sha256(history.encode()).hexdigest()
        histories[history_hash] = history
        source_flat = [ch for ch, amplitude in zip(CHANNELS, np.ptp(x, axis=1)) if amplitude <= config["flat_ptp_uv"]]
        for ch, signal in zip(CHANNELS, x):
            finite = signal[np.isfinite(signal)]
            channel_qc.append({"participant_id": pid, "group": participant["group"], "channel": ch,
                               "nonfinite_samples": int(signal.size - finite.size),
                               "min_uv": finite.min() if finite.size else np.nan,
                               "max_uv": finite.max() if finite.size else np.nan,
                               "sd_uv": finite.std(ddof=1) if finite.size > 1 else np.nan,
                               "abs_p99_uv": np.percentile(np.abs(finite), 99) if finite.size else np.nan,
                               "ptp_uv": np.ptp(finite) if finite.size else np.nan,
                               "flat": ch in source_flat})
        if not units_checked:
            import mne
            # Independent reader confirms EEGLAB µV -> MNE volts conversion.
            raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
            np.testing.assert_allclose(raw.get_data(start=0, stop=1000) * 1e6,
                                       x[[CHANNELS.index(ch) for ch in labels], :1000], rtol=1e-6, atol=1e-8)
            units_checked = True
        x = prepare(x, fs, config)
        intervals = select_intervals(x, fs, original, config)
        intervals.insert(0, "participant_id", pid)
        intervals.insert(1, "group", participant["group"])
        intervals.insert(2, "recording_path", relative)
        intervals["original_time_mapping"] = timeline
        segments.append(intervals)
        usable = intervals.loc[intervals.usable].copy()
        row.update({"sampling_hz": fs, "analysis_sampling_hz": config["sampling_hz"] or fs,
                    "channel_names": ";".join(labels), "channel_count": len(labels),
                    "recording_sha256": digest, "header_reference": str(eeg.get("ref", "")),
                    "raw_sidecar_reference": info.get("EEGReference", "unknown"),
                    "raw_sidecar_software_filters": json.dumps(info.get("SoftwareFilters", {})),
                    "raw_sidecar_duration_seconds": info.get("RecordingDuration", np.nan),
                    "history_sha256": history_hash, "filter_history_eegfiltnew": "pop_eegfiltnew" in history,
                    "asr_history_present": "clean_artifacts" in history, "ica_history_present": "pop_runica" in history,
                    "iclabel_history_present": "pop_iclabel" in history,
                    "original_duration_seconds": (len(original) + removed) / fs,
                    "recording_duration_seconds": x.shape[1] / fs,
                    "provider_removed_seconds": removed / fs,
                    "contiguous_runs": int(np.sum(np.diff(original) != 1) + 1),
                    "candidate_epochs": len(intervals), "usable_epochs": len(usable),
                    "rejected_epochs": len(intervals) - len(usable),
                    "usable_duration_seconds": len(usable) * config["epoch_seconds"],
                    "unepoched_tail_seconds": x.shape[1] / fs - len(intervals) * config["epoch_seconds"],
                    "nonfinite_samples": int((~np.isfinite(x)).sum()), "flat_channels": ";".join(source_flat),
                    "max_abs_uv": float(np.nanmax(np.abs(x))),
                    "amplitude_review_epochs": int(intervals.amplitude_review_flag.sum()),
                    "original_time_mapping": timeline, "excluded": not len(usable),
                    "exclusion_reason": "no_usable_epochs" if not len(usable) else ""})
        manifests.append(row)
        if not len(usable):
            continue
        identity = hashlib.sha256((fingerprint + digest).encode()).hexdigest()
        cached = cache / f"{pid}_{identity}.npz"
        if cached.exists():
            with np.load(cached) as data:
                vectors, subject_psd, freq = data["features"], data["mean_psd"], data["frequency"]
                floors = int(data["floored_powers"])
                if str(data["identity"]) != identity:
                    raise ValueError("Cache identity mismatch")
            cached_count += 1
        else:
            vectors, psds, floors = [], [], 0
            for interval, epoch, target in iter_epochs(x, fs, intervals, config):
                vector, freq, psd, floored = extract(epoch, target, config)
                vectors.append(vector)
                psds.append(psd)
                floors += floored
            vectors, subject_psd = np.stack(vectors), np.stack(psds).mean(axis=0)
            np.savez_compressed(cached, features=vectors, mean_psd=subject_psd, frequency=freq,
                                floored_powers=floors, identity=identity)
        if vectors.shape != (len(usable), len(feature_names())) or not np.isfinite(vectors).all():
            raise ValueError("Invalid cached epoch features")
        if frequency is not None:
            np.testing.assert_array_equal(freq, frequency)
        frequency = freq
        row["floored_band_powers"] = floors
        features = pd.DataFrame(vectors, columns=feature_names(), index=usable.index)
        epoch_tables.append(pd.concat([usable, features], axis=1))
        subject_rows.append({"participant_id": pid, "group": participant["group"],
                             **dict(zip(feature_names(), vectors.mean(axis=0)))})
        spectra[pid] = subject_psd
        print(f"{pid} {participant['group']}: {len(usable)} epochs, {row['usable_duration_seconds']:.0f} usable seconds", flush=True)
    manifest = pd.DataFrame(manifests)
    manifest.to_csv(output / "participant_manifest.csv", index=False)
    pd.DataFrame(channel_qc).to_csv(output / "channel_qc.csv", index=False)
    write_json(output / "recording_histories.json", histories)
    group_summary(manifest).to_csv(output / "group_summary.csv", index=False)
    report = {"dataset_participant_count": len(participants),
              "metadata_group_counts": participants.group.value_counts().to_dict(),
              "included_group_counts": manifest.loc[~manifest.excluded, "group"].value_counts().to_dict(),
              "exclusions": manifest.loc[manifest.excluded, ["participant_id", "exclusion_reason"]].to_dict("records"),
              "all_participants_matched_one_recording": bool(manifest.recording_count.eq(1).all()),
              "unmatched_recordings": unmatched,
              "independent_mne_microvolt_check_passed": units_checked,
              "analysis_fingerprint": fingerprint, "source_cache_hits": cached_count,
              "amplitude_warning_uv": config["amplitude_warning_uv"],
              "amplitude_warning_policy": "review only, no automatic amplitude rejection",
              "flat_policy": "exclude epochs containing any channel with peak-to-peak <= numerical flat tolerance",
              "preprocessing_discrepancy": "README calls filter Butterworth; supplied histories use pop_eegfiltnew. Raw acquisition README and sidecars disagree on reference/filter details.",
              "original_mapping": "clean_sample_mask verified against every internal EEGLAB boundary event",
              "run_date": datetime.now(ZoneInfo("America/Edmonton")).date().isoformat()}
    write_json(output / "qc_summary.json", report)
    if not epoch_tables:
        raise ValueError("No real-data usable epochs: QC audit saved, classification unavailable")
    epoch_features = pd.concat(epoch_tables, ignore_index=True)
    intervals = pd.concat(segments, ignore_index=True)
    intervals.to_csv(output / "shared_intervals.csv", index=False)
    epoch_features.to_csv(output / "epoch_features.csv.gz", index=False,
                          compression={"method": "gzip", "mtime": 0})
    subjects = pd.DataFrame(subject_rows).sort_values("participant_id").reset_index(drop=True)
    subjects.to_csv(output / "participant_features.csv", index=False)
    write_json(output / "feature_schema.json", {"feature_names": feature_names(), "dimension": len(feature_names()),
                                               "aggregation": "Arithmetic mean of epoch feature vectors per participant",
                                               "metadata_predictors": [], "channel_order": CHANNELS})
    np.savez_compressed(output / "participant_psd.npz", frequency_hz=frequency, channels=np.array(CHANNELS),
                        participant_ids=subjects.participant_id.to_numpy(dtype=str),
                        mean_psd_uv2_per_hz=np.stack([spectra[p] for p in subjects.participant_id]))
    # Assert equal-participant aggregation and identical temporal intervals used by future adapters.
    recomputed = epoch_features.groupby("participant_id")[feature_names()].mean().loc[subjects.participant_id]
    np.testing.assert_allclose(recomputed, subjects[feature_names()], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(epoch_features.original_end_seconds - epoch_features.original_start_seconds,
                               config["epoch_seconds"], atol=1e-9)
    folds = saved_folds(subjects, output, config["seed"])
    metrics, oof = evaluate(subjects, feature_names(), folds, output, config)
    summaries(subjects, spectra, frequency, output, config["seed"])
    representative_qc(root, manifest, intervals, output, config)
    confusion(oof, output)
    print(metrics.to_string(index=False), flush=True)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/ds004504")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--config", type=Path, help="JSON overrides for DEFAULTS; unknown keys rejected")
    args = parser.parse_args()
    configuration = DEFAULTS.copy()
    if args.config:
        override = json.loads(args.config.read_text())
        if set(override) - set(DEFAULTS):
            raise ValueError(f"Unknown configuration options: {set(override) - set(DEFAULTS)}")
        configuration.update(override)
    if configuration["source"] not in ["raw", "derivatives"] or configuration["reference"] not in ["preserve", "average"]:
        raise ValueError("Invalid source/reference")
    run(args.data_dir, args.output_dir, configuration)
