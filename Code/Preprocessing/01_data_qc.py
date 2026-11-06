"""Audit all 176 EEG files and verify ds004504 v1.0.9 without changing inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_common import (CHANNELS, DEFAULT_DATA, DEFAULT_OUTPUT, HERE, original_indices,
                        participants, read_eeg, recording_path, sha256,
                        snapshot_checksums, verify_provider_history, write_json)


def audit(root, output, archive):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    people = participants(root)
    if len(people) != 88 or people.diagnosis.value_counts().to_dict() != {"AD": 36, "HC": 29, "FTD": 23}:
        raise ValueError("The local cohort does not match the selected dataset")
    expected = snapshot_checksums(archive)
    files, records, channel_rows, history_text = [], [], [], {}
    required = [root / name for name in ["README", "CHANGES", "dataset_description.json", "participants.tsv", "participants.json"]]
    for pid in people.participant_id:
        required += sorted((root / pid / "eeg").glob("*.json"))
        required += sorted((root / pid / "eeg").glob("*.tsv"))
        required += [recording_path(root, pid, kind) for kind in ["raw", "derivatives"]]
    if len(list(root.glob("sub-*/eeg/*.set"))) != 88 or len(list(root.glob("derivatives/sub-*/eeg/*.set"))) != 88:
        raise ValueError("Unexpected additional or missing EEG recordings")
    for path in required:
        relative = str(path.relative_to(root))
        entry = expected.get(relative)
        digest = sha256(path)
        if entry is None or entry["sha256"] != digest or entry["size_bytes"] != path.stat().st_size:
            raise ValueError(f"Published v1.0.9 checksum/size mismatch: {relative}")
        files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": digest,
                      "matches_published_1_0_9": True})
    print(f"Verified {len(files)} files against published version 1.0.9", flush=True)
    for person in people.to_dict("records"):
        pid = person["participant_id"]
        raw_path = recording_path(root, pid, "raw")
        derivative_path = recording_path(root, pid)
        raw_eeg, raw, fs, info = read_eeg(raw_path, root, pid)
        eeg, data, fs2, _ = read_eeg(derivative_path, root, pid)
        if not np.isfinite(raw).all() or not np.isfinite(data).all():
            raise ValueError(f"Nonfinite source EEG: {pid}; requires manual investigation")
        original, edges = original_indices(eeg, data.shape[1], raw.shape[1])
        history = verify_provider_history(eeg)
        history_id = hashlib.sha256(history.encode()).hexdigest()
        history_text[history_id] = history
        # An independent MNE reader verifies native microvolt -> SI volt conversion.
        if pid in ["sub-001", "sub-037", "sub-066"]:
            import mne
            loaded = mne.io.read_raw_eeglab(derivative_path, preload=False, verbose="ERROR")
            order = [loaded.ch_names.index(ch) for ch in CHANNELS]
            np.testing.assert_allclose(loaded.get_data(start=0, stop=2000)[order] * 1e6,
                                       data[:, :2000], rtol=1e-6, atol=1e-8)
        n_baseline = sum((right - left) // int(8 * fs) for left, right in zip(edges[:-1], edges[1:]))
        ptps = []
        for left, right in zip(edges[:-1], edges[1:]):
            for start in range(left, right - 4000 + 1, 4000):
                ptps.append(np.ptp(data[:, start:start + 4000], axis=1).max())
        for kind, signal in [("raw", raw), ("derivatives", data)]:
            for ch, x in zip(CHANNELS, signal):
                channel_rows.append({"participant_id": pid, "diagnosis": person["diagnosis"],
                                     "source": kind, "channel": ch, "sd_uv": float(x.std()),
                                     "min_uv": float(x.min()), "max_uv": float(x.max()),
                                     "peak_to_peak_uv": float(np.ptp(x)),
                                     "abs_p99_uv": float(np.percentile(np.abs(x), 99)),
                                     "nonfinite_samples": 0})
        records.append({**person, "raw_samples": raw.shape[1], "derivative_samples": data.shape[1],
                        "source_sfreq_hz": fs, "channel_count": len(CHANNELS),
                        "raw_seconds": raw.shape[1] / fs, "derivative_seconds": data.shape[1] / fs,
                        "provider_removed_seconds": (raw.shape[1] - data.shape[1]) / fs,
                        "sidecar_seconds": info["RecordingDuration"],
                        "sidecar_duration_matches_raw": bool(np.isclose(raw.shape[1] / fs, info["RecordingDuration"])),
                        "contiguous_runs": len(edges) - 1, "baseline_8s_epochs": int(n_baseline),
                        "baseline_epochs_ptp_over_200uv": int((np.array(ptps) > 200).sum()),
                        "header_reference": str(eeg.get("ref", "")).strip(),
                        "sidecar_reference": info["EEGReference"], "history_sha256": history_id,
                        "remaining_ica_components": int(np.asarray(eeg["icaweights"]).shape[0]),
                        "derivative_path": str(derivative_path.relative_to(root)),
                        "raw_path": str(raw_path.relative_to(root))})
        print(f"{pid}: raw {raw.shape[1]/fs:.1f}s, derivative {data.shape[1]/fs:.1f}s, {len(edges)-1} uninterrupted runs", flush=True)
    manifest = pd.DataFrame(records)
    manifest.to_csv(output / "dataset_inventory.csv", index=False)
    pd.DataFrame(files).to_csv(output / "input_checksums.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(output / "source_channel_qc.csv", index=False)
    write_json(output / "provider_histories.json", history_text)
    summary = {"dataset_id": "ds004504", "version": "1.0.9", "participants": len(people),
               "group_counts": people.diagnosis.value_counts().to_dict(),
               "recordings": 176, "verified_files": len(files), "all_checksums_match": True,
               "raw_hours": float(manifest.raw_seconds.sum() / 3600),
               "derivative_hours": float(manifest.derivative_seconds.sum() / 3600),
               "provider_removed_seconds": float(manifest.provider_removed_seconds.sum()),
               "baseline_8s_epochs": int(manifest.baseline_8s_epochs.sum()),
               "internal_boundaries": int((manifest.contiguous_runs - 1).sum()),
               "all_source_values_finite": True,
               "independent_mne_unit_check_participants": ["sub-001", "sub-037", "sub-066"],
               "archive_sha256": sha256(archive),
               "snapshot_archive_url": "https://codeload.github.com/OpenNeuroDatasets/ds004504/tar.gz/refs/tags/1.0.9",
               "metadata_discrepancies": [
                   "README calls the derivative filter Butterworth; every embedded history is checked for pop_eegfiltnew FIR.",
                   "README acquisition describes Cz and 70 Hz; raw sidecars specify A1 A2 and 0.4-50 Hz; derivative headers say A1 A2.",
                   "FTD MMSE SD is computed from participants.tsv; README's reported SD must not be copied unchecked."]}
    write_json(output / "dataset_audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "qc")
    parser.add_argument("--snapshot-archive", type=Path, default=HERE / "reference" / "ds004504-1.0.9.tar.gz")
    args = parser.parse_args()
    audit(args.data_dir, args.output_dir, args.snapshot_archive)
