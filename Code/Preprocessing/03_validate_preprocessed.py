"""Validate every saved array, provenance, timing and original-source correspondence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from eeg_common import (CHANNELS, DEFAULT_DATA, DEFAULT_OUTPUT, original_indices,
                        read_eeg, recording_path, sha256, write_json)


def validate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    processing = json.loads((output / "processing.json").read_text())
    if processing["status"] != "complete":
        raise ValueError("Preprocessing has not completed")
    config = processing["config"]
    sfreq = config["target_sfreq_hz"]
    manifest = pd.read_csv(output / "participant_manifest.csv")
    inventory = pd.read_csv(output / "qc" / "dataset_inventory.csv").set_index("participant_id")
    epoch_qc = pd.read_csv(output / "qc" / "epoch_qc.csv", dtype={"rejection_reason": str}, keep_default_na=False)
    saved_people = pd.read_csv(output / "participants.tsv", sep="\t")
    if not manifest.participant_id.is_unique or set(manifest.participant_id) != set(saved_people.participant_id):
        raise ValueError("Missing, additional or duplicate participant")
    if not epoch_qc.epoch_id.is_unique:
        raise ValueError("Duplicate epoch IDs")
    checksum_count = 0
    for list_path, base in [(output / "qc" / "input_checksums.csv", root), (output / "output_checksums.csv", output)]:
        for row in pd.read_csv(list_path).itertuples():
            path = base / row.path
            if path.stat().st_size != row.size_bytes or sha256(path) != row.sha256:
                raise ValueError(f"Checksum/size mismatch: {path}")
            checksum_count += 1
    checked_epochs, source_spot_checks = 0, 0
    for row in manifest.itertuples():
        pid = row.participant_id
        data = np.load(output / row.array_path, mmap_mode="r", allow_pickle=False)
        tables = pd.read_csv(output / row.epoch_table_path, sep="\t", dtype={"rejection_reason": str}, keep_default_na=False)
        retained = tables.loc[tables.retained].reset_index(drop=True)
        expected = (row.retained_epochs, 19, round(config["epoch_seconds"] * sfreq))
        if data.shape != expected or data.dtype != np.float32:
            raise ValueError(f"Wrong dimensions/dtype: {pid}")
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite saved EEG: {pid}")
        if not np.array_equal(retained.array_index, np.arange(len(data))):
            raise ValueError(f"Array indices do not align with metadata: {pid}")
        if not tables.participant_id.eq(pid).all() or not tables.diagnosis.eq(row.diagnosis).all():
            raise ValueError(f"Epoch identity or diagnosis mismatch: {pid}")
        if not tables.loc[~tables.retained].array_index.eq(-1).all():
            raise ValueError("Rejected epoch has an output index")
        global_table = epoch_qc.loc[epoch_qc.participant_id.eq(pid)].reset_index(drop=True)
        pd.testing.assert_frame_equal(tables, global_table)
        if len(tables) != row.candidate_epochs or tables.retained.sum() != row.retained_epochs:
            raise ValueError("Counts disagree")
        guard = round(config["boundary_guard_seconds"] * 500)
        for table in [tables, retained]:
            np.testing.assert_array_equal(table.source_end_sample - table.source_start_sample,
                                          np.full(len(table), round(config["epoch_seconds"] * 500)))
            np.testing.assert_array_equal(table.original_end_sample - table.original_start_sample,
                                          np.full(len(table), round(config["epoch_seconds"] * 500)))
            if not ((table.source_start_sample >= table.run_source_start_sample + guard) &
                    (table.source_end_sample <= table.run_source_end_sample - guard)).all():
                raise ValueError("Epoch enters boundary guard")
        for _, group in tables.groupby("run_id"):
            if (group.source_start_sample.to_numpy()[1:] < group.source_end_sample.to_numpy()[:-1]).any():
                raise ValueError("Overlapping epochs")
        if len(data):
            ptp = np.ptp(data, axis=-1) * 1e6
            if ptp.max() > config["reject_peak_to_peak_uv"] + 0.01 or ptp.min() <= config["reject_flat_peak_to_peak_uv"] - 1e-5:
                raise ValueError(f"Saved epoch violates artifact rules: {pid}")
        eeg, source, fs, _ = read_eeg(recording_path(root, pid), root, pid)
        original, edges = original_indices(eeg, source.shape[1], int(inventory.loc[pid, "raw_samples"]))
        # Check original mapping against the actual mask for every candidate window.
        for event in tables.itertuples():
            if original[event.source_start_sample] != event.original_start_sample or original[event.source_end_sample - 1] + 1 != event.original_end_sample:
                raise ValueError("Original timestamp mismatch")
            if not np.all(np.diff(original[event.source_start_sample:event.source_end_sample]) == 1):
                raise ValueError("Epoch bridges a source discontinuity")
        if config["reference"] == "average":
            source -= source.mean(axis=0, keepdims=True)
        # Independently recompute beginning/middle/end saved windows from source.
        for index in sorted(set([0, len(data) // 2, len(data) - 1])) if len(data) else []:
            event = retained.iloc[index]
            left, right = int(event.run_source_start_sample), int(event.run_source_end_sample)
            from fractions import Fraction
            ratio = Fraction(sfreq / fs).limit_denominator(10000)
            native = source[:, left:right]
            rebuilt = resample_poly(native, ratio.numerator, ratio.denominator, axis=-1,
                                    window=("kaiser", 5.0), padtype="line") if sfreq != fs else native
            start = round((event.source_start_sample - left) / fs * sfreq)
            rebuilt = rebuilt[:, start:start + expected[-1]] * 1e-6
            np.testing.assert_allclose(data[index], rebuilt, rtol=1e-6, atol=1e-10)
            source_spot_checks += 1
        checked_epochs += len(data)
    summary = json.loads((output / "qc" / "preprocessing_summary.json").read_text())
    if checked_epochs != summary["retained_epochs"] or len(manifest) != summary["participants_processed"]:
        raise ValueError("Summary does not match validated arrays")
    result = {"status": "passed", "participants_checked": len(manifest), "saved_epochs_checked": checked_epochs,
              "files_checksum_checked": checksum_count, "independent_source_epoch_spot_checks": source_spot_checks,
              "all_saved_values_finite": True, "all_epochs_nonoverlapping": True,
              "all_epochs_respect_original_cuts_and_guards": True, "all_arrays_match_subject_metadata": True,
              "units_and_resampling_match_native_microvolt_source": True,
              "all_original_input_checksums_unchanged": True,
              "fingerprint": processing["fingerprint"], "validation_code_sha256": sha256(Path(__file__))}
    write_json(output / "qc" / "validation_results.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    validate(args.data_dir, args.output_dir)
