"""Validated native EEGLAB loading and sample mapping; physical units are microvolts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "Dataset Completed"
DEFAULT_OUTPUT = HERE.parent / "Outputs" / "preprocessed data"
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def participants(root):
    root = Path(root)
    frame = pd.read_csv(root / "participants.tsv", sep="\t")
    definitions = json.loads((root / "participants.json").read_text())
    diagnoses = {}
    for code, text in definitions["Group"]["Levels"].items():
        text = text.lower()
        if "alzheimer" in text:
            diagnoses[code] = "AD"
        elif "frontotemporal" in text:
            diagnoses[code] = "FTD"
        elif "healthy" in text:
            diagnoses[code] = "HC"
        else:
            raise ValueError(f"Undocumented diagnosis {code}")
    if frame.participant_id.isna().any() or not frame.participant_id.is_unique:
        raise ValueError("Missing or duplicate participant IDs")
    if not frame.participant_id.str.fullmatch(r"sub-\d{3}").all():
        raise ValueError("Unexpected participant ID")
    frame["diagnosis"] = frame.Group.map(diagnoses)
    if frame.diagnosis.isna().any():
        raise ValueError("Missing diagnostic labels")
    for col in ["Age", "MMSE"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[col].isna().any():
            raise ValueError(f"Missing {col}")
    if not frame.Gender.isin(definitions["Gender"]["Levels"]).all():
        raise ValueError("Undocumented Gender values")
    return frame.sort_values("participant_id").reset_index(drop=True)


def snapshot_checksums(archive):
    """Read git-annex SHA256E keys without extracting archive files or symlinks."""
    result = {}
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts or member.isdir():
                continue
            name = str(Path(*parts))
            if member.issym():
                match = re.search(r"SHA256E-s(\d+)--([0-9a-f]{64})(?:\.|/|$)", member.linkname)
                if not match:
                    raise ValueError(f"Unrecognized annex key: {name}")
                result[name] = {"sha256": match[2], "size_bytes": int(match[1])}
            elif member.isfile():
                data = source.extractfile(member).read()
                result[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    return result


def recording_path(root, pid, source="derivatives"):
    directory = Path(root) / ("derivatives" if source == "derivatives" else "") / pid / "eeg"
    paths = sorted(directory.glob("*_task-eyesclosed_eeg.set"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {source} recording for {pid}; got {len(paths)}")
    return paths[0]


def read_eeg(path, root, pid):
    """Load self-contained MATLAB-format .set. Fail on a missing external .fdt."""
    value = loadmat(path, simplify_cells=True)
    eeg = value.get("EEG", value)
    data = np.asarray(eeg["data"])
    if data.dtype.kind not in "fi" or data.ndim != 2:
        raise ValueError(f"Expected embedded numeric continuous data: {path}")
    if data.shape != (int(eeg["nbchan"]), int(eeg["pnts"])) or int(eeg["trials"]) != 1:
        raise ValueError(f"Invalid continuous dimensions: {path}")
    labels = [str(ch["labels"]) for ch in eeg["chanlocs"]]
    table = pd.read_csv(Path(root) / pid / "eeg" / f"{pid}_task-eyesclosed_channels.tsv", sep="\t")
    info = json.loads((Path(root) / pid / "eeg" / f"{pid}_task-eyesclosed_eeg.json").read_text())
    if labels != table.name.tolist() or len(set(labels)) != 19 or set(labels) != set(CHANNELS):
        raise ValueError(f"Channel metadata mismatch: {pid}")
    if not table.type.eq("EEG").all() or not table.units.isin(["microV", "uV", "µV"]).all():
        raise ValueError(f"Expected documented EEG microvolt units: {pid}")
    fs = float(eeg["srate"])
    if fs != float(info["SamplingFrequency"]) or int(info["EEGChannelCount"]) != 19:
        raise ValueError(f"Sampling metadata mismatch: {pid}")
    if fs != 500:
        raise ValueError(f"Unexpected source sample rate: {fs}")
    data = np.asarray(data[[labels.index(ch) for ch in CHANNELS]], dtype=np.float64)
    return eeg, data, fs, info


def original_indices(eeg, n_samples, raw_n_samples):
    """EEGLAB latency is one-based; a cut after sample k has latency k+0.5."""
    etc = eeg.get("etc", {})
    mask = np.asarray(etc.get("clean_sample_mask", [])).ravel()
    if mask.size != raw_n_samples or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid or missing original clean_sample_mask")
    index = np.flatnonzero(mask)
    if index.size != n_samples:
        raise ValueError("Clean sample mask and derivative length disagree")
    breaks = (np.flatnonzero(np.diff(index) != 1) + 1).tolist()
    events = eeg.get("event", [])
    if isinstance(events, dict):
        events = [events]
    boundaries = sorted(int(np.floor(float(e["latency"]))) for e in events
                        if str(e.get("type", "")).lower() == "boundary")
    if breaks != [b for b in boundaries if 0 < b < n_samples]:
        raise ValueError("Internal boundary events disagree with original sample mask")
    return index, np.r_[0, breaks, n_samples].astype(int)


def verify_provider_history(eeg):
    history = str(eeg.get("history", ""))
    required = ["pop_eegfiltnew", "'locutoff',0.5", "'hicutoff',45", "clean_artifacts",
                "'BurstCriterion',17", "pop_runica", "pop_iclabel", "pop_icflag", "pop_subcomp"]
    if not all(token in history for token in required):
        raise ValueError("Derivative does not match the expected provider preprocessing history")
    if str(eeg.get("ref", "")).strip() != "A1 A2":
        raise ValueError("Unexpected provider reference")
    return history
