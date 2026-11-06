"""Optional frozen BENDR convolutional representations using the official release code.

Only run after run.py succeeds. The historical DN3 training framework is not needed:
AST selects unmodified upstream model/transform definitions with their dependencies.
"""
import argparse
import ast
from collections import OrderedDict
from math import ceil
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

from download import sha256
from evaluation import evaluate
from plots import confusion
from shared import CHANNELS, iter_epochs, prepare, read_recording

HERE = Path(__file__).resolve().parent
CHECKPOINT_URL = "https://github.com/SPOClab-ca/BENDR/releases/download/v0.1-alpha/encoder.pt"


def official_sources(model_dir):
    """Pin release/tag commits and hash both checkpoint and downloaded implementation."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "provenance.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for item in manifest["files"]:
            if sha256(model_dir / item["local_name"]) != item["sha256"]:
                raise ValueError("Encoder source/checkpoint cache checksum mismatch")
        return manifest
    files, commits = [], {}
    for repo, tag in [("SPOClab-ca/BENDR", "v0.1-alpha"), ("SPOClab-ca/dn3", "v0.2-alpha")]:
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/{tag}", timeout=60)
        response.raise_for_status()
        commits[repo] = response.json()["sha"]
    sources = [("SPOClab-ca/BENDR", "dn3_ext.py", "bendr_dn3_ext.py"),
               ("SPOClab-ca/BENDR", "configs/pretraining.yml", "pretraining.yml"),
               ("SPOClab-ca/BENDR", "configs/downstream.yml", "downstream.yml"),
               ("SPOClab-ca/BENDR", "utils.py", "bendr_utils.py"),
               ("SPOClab-ca/dn3", "dn3/transforms/channels.py", "channels.py"),
               ("SPOClab-ca/dn3", "dn3/transforms/instance.py", "instance.py"),
               ("SPOClab-ca/dn3", "dn3/utils.py", "dn3_utils.py")]
    for repo, relative, local in sources:
        url = f"https://raw.githubusercontent.com/{repo}/{commits[repo]}/{relative}"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        path = model_dir / local
        path.write_bytes(response.content)
        files.append({"local_name": local, "url": url, "sha256": sha256(path)})
    response = requests.get("https://api.github.com/repos/SPOClab-ca/BENDR/releases/tags/v0.1-alpha", timeout=60)
    response.raise_for_status()
    release = response.json()
    asset = next(a for a in release["assets"] if a["name"] == "encoder.pt")
    path = model_dir / "encoder.pt"
    # Fetch the official release even when an exploratory copy already exists.
    response = requests.get(CHECKPOINT_URL, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    if path.stat().st_size != asset["size"]:
        raise ValueError("Checkpoint size does not match release metadata")
    files.append({"local_name": "encoder.pt", "url": CHECKPOINT_URL, "sha256": sha256(path),
                  "github_asset_id": asset["id"], "bytes": asset["size"],
                  "published_digest": asset.get("digest")})
    manifest = {"candidate": "BENDR", "checkpoint_release": "v0.1-alpha",
                "release_date": release["published_at"], "git_commits": commits, "files": files,
                "pretraining": "TUEG v1.1/v1.2; official pretraining config also uses BCI IV 2a for validation",
                "overlap_assessment": "No known ds004504 overlap in documented pretraining/evaluation; release is from 2021 and ds004504 was released in 2023. No exhaustive participant identity audit is available.",
                "dataset_finetuned_checkpoint": False,
                "digest_note": "Locally computed SHA256 records asset identity; original release may lack an independently published digest."}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def definitions(path, names, namespace):
    """Use selected official definitions verbatim, without importing legacy trainers."""
    parsed = ast.parse(Path(path).read_text())
    nodes = [node for node in parsed.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Official source definitions changed: {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load_official(model_dir):
    import torch
    from torch import nn
    from torch.nn.functional import interpolate
    from mne.utils._bunch import NamedInt
    from mne._fiff.constants import FIFF
    namespace = {"torch": torch, "nn": nn, "np": np, "ceil": ceil,
                 "OrderedDict": OrderedDict, "NamedInt": NamedInt, "FIFF": FIFF, "interpolate": interpolate}
    # Channel constants/functions are unchanged; only obsolete import statements are bypassed.
    parsed = ast.parse((model_dir / "channels.py").read_text())
    body = [n for n in parsed.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(model_dir / "channels.py"), "exec"), namespace)
    definitions(model_dir / "dn3_utils.py", ["min_max_normalize"], namespace)
    definitions(model_dir / "instance.py", ["InstanceTransform", "TemporalInterpolation", "MappingDeep1010", "To1020"], namespace)
    definitions(model_dir / "bendr_dn3_ext.py", ["_BENDREncoder", "ConvEncoderBENDR"], namespace)
    model = namespace["ConvEncoderBENDR"](20, encoder_h=512, dropout=0., projection_head=False)
    model.load_state_dict(torch.load(model_dir / "encoder.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    renamed = ["T7" if ch == "T3" else "T8" if ch == "T4" else ch for ch in CHANNELS]
    dataset = SimpleNamespace(channels=np.array([(name, FIFF.FIFFV_EEG_CH) for name in renamed], dtype=object),
                              info=SimpleNamespace(data_max=None, data_min=None))
    mapping = namespace["MappingDeep1010"](dataset)
    to1020 = namespace["To1020"]()
    output_names = to1020.EEG_20_div + ["SCALE"]
    # Every EEG output has exactly one supplied sensor, with T3/T4 equivalent to T7/T8.
    assert mapping.mapping[:, to1020._inds_20_div[:-1]].sum(axis=0).eq(1).all()

    def adapt(epoch_uv):
        # DN3's MNE loader supplies volts. Mapping then normalizes each EEG trial globally.
        signal = torch.tensor(epoch_uv * 1e-6, dtype=torch.float32)
        return mapping(signal), to1020

    return model, adapt, namespace["TemporalInterpolation"], output_names


def run(data_dir, output_dir, model_dir, device="cpu", batch_size=16, threads=4):
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(600)
    np.random.seed(600)
    torch.use_deterministic_algorithms(True)
    output, root, model_dir = Path(output_dir), Path(data_dir), Path(model_dir)
    required = ["spectral_metrics.csv", "participant_features.csv", "shared_intervals.csv", "fold_assignments.csv", "analysis_config.json"]
    if not all((output / name).exists() for name in required):
        raise ValueError("Complete and save steps 1–3 before encoder extraction")
    baseline = json.loads((output / "analysis_config.json").read_text())
    provenance = official_sources(model_dir)
    model, adapt, interpolation, channel_order = load_official(model_dir)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable; CPU extraction is supported")
    model.to(device)
    specification = {"checkpoint_provenance": provenance, "torch_version": torch.__version__,
                     "device": device, "threads": threads, "batch_size": batch_size,
                     "shared_analysis_fingerprint": baseline["analysis_fingerprint"],
                     "shared_intervals_sha256": sha256(output / "shared_intervals.csv"),
                     "adapter_source_sha256": sha256(__file__), "input_sample_rate_hz": 256,
                     "window_seconds": baseline["epoch_seconds"], "input_channel_order": channel_order,
                     "source_units": "microvolts", "dn3_input_units": "volts",
                     "normalization": "Official DN3 MappingDeep1010 per-window global EEG minmax to [-1,1]",
                     "scale_channel": "0: official missing dataset amplitude-bounds behavior; no cohort-derived normalization",
                     "resampling": "Official TemporalInterpolation nearest (after mapping); data already low-passed to 45 Hz",
                     "aliases": {"T3": "T7", "T4": "T8"},
                     "representation": "ConvEncoderBENDR final convolution block, 512 channels, mean across time",
                     "participant_pooling": "Arithmetic mean of shared 8-second epoch embeddings",
                     "subdivision": "none; CNN accepts the full variable-length shared epoch",
                     "frozen": True, "gradients": False, "eval_mode": True, "fine_tuning": False}
    identity = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    specification["embedding_fingerprint"] = identity
    cache = output / "cache" / "bendr"
    cache.mkdir(parents=True, exist_ok=True)
    intervals = pd.read_csv(output / "shared_intervals.csv")
    intervals = intervals.loc[intervals.usable]
    manifest = pd.read_csv(output / "participant_manifest.csv").set_index("participant_id")
    rows, epoch_rows, smoke = [], [], None
    started = time.monotonic()
    for pid, segments in intervals.groupby("participant_id", sort=True):
        recorded = manifest.loc[pid]
        path = root / recorded.recording_path
        if sha256(path) != recorded.recording_sha256:
            raise ValueError("Shared source recording changed after spectral extraction")
        cached = cache / f"{pid}_{identity}.npz"
        if cached.exists() and smoke is not None:
            with np.load(cached) as stored:
                embeddings = stored["embeddings"]
                assert str(stored["identity"]) == identity
        else:
            _, x, fs, _, _ = read_recording(path, root / pid / "eeg" / f"{pid}_task-eyesclosed_channels.tsv",
                                          root / pid / "eeg" / f"{pid}_task-eyesclosed_eeg.json")
            x = prepare(x, fs, baseline)
            transform = interpolation(int(round(baseline["epoch_seconds"] * 256)), mode="nearest", new_sfreq=256)
            adapted = []
            for segment, epoch, shared_fs in iter_epochs(x, fs, segments, baseline):
                deep, to1020 = adapt(epoch)
                adapted.append(to1020(transform(deep))[0])
            inputs = torch.stack(adapted)
            assert inputs.shape == (len(segments), 20, int(round(baseline["epoch_seconds"] * 256)))
            assert torch.isfinite(inputs).all() and not model.training
            assert all(not p.requires_grad for p in model.parameters())
            if smoke is None:
                with torch.inference_mode():
                    encoded = model(inputs[:3].to(device))
                    repeated = model(inputs[:3].to(device))
                assert encoded.ndim == 3 and encoded.shape[1] == 512 and torch.isfinite(encoded).all()
                assert not encoded.requires_grad and not torch.allclose(encoded[0], encoded[1])
                torch.testing.assert_close(encoded, repeated, rtol=0, atol=0)
                smoke = {"participant_id": pid, "input_shape": list(inputs[:3].shape),
                         "encoded_shape": list(encoded.shape), "finite": True,
                         "different_inputs_produce_different_outputs": True,
                         "repeat_inference_identical": True, "frozen_eval_no_grad": True,
                         "representations_not_logits": True, "checkpoint_loaded_strictly": True}
                (output / "encoder_smoke_test.json").write_text(json.dumps(smoke, indent=2) + "\n")
                print(f"Encoder smoke test passed: {smoke}", flush=True)
            with torch.inference_mode():
                embeddings = torch.cat([model(batch.to(device)).mean(dim=-1).cpu()
                                        for batch in inputs.split(batch_size)]).numpy()
            np.savez_compressed(cached, embeddings=embeddings, identity=identity)
        if embeddings.shape != (len(segments), 512) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid cached representations")
        columns = [f"bendr_{i:03d}" for i in range(512)]
        frame = pd.DataFrame(embeddings, columns=columns, index=segments.index)
        epoch_rows.append(pd.concat([segments, frame], axis=1))
        rows.append({"participant_id": pid, "group": recorded.group, **dict(zip(columns, embeddings.mean(axis=0)))})
        print(f"BENDR {pid}: {len(segments)} matching shared intervals ({time.monotonic()-started:.1f}s elapsed)", flush=True)
    all_epochs = pd.concat(epoch_rows, ignore_index=True)
    all_epochs.to_csv(output / "encoder_epoch_embeddings.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    subjects = pd.DataFrame(rows)
    subjects.to_csv(output / "encoder_participant_embeddings.csv", index=False)
    folds = pd.read_csv(output / "fold_assignments.csv")
    (output / "encoder_config.json").write_text(json.dumps(specification, indent=2) + "\n")
    metrics, oof = evaluate(subjects, columns, folds, output, baseline, representation="bendr", include_dummy=False)
    confusion(oof, output, representation="bendr")
    comparison = pd.concat([pd.read_csv(output / "spectral_metrics.csv"), metrics], ignore_index=True)
    comparison.to_csv(output / "representation_comparison.csv", index=False)
    paired = pd.read_csv(output / "spectral_fold_scores.csv")
    paired = paired.loc[paired.classifier == "logistic"].merge(pd.read_csv(output / "bendr_fold_scores.csv"), on="fold", suffixes=("_spectral", "_bendr"), validate="one_to_one")
    paired["balanced_accuracy_difference_bendr_minus_spectral"] = paired.balanced_accuracy_bendr - paired.balanced_accuracy_spectral
    paired.to_csv(output / "paired_fold_scores.csv", index=False)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/ds004504")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--model-dir", type=Path, default=HERE / "models/bendr")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.model_dir, args.device, args.batch_size, args.threads)
