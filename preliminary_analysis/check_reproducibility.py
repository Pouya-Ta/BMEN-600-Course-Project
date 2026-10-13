"""Repeat saved analyses and compare scientific table/prediction/figure byte hashes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from download import sha256
from run import DEFAULTS

HERE = Path(__file__).resolve().parent


def check(data, output):
    output = Path(output).resolve()
    baseline = json.loads((output / "analysis_config.json").read_text())
    names = ["analysis_config.json", "software_versions.json", "dataset_provenance.json", "participant_manifest.csv",
             "channel_qc.csv", "group_summary.csv", "shared_intervals.csv", "epoch_features.csv.gz",
             "participant_features.csv", "participant_psd.npz", "fold_assignments.csv",
             "spectral_fold_scores.csv", "spectral_metrics.csv", "spectral_oof_predictions.csv",
             "spectral_logistic_confusion_matrix.csv", "spectral_dummy_confusion_matrix.csv",
             "group_psd_summary.csv", "relative_theta_alpha_points.csv"]
    for stem in ["group_psd", "relative_theta_alpha", "qc_representative_traces", "qc_representative_spectra", "spectral_confusion_matrix"]:
        names.extend(f"{stem}.{extension}" for extension in ["png", "pdf"])
    encoder = None
    if (output / "encoder_config.json").exists():
        encoder = json.loads((output / "encoder_config.json").read_text())
        names += ["encoder_config.json", "encoder_smoke_test.json", "encoder_epoch_embeddings.csv.gz",
                  "encoder_participant_embeddings.csv", "bendr_metrics.csv", "bendr_fold_scores.csv",
                  "bendr_oof_predictions.csv", "bendr_logistic_confusion_matrix.csv",
                  "bendr_confusion_matrix.png", "bendr_confusion_matrix.pdf",
                  "paired_fold_scores.csv", "representation_comparison.csv"]
    before = {name: sha256(output / name) for name in names}
    qc_before = json.loads((output / "qc_summary.json").read_text())
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    log_path = output / "cache" / "reproducibility_run.log"
    with TemporaryDirectory() as tmp, log_path.open("w") as log:
        config = Path(tmp) / "config.json"
        config.write_text(json.dumps({key: baseline[key] for key in DEFAULTS}))
        subprocess.run([sys.executable, str(HERE / "run.py"), "--data-dir", str(data),
                        "--output-dir", str(output), "--config", str(config)],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if encoder:
            subprocess.run([sys.executable, str(HERE / "encoder.py"), "--data-dir", str(data),
                            "--output-dir", str(output), "--device", encoder["device"],
                            "--threads", str(encoder["threads"]), "--batch-size", str(encoder["batch_size"])],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
    qc_after = json.loads((output / "qc_summary.json").read_text())
    for record in [qc_before, qc_after]:
        for key in ["run_date", "source_cache_hits"]:
            record.pop(key, None)
    after = {name: sha256(output / name) for name in names}
    identical = {name: before[name] == after[name] for name in names}
    report = {"identical_outputs": identical, "sha256": after,
              "all_checked_files_identical": all(identical.values()),
              "scientific_qc_fields_identical": qc_before == qc_after,
              "execution_metadata_excluded": ["qc_summary.run_date", "qc_summary.source_cache_hits"],
              "baseline_cache_hits_on_repeat": json.loads((output / "qc_summary.json").read_text())["source_cache_hits"],
              "encoder_cache_reused": encoder is not None,
              "scope": "Repeat pipeline and cache reuse; encoder first participant is recomputed after a fresh smoke test"}
    (output / "reproducibility_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Identical files: {sum(identical.values())}/{len(names)}; scientific QC unchanged: {qc_before == qc_after}")
    if not report["all_checked_files_identical"] or not report["scientific_qc_fields_identical"]:
        raise AssertionError(f"Nonreproducible outputs: {[name for name, same in identical.items() if not same]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data/ds004504")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    check(args.data_dir, args.output_dir)
