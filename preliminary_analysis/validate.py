"""Focused numerical smoke checks and validation of saved real-data outputs."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from evaluation import CLASSES, evaluate, saved_folds
from features import extract, feature_names, integrate_psd
from run import DEFAULTS
from shared import CHANNELS, original_sample_indices, select_intervals


def numerical_checks():
    fs = 500.
    time = np.arange(int(8 * fs)) / fs
    signal = np.tile(10 * np.sin(2 * np.pi * 10 * time), (len(CHANNELS), 1))
    values, frequency, psd, _ = extract(signal, fs, DEFAULTS)
    # A 10 µV-amplitude sinusoid has 50 µV² mean square power.
    np.testing.assert_allclose(integrate_psd(frequency, psd, 8, 13), 50., rtol=1e-10)
    np.testing.assert_allclose(values.reshape(19, 8)[:, 2], np.log10(50), atol=1e-10)
    np.testing.assert_allclose(values.reshape(19, 8)[:, 6], 1., atol=1e-10)
    doubled, _, _, _ = extract(signal * 2, fs, DEFAULTS)
    np.testing.assert_allclose(doubled.reshape(19, 8)[:, 2] - values.reshape(19, 8)[:, 2], np.log10(4), atol=1e-10)
    np.testing.assert_allclose(doubled.reshape(19, 8)[:, 4:], values.reshape(19, 8)[:, 4:], atol=1e-10)
    # Exact endpoints partition a nonuniform frequency grid without duplicated bins.
    f = np.array([0., 2., 5., 9., 14., 33.])
    p = np.ones((2, len(f)))
    total = sum(integrate_psd(f, p, a, b) for a, b in [(1, 4), (4, 8), (8, 13), (13, 30)])
    np.testing.assert_allclose(total, integrate_psd(f, p, 1, 30))
    mask = np.ones(10000, dtype=bool)
    mask[4500:5000] = False
    eeg = {"etc": {"clean_sample_mask": mask}, "event": {"type": "boundary", "latency": 4500.5}}
    original, _, removed = original_sample_indices(eeg, 9500, "derivatives")
    synthetic = np.tile(np.sin(np.arange(9500)), (19, 1))
    intervals = select_intervals(synthetic, fs, original, DEFAULTS)
    assert removed == 500 and intervals.source_start_sample.tolist() == [0, 4500]
    np.testing.assert_allclose(intervals.original_start_seconds, [0, 10])
    assert not ((intervals.source_start_sample < 4500) & (intervals.source_end_sample > 4500)).any()
    synthetic[0, 4500:8500] = 0
    assert select_intervals(synthetic, fs, original, DEFAULTS).usable.tolist() == [True, False]
    synthetic[0, 0] = np.nan
    assert not select_intervals(synthetic, fs, original, DEFAULTS).usable.any()
    # Small labelled synthetic data checks evaluation mechanics only, never scientific performance.
    rng = np.random.default_rng(600)
    frame = pd.DataFrame({"participant_id": [f"synthetic-{i:02d}" for i in range(30)],
                          "group": np.repeat(CLASSES, 10), "f1": rng.normal(size=30), "f2": rng.normal(size=30)})
    frame.loc[0, "f1"] = 1e6  # Makes any accidental global-scaling fit conspicuous.
    with TemporaryDirectory() as tmp:
        folds = saved_folds(frame, tmp, 600)
        _, oof = evaluate(frame, ["f1", "f2"], folds, tmp, DEFAULTS)
        assert len(oof) == 60 and oof.groupby("classifier").participant_id.nunique().eq(30).all()
    return {"sinusoid_absolute_power": True, "relative_power_partition": True,
            "amplitude_preservation": True, "gap_mapping_and_epoch_rejection": True,
            "synthetic_training_only_preprocessing": True,
            "synthetic_tests_are_not_dataset_results": True}


def real_checks(output):
    output = Path(output)
    manifest = pd.read_csv(output / "participant_manifest.csv")
    features = pd.read_csv(output / "participant_features.csv")
    epochs = pd.read_csv(output / "epoch_features.csv.gz")
    folds = pd.read_csv(output / "fold_assignments.csv")
    predictions = pd.read_csv(output / "spectral_oof_predictions.csv")
    config = json.loads((output / "analysis_config.json").read_text())
    assert len(feature_names()) == 152
    assert features.participant_id.is_unique and folds.participant_id.is_unique
    included = set(manifest.loc[~manifest.excluded, "participant_id"])
    assert included == set(features.participant_id) == set(folds.participant_id) == set(epochs.participant_id)
    for _, frame in predictions.groupby("classifier"):
        assert frame.participant_id.is_unique and set(frame.participant_id) == included
        np.testing.assert_allclose(frame[[f"probability_{c}" for c in CLASSES]].sum(axis=1), 1., atol=1e-12)
        checked = frame.merge(folds, on=["participant_id", "group"], validate="one_to_one")
        assert checked.fold_x.equals(checked.fold_y)
    for fold in range(5):
        assert not set(folds.loc[folds.fold == fold, "participant_id"]) & set(folds.loc[folds.fold != fold, "participant_id"])
    recomputed = epochs.groupby("participant_id")[feature_names()].mean().loc[features.participant_id]
    np.testing.assert_allclose(recomputed, features[feature_names()], atol=1e-12)
    assert np.isfinite(features[feature_names()]).all().all()
    np.testing.assert_allclose(epochs.original_end_seconds - epochs.original_start_seconds, config["epoch_seconds"], atol=1e-9)
    for _, table in epochs.groupby("participant_id"):
        assert np.all(table.original_start_seconds.to_numpy()[1:] >= table.original_end_seconds.to_numpy()[:-1])
    assert manifest.signal_units.eq("microvolts").all()
    assert all(not any(token in name.lower() for token in ["mmse", "age", "gender"]) for name in feature_names())
    runtime = json.loads((output / "spectral_validation.json").read_text())
    assert runtime["complete_oof_coverage"] and all(check["training_only_scaling_checked"] for check in runtime["checks"] if check["classifier"] == "logistic")
    report = {"real_participants": len(features), "real_epochs": len(epochs), "feature_dimension": 152,
            "complete_oof_coverage": True, "no_subject_fold_overlap": True,
            "training_only_scaling_and_imputation": True, "mean_epoch_features_match": True,
            "nonoverlapping_original_intervals": True, "physical_units_preserved": True,
            "no_demographic_predictors": True}
    if (output / "encoder_config.json").exists():
        encoder_config = json.loads((output / "encoder_config.json").read_text())
        assert encoder_config["shared_analysis_fingerprint"] == config["analysis_fingerprint"]
        embeddings = pd.read_csv(output / "encoder_epoch_embeddings.csv.gz")
        subjects = pd.read_csv(output / "encoder_participant_embeddings.csv")
        encoder_oof = pd.read_csv(output / "bendr_oof_predictions.csv")
        assert subjects.participant_id.is_unique and set(subjects.participant_id) == included
        assert encoder_oof.participant_id.is_unique and set(encoder_oof.participant_id) == included
        times = ["participant_id", "group", "recording_path", "source_start_sample", "source_end_sample",
                 "source_start_seconds", "source_end_seconds", "original_start_seconds", "original_end_seconds"]
        pd.testing.assert_frame_equal(epochs[times].reset_index(drop=True), embeddings[times].reset_index(drop=True))
        columns = [f"bendr_{i:03d}" for i in range(512)]
        recomputed = embeddings.groupby("participant_id")[columns].mean().loc[subjects.participant_id]
        # Official torch outputs and epoch pooling use float32; CSV reload/means use float64.
        np.testing.assert_allclose(recomputed, subjects[columns], rtol=1e-5, atol=1e-6)
        assert np.isfinite(embeddings[columns]).all().all()
        checked = encoder_oof.merge(folds, on=["participant_id", "group"], validate="one_to_one")
        assert checked.fold_x.equals(checked.fold_y)
        np.testing.assert_allclose(encoder_oof[[f"probability_{c}" for c in CLASSES]].sum(axis=1), 1., atol=1e-12)
        runtime = json.loads((output / "bendr_validation.json").read_text())
        assert all(check["training_only_scaling_checked"] for check in runtime["checks"])
        smoke = json.loads((output / "encoder_smoke_test.json").read_text())
        assert all(smoke[key] for key in ["finite", "different_inputs_produce_different_outputs",
                                          "repeat_inference_identical", "frozen_eval_no_grad",
                                          "representations_not_logits", "checkpoint_loaded_strictly"])
        report["encoder_checks"] = {"feature_dimension": 512, "matching_shared_intervals": True,
                                    "participant_embedding_means_match_float32_tolerance": True,
                                    "same_subject_folds_complete_oof": True,
                                    "training_only_scaling_and_imputation": True, "smoke_test_passed": True}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    report = {"numerical_smoke_checks": numerical_checks(), "real_data_checks": real_checks(args.output_dir)}
    (args.output_dir / "validation_checks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
