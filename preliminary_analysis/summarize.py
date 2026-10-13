"""Generate a compact table and a short factual handoff note from saved results."""
import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(output):
    output = Path(output)
    groups = pd.read_csv(output / "group_summary.csv")
    manifest = pd.read_csv(output / "participant_manifest.csv")
    points = pd.read_csv(output / "relative_theta_alpha_points.csv")
    provenance = json.loads((output / "dataset_provenance.json").read_text())
    qc = json.loads((output / "qc_summary.json").read_text())
    compact = []
    for row in groups.to_dict("records"):
        compact.append({"group": row["group"], "participants_included": int(row["n_included"]),
                        "participants_excluded": int(row["n_excluded"]),
                        "age_years_mean_sd": f"{row['age_years_mean']:.1f} ± {row['age_years_sd']:.1f}",
                        "sex_gender_F_M": f"{row['sex_F']}/{row['sex_M']}",
                        "mmse_mean_sd": f"{row['mmse_mean']:.2f} ± {row['mmse_sd']:.2f}",
                        "original_minutes_mean_sd": f"{row['original_duration_seconds_mean']/60:.2f} ± {row['original_duration_seconds_sd']/60:.2f}",
                        "derivative_minutes_mean_sd": f"{row['recording_duration_seconds_mean']/60:.2f} ± {row['recording_duration_seconds_sd']/60:.2f}",
                        "usable_minutes_mean_sd": f"{row['usable_duration_seconds_mean']/60:.2f} ± {row['usable_duration_seconds_sd']/60:.2f}",
                        "usable_epochs_total": int(row["usable_epochs_total"]),
                        "age_sex_mmse_missing": f"{row['age_years_missing']}/{row['sex_missing']}/{row['mmse_missing']}"})
    pd.DataFrame(compact).to_csv(output / "group_summary_compact.csv", index=False)
    files = ["spectral_metrics.csv"] + (["bendr_metrics.csv"] if (output / "bendr_metrics.csv").exists() else [])
    metrics = pd.concat([pd.read_csv(output / name) for name in files], ignore_index=True)
    lines = ["# Preliminary results and limitations", "",
             "Generated from saved real-data outputs; this is a workflow handoff note, not the research report.", "",
             f"Dataset: ds004504 version {provenance['version']}, accessed {provenance['access_date']} (America/Edmonton).",
             f"Snapshot commit: `{provenance['git_commit']}`.", "",
             f"All {len(manifest)} participants matched one derivative recording and were retained: " +
             ", ".join(f"{item['group']} {item['participants_included']}" for item in compact) + ".",
             f"There are {int(manifest.usable_epochs.sum()):,} usable 8-second epochs "
             f"({manifest.usable_duration_seconds.sum()/3600:.2f} hours). "
             f"Provider masks remove {manifest.provider_removed_seconds.sum()/60:.2f} minutes; "
             f"epoching leaves {manifest.unepoched_tail_seconds.sum()/60:.2f} minutes of short contiguous-run tails unused.",
             f"The audit found {int(manifest.nonfinite_samples.sum())} nonfinite samples, "
             f"{int(manifest.rejected_epochs.sum())} additionally rejected candidate epochs, "
             f"and {int(manifest.amplitude_review_epochs.sum())} complete epochs above the 1,000 µV review flag. "
             f"Whole-recording maxima range from {manifest.max_abs_uv.min():.1f} to {manifest.max_abs_uv.max():.1f} µV; "
             "the largest excursions lie outside retained complete epochs. This does not establish artifact freedom.", "",
             "| Group | n | Age, years (mean ± SD) | F/M as recorded | MMSE (mean ± SD) | Usable minutes (mean ± SD) | Epochs |",
             "|---|---:|---|---|---|---|---:|"]
    for row in compact:
        lines.append(f"| {row['group']} | {row['participants_included']} | {row['age_years_mean_sd']} | "
                     f"{row['sex_gender_F_M']} | {row['mmse_mean_sd']} | {row['usable_minutes_mean_sd']} | {row['usable_epochs_total']} |")
    lines.extend(["", "Age, recorded sex/gender, and MMSE have no missing values. The numeric group summary also reports duration missingness and exclusions.",
                  "Age, sex/gender composition, cognition, and recording lengths differ across groups; these are potential imbalances, not causal findings.", "",
                  "| Representation / classifier | Fold balanced accuracy, mean ± SD | Fold macro-F1, mean ± SD | Pooled OOF balanced accuracy | Pooled recall AD / FTD / HC |",
                  "|---|---|---|---|---|"])
    for row in metrics.to_dict("records"):
        lines.append(f"| {row['representation']} / {row['classifier']} | "
                     f"{row['balanced_accuracy_fold_mean']:.3f} ± {row['balanced_accuracy_fold_sd']:.3f} | "
                     f"{row['macro_f1_fold_mean']:.3f} ± {row['macro_f1_fold_sd']:.3f} | "
                     f"{row['balanced_accuracy_pooled_oof']:.3f} | {row['recall_AD_pooled_oof']:.3f} / "
                     f"{row['recall_FTD_pooled_oof']:.3f} / {row['recall_HC_pooled_oof']:.3f} |")
    medians = points.groupby(["band", "group"]).relative_power.median()
    lines.extend(["", "Five fixed stratified folds contain participants, with one OOF prediction per participant per classifier. "
                  "Scaling and median imputation are fit on each training fold. C=1, L2 multinomial logistic regression, no class weighting or hyperparameter search. "
                  "Fold SD uses ddof=1 and describes split variability; it is not a confidence interval. Three-class balanced-accuracy chance is 1/3.", "",
                  f"Participant median channel-averaged relative alpha power is AD {medians.loc['alpha','AD']:.3f}, "
                  f"FTD {medians.loc['alpha','FTD']:.3f}, HC {medians.loc['alpha','HC']:.3f}. "
                  "The group spectra show a larger average alpha-region peak in HC, with overlapping participant distributions. "
                  "Substantial 1–4 Hz power influences the 1–30 Hz relative-power denominator. "
                  "The plots demonstrate feasibility of participant-level summaries, not diagnostic validation.", "",
                  "The handcrafted classifier's FTD recall is low. A result above the dummy on these folds does not establish clinical utility. "
                  "The small single-site cohort, unequal group sizes, demographic differences, residual artifacts, provided preprocessing, "
                  "and absence of external validation limit interpretation. No demographic predictors were used; no confidence intervals or inferential tests were added.", "",
                  "Provenance limitations: the README calls the provided filter Butterworth, while every supplied recording history uses `pop_eegfiltnew`; "
                  "the raw README and sidecars disagree on reference/filter details. The FTD MMSE SD is 2.64 from participants.tsv, "
                  "although the dataset README states 8.22. We preserve derivative amplitudes/reference, apply no new filtering/ICA, and use the actual metadata."])
    if (output / "encoder_config.json").exists():
        config = json.loads((output / "encoder_config.json").read_text())
        lines.extend(["", "Optional step 4 completed with the official 2021 BENDR encoder release: frozen convolutional representations, "
                      "512 dimensions, temporal mean per matching shared epoch, then participant mean. Official DN3 adaptation uses "
                      "256 Hz, 19 EEG sensors plus SCALE, per-window global minmax normalization, and the documented zero scale channel when dataset bounds are unspecified. "
                      "These decisions affect transfer and are recorded in encoder_config.json. No weights were fine-tuned.",
                      "No ds004504 overlap is documented in the official pretraining/evaluation configurations, and the checkpoint predates the dataset release. "
                      "An exhaustive participant-level contamination audit is unavailable. The comparison is a preliminary protocol-specific result; "
                      "paired folds do not justify a superiority claim.",
                      f"Encoder extraction ran on {config['device']}. NVIDIA NVML reported a driver/library mismatch on this computer; "
                      "system drivers were not changed. The optional CPU installation and extraction succeeded."])
    else:
        lines.extend(["", "Optional encoder extraction is not complete; there are no fabricated embeddings or comparison results."])
    lines.extend(["", "Start with group_summary_compact.csv, group_psd.png/.pdf, relative_theta_alpha.png/.pdf, "
                  "spectral_metrics.csv, and the pooled OOF confusion matrices. "
                  "See README_PRELIMINARY.md for reproduction, complete output definitions, and validation commands.", ""])
    (output / "RESULTS_NOTES.md").write_text("\n".join(lines))
    print(f"Saved {output / 'group_summary_compact.csv'} and {output / 'RESULTS_NOTES.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    summarize(args.output_dir)
