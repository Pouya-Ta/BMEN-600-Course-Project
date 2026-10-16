# Preliminary results and limitations

Generated from saved real-data outputs; this is a workflow handoff note, not the research report.

Dataset: ds004504 version 1.0.9, accessed 2026-10-10 (America/Edmonton).
Snapshot commit: `63b2ec152bc3cd4487d131124f10f767ebe543bc`.

All 88 participants matched one derivative recording and were retained: AD 36, FTD 23, HC 29.
There are 8,464 usable 8-second epochs (18.81 hours). Provider masks remove 12.65 minutes; epoching leaves 35.32 minutes of short contiguous-run tails unused.
The audit found 0 nonfinite samples, 0 additionally rejected candidate epochs, and 0 complete epochs above the 1,000 µV review flag. Whole-recording maxima range from 133.6 to 1749.5 µV; the largest excursions lie outside retained complete epochs. This does not establish artifact freedom.

| Group | n | Age, years (mean ± SD) | F/M as recorded | MMSE (mean ± SD) | Usable minutes (mean ± SD) | Epochs |
|---|---:|---|---|---|---|---:|
| AD | 36 | 66.4 ± 7.9 | 24/12 | 17.75 ± 4.50 | 13.10 ± 2.70 | 3538 |
| FTD | 23 | 63.7 ± 8.2 | 9/14 | 22.17 ± 2.64 | 11.69 ± 2.52 | 2016 |
| HC | 29 | 67.9 ± 5.4 | 11/18 | 30.00 ± 0.00 | 13.38 ± 1.17 | 2910 |

Age, recorded sex/gender, and MMSE have no missing values. The numeric group summary also reports duration missingness and exclusions.
Age, sex/gender composition, cognition, and recording lengths differ across groups; these are potential imbalances, not causal findings.

| Representation / classifier | Fold balanced accuracy, mean ± SD | Fold macro-F1, mean ± SD | Pooled OOF balanced accuracy | Pooled recall AD / FTD / HC |
|---|---|---|---|---|
| spectral / logistic | 0.500 ± 0.119 | 0.492 ± 0.129 | 0.501 | 0.639 / 0.174 / 0.690 |
| spectral / dummy | 0.333 ± 0.000 | 0.193 ± 0.008 | 0.333 | 1.000 / 0.000 / 0.000 |
| bendr / logistic | 0.386 ± 0.124 | 0.369 ± 0.126 | 0.380 | 0.500 / 0.087 / 0.552 |

Five fixed stratified folds contain participants, with one OOF prediction per participant per classifier. Scaling and median imputation are fit on each training fold. C=1, L2 multinomial logistic regression, no class weighting or hyperparameter search. Fold SD uses ddof=1 and describes split variability; it is not a confidence interval. Three-class balanced-accuracy chance is 1/3.

Participant median channel-averaged relative alpha power is AD 0.040, FTD 0.045, HC 0.074. The group spectra show a larger average alpha-region peak in HC, with overlapping participant distributions. Substantial 1–4 Hz power influences the 1–30 Hz relative-power denominator. The plots demonstrate feasibility of participant-level summaries, not diagnostic validation.

The handcrafted classifier's FTD recall is low. A result above the dummy on these folds does not establish clinical utility. The small single-site cohort, unequal group sizes, demographic differences, residual artifacts, provided preprocessing, and absence of external validation limit interpretation. No demographic predictors were used; no confidence intervals or inferential tests were added.

Provenance limitations: the README calls the provided filter Butterworth, while every supplied recording history uses `pop_eegfiltnew`; the raw README and sidecars disagree on reference/filter details. The FTD MMSE SD is 2.64 from participants.tsv, although the dataset README states 8.22. We preserve derivative amplitudes/reference, apply no new filtering/ICA, and use the actual metadata.

Optional step 4 completed with the official 2021 BENDR encoder release: frozen convolutional representations, 512 dimensions, temporal mean per matching shared epoch, then participant mean. Official DN3 adaptation uses 256 Hz, 19 EEG sensors plus SCALE, per-window global minmax normalization, and the documented zero scale channel when dataset bounds are unspecified. These decisions affect transfer and are recorded in encoder_config.json. No weights were fine-tuned.
No ds004504 overlap is documented in the official pretraining/evaluation configurations, and the checkpoint predates the dataset release. An exhaustive participant-level contamination audit is unavailable. The comparison is a preliminary protocol-specific result; paired folds do not justify a superiority claim.
Encoder extraction ran on cpu. NVIDIA NVML reported a driver/library mismatch on this computer; system drivers were not changed. The optional CPU installation and extraction succeeded.

Start with group_summary_compact.csv, group_psd.png/.pdf, relative_theta_alpha.png/.pdf, spectral_metrics.csv, and the pooled OOF confusion matrices. See [the preliminary workflow README](../README_PRELIMINARY.md) for reproduction, complete output definitions, and validation commands.
