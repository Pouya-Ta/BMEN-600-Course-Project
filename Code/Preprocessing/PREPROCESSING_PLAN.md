# Preprocessing protocol

Scope: prepare shared EEG input for a comparison of classical features and frozen
foundation-model embeddings. This stage fits no classifier and estimates no cohort-wide
normalization. The proposal's eight-second windows and subject-level unit of analysis
are retained. Implementation date: October 14, 2026, America/Edmonton.

1. Verify local data against the Git-annex SHA256E keys and metadata bytes in the
   published OpenNeuro ds004504 tag 1.0.9. Audit all 88 original and 88 author derivative
   recordings, not only a representative subject. Decode diagnosis from participants.json.
2. Use the author derivatives as the primary source, consistent with the proposal and
   existing repository. Confirm the embedded history: A1/A2 reference, 0.5-45 Hz
   `pop_eegfiltnew` FIR filtering, ASR (burst criterion 17), extended RunICA, and ICLabel
   eye/muscle component rejection at probability >=0.9. The README's Butterworth
   description conflicts with the history; report the history-based interpretation.
3. Preserve the supplied reference and measured amplitude. A1/A2 are absent from the
   19 saved channels, so another mastoid rereference cannot be reconstructed. Do not
   repeat ICA, ASR, or the bandpass on these already cleaned recordings. No added 50 Hz
   notch is used for the existing 0.5-45 Hz passband. Raw recording annotations were
   omitted upstream, and provider cleaning is not evidence that all artifacts are gone.
4. Recover the original sample timeline from `clean_sample_mask`, cross-check every
   internal EEGLAB boundary, and restart segmentation at each uninterrupted stretch.
   Resample each stretch independently from 500 to 256 Hz with a centered anti-aliasing
   polyphase FIR (SciPy, Kaiser beta 5, line padding). Reserve one second at each edge
   of each stretch, including recording endpoints. The guard is a conservative design
   choice for discontinuities and resampling edges; it cannot guarantee removal of all
   transients created upstream. Eight-second windows never overlap or cross removed data.
5. Reject a window if any channel has peak-to-peak amplitude >500 microvolts, peak-to-peak
   <=0.1 microvolts, or nonfinite values, either before or after resampling. Apply rules
   equally to all diagnoses; never tune them on classification scores. Flag retained
   windows above 200 microvolts peak-to-peak for visual review. Do not automatically
   interpolate bad channels or assume 19 independent spatial dimensions after ICA.
6. The 500 microvolt threshold is a **data-informed conservative engineering choice**,
   not a clinically validated or preregistered cutoff: the initial 200 microvolt candidate
   would remove roughly half of provider-cleaned baseline windows and leave three
   participants with zero windows. Save 200/300/500/1000 microvolt sensitivity counts.
   Manual review and later performance sensitivity analyses remain necessary for a paper.
7. Store all retained epochs as float32 NumPy arrays in SI volts, ordered as
   `(epoch, 19 channels, 2048 samples)`. Preserve original and derivative half-open sample
   intervals and IDs in TSV sidecars, including rejected windows and reasons. Publish
   participant/group retention, source checksums, software versions, code hashes, and QC.
   Flag <120 seconds or <50% candidate-window retention for review, without silently
   excluding participants. No age, Gender or MMSE is embedded in the signal tensor.
8. Validate all saved values, dimensions, counts, units, input/output checksums and all
   original-timeline intervals. Recompute three saved epochs per subject directly from
   the original derivative to check amplitude, unit conversion and resampling alignment.

Both representation branches must consume the **same retained epoch IDs**. A foundation
model may need a documented channel/rate/unit adapter; resample from this common input
with anti-aliasing and fit no transform on held-out participants. Never apply global
z-scoring to the shared signals: it would erase absolute power information. Future
feature scaling, PCA, feature selection and hyperparameter fitting belong inside
training folds. All epochs from a participant must stay together. These 88 participants,
rather than thousands of correlated windows, are the independent sample size.

The default output is `../Outputs/preprocessed data`. Arrays are project-specific
derivatives, not a claim of full BIDS compliance. Original dataset files are read only.

References: https://openneuro.org/datasets/ds004504/versions/1.0.9 ;
https://doi.org/10.3390/data8060095 ;
https://github.com/Pouya-Ta/BMEN-600-Course-Project/tree/7fa3c0491839cd2157bf9bec1be4370b28cafecf ;
https://mne.tools/stable/auto_tutorials/preprocessing/30_filtering_resampling.html
