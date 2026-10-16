# BMEN 600: EEG classification of dementia

Team **Seventieth** — Luke Fatovich, Pouya Taghipour L., Hamid Salmani,
Dera Okemeziem, and David A. Martinez G.

**Research question:** Do frozen pretrained EEG representations with a linear probe
outperform handcrafted spectral features for distinguishing Alzheimer's disease (AD),
frontotemporal dementia (FTD), and healthy controls (HC) when data are limited?

**Repository:** <https://github.com/Pouya-Ta/BMEN-600-Course-Project>

## Open the current progress check

Start with [preliminary results and limitations](preliminary_analysis/results/RESULTS_NOTES.md),
the [participant summary table](preliminary_analysis/results/group_summary_compact.csv),
and the [group EEG spectra](preliminary_analysis/results/group_psd.png).
These committed outputs can be viewed on GitHub without installing software or downloading EEG.
The [baseline scores](preliminary_analysis/results/spectral_metrics.csv) and
[representation comparison](preliminary_analysis/results/representation_comparison.csv)
contain the current classification results.

The workflow retained 88 participants (36 AD, 23 FTD, 29 HC) and 8,464 usable eight-second
epochs. Five fixed participant-level folds keep each person's epochs together.
Mean fold balanced accuracy is 0.500 for spectral logistic regression, 0.386 for the
frozen BENDR encoder plus logistic regression, and 0.333 for the majority-class baseline.
These are preliminary single-site results with weak FTD recall and no external validation;
they do not establish clinical utility or model superiority.

## Dataset access and availability

Use [OpenNeuro ds004504, version 1.0.9](https://openneuro.org/datasets/ds004504/versions/1.0.9)
([DOI](https://doi.org/10.18112/openneuro.ds004504.v1.0.9)). It contains eyes-closed
resting EEG, 19 channels, raw recordings, supplied cleaned derivatives, and participant metadata.
The analysis uses the supplied derivatives; the downloader also obtains three raw QC examples.
The pinned [dataset metadata](https://github.com/OpenNeuroDatasets/ds004504/blob/1.0.9/dataset_description.json)
licenses the dataset under **CC0**. Public download requires no account or application.

EEG files (~2.9 GB), model weights, environments, and caches are downloaded separately
and excluded from Git for size and reproducibility. There is no dataset access restriction
identified in the published snapshot. Code is available under the [MIT license](LICENSE);
third-party data and model materials retain their own terms. See the complete
[Data and Code Availability statement](DATA_AND_CODE_AVAILABILITY.md).

## Setup and reproduce

Platform: **Python scripts on Linux, CPU**. The saved run used Python **3.12.3**;
use Python 3.12. Main packages are NumPy, SciPy, pandas, scikit-learn, Matplotlib,
MNE, and requests, pinned in [requirements.txt](preliminary_analysis/requirements.txt).
A GPU, notebook server, and MATLAB are unnecessary. Internet access and several GB
of free storage are needed for a full rerun.

Run these commands in a Linux/macOS terminal. On Windows, use WSL for these shell commands.

```bash
git clone https://github.com/Pouya-Ta/BMEN-600-Course-Project.git
cd BMEN-600-Course-Project
python3.12 -m venv preliminary_analysis/.venv
source preliminary_analysis/.venv/bin/activate
python -m pip install -r preliminary_analysis/requirements.txt
```

For a quick progress check using the committed outputs, without downloading EEG:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/validate.py \
  --output-dir preliminary_analysis/results
python preliminary_analysis/summarize.py --output-dir preliminary_analysis/results
```

This checks numerical behavior and saved real-data outputs, then regenerates
`group_summary_compact.csv` and `RESULTS_NOTES.md`. It does not rerun signal extraction.
A successful validation exits with code 0 and prints a JSON check report.
The ignored encoder epoch cache is unnecessary for this check: if absent, the report
explicitly marks encoder interval/epoch-mean checks as skipped. Regenerate it with
`encoder.py` to include those checks.

For a full baseline rerun from source EEG, keep commands at the repository root:

```bash
python preliminary_analysis/download.py --data-dir preliminary_analysis/data/ds004504
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/run.py \
  --data-dir preliminary_analysis/data/ds004504 \
  --output-dir preliminary_analysis/results \
  --config preliminary_analysis/config.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python preliminary_analysis/validate.py \
  --output-dir preliminary_analysis/results
python preliminary_analysis/summarize.py --output-dir preliminary_analysis/results
```

This regenerates participant/channel QC, spectral features, group tables and figures,
folds, predictions, and baseline scores. The downloader verifies snapshot checksums
and creates the required download manifest; a manually copied EEG folder alone is insufficient.
These commands update the saved baseline outputs. Use a separate output directory
if changing the scientific configuration.

The optional BENDR comparison needs CPU PyTorch and separately downloaded pretrained
weights/source. Follow the [detailed workflow](preliminary_analysis/README_PRELIMINARY.md#optional-frozen-encoder)
for its installation and run commands, preprocessing, provenance, and limitations.
The same document explains all outputs and the repeated-run hash comparison.

## File map

| Path | Purpose |
|---|---|
| [preliminary_analysis/README_PRELIMINARY.md](preliminary_analysis/README_PRELIMINARY.md) | Detailed reproduction, methods, optional encoder, output definitions |
| [preliminary_analysis/download.py](preliminary_analysis/download.py) | Retrieve and verify the pinned dataset |
| [preliminary_analysis/run.py](preliminary_analysis/run.py), [config.json](preliminary_analysis/config.json) | Current end-to-end QC, spectral summaries, and classification baseline |
| `preliminary_analysis/{shared,features,evaluation,plots}.py` | Loading/epoching, spectral features, participant-level evaluation, plotting |
| [preliminary_analysis/encoder.py](preliminary_analysis/encoder.py) | Optional frozen BENDR representations and linear probe |
| `preliminary_analysis/{validate,summarize,check_reproducibility}.py` | Integrity checks, report tables/notes, repeated-run comparison |
| `preliminary_analysis/requirements*.txt` | Pinned baseline and optional encoder dependencies |
| [preliminary_analysis/results/](preliminary_analysis/results/) | Committed preliminary outputs, provenance, settings, and software versions |
| `preliminary_analysis/data/`, `results/cache/`, `models/`, `vendor/` | Generated local downloads/caches; ignored by Git |
| [analysis/](analysis/) | Earlier exploratory raw-EEG scripts and figures; current reproduction starts in `preliminary_analysis/` |
| [Project-plan/](Project-plan/) | Team planning spreadsheet |
| [Code/](Code/) | Original project scaffold; current executable code is in `preliminary_analysis/` |
| [DATA_AND_CODE_AVAILABILITY.md](DATA_AND_CODE_AVAILABILITY.md) | Dataset source, licensing, access, and omitted materials |
