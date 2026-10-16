# Data and Code Availability

Project code, configuration, pinned software requirements, preliminary tables/figures,
and validation/provenance records are available at
<https://github.com/Pouya-Ta/BMEN-600-Course-Project> (public access verified October 15, 2026).
The repository's original code is distributed under its [MIT license](LICENSE).
The [README](README.md) identifies what to open and gives setup and reproduction commands.

The source dataset is **OpenNeuro ds004504, version 1.0.9**,
DOI [10.18112/openneuro.ds004504.v1.0.9](https://doi.org/10.18112/openneuro.ds004504.v1.0.9),
available at <https://openneuro.org/datasets/ds004504/versions/1.0.9>.
Its [published metadata](https://github.com/OpenNeuroDatasets/ds004504/blob/1.0.9/dataset_description.json)
specifies **CC0**. No login, individual authorization, or data-use application is
required for the public files. The original dataset was collected by the dataset authors;
this team performs secondary analysis of the released data.

The dataset includes 88 participants: 36 AD, 23 FTD, and 29 HC, with 19-channel
resting eyes-closed EEG and demographic/cognitive metadata. We use supplied cleaned
EEGLAB derivatives and three raw recordings for QC examples. The saved provenance
records snapshot commit `63b2ec152bc3cd4487d131124f10f767ebe543bc`,
access date October 10, 2026, source URLs, file sizes, and SHA256 checksums.
The downloader verifies these files and writes a manifest required by the analysis.

Raw/cleaned EEG (~2.9 GB for this workflow), downloaded model weights, third-party
source caches, virtual environments, and large epoch embedding caches are omitted
from Git because of size and regeneration, rather than a known dataset sharing
restriction. Participant features/embeddings, folds, predictions, metrics, and figures
are committed. Original automatic artifact annotations were not provided by the
dataset authors and cannot be shared or reconstructed here. Third-party BENDR/DN3
materials are fetched from the sources identified in the workflow and retain their
upstream license terms; the repository's MIT license does not relicense them.

Please acknowledge the dataset authors using the citations requested in their metadata:
[Miltiadous et al., data descriptor (2023)](https://doi.org/10.3390/data8060095),
[DICE-net (2023)](https://doi.org/10.1109/ACCESS.2023.3294618), and
[AHEPA EEG benchmark (2026)](https://doi.org/10.1007/s11571-026-10464-w).

