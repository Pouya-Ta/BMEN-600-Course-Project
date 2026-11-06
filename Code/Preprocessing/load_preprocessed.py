"""Read one subject's retained epochs (volts), with matching timestamps."""
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_common import DEFAULT_OUTPUT


def load_subject(participant_id, output_dir=DEFAULT_OUTPUT, mmap=True):
    output_dir = Path(output_dir)
    manifest = pd.read_csv(output_dir / "participant_manifest.csv")
    selected = manifest.loc[manifest.participant_id.eq(participant_id)]
    if len(selected) != 1:
        raise ValueError(f"Unknown/ambiguous subject: {participant_id}")
    row = selected.iloc[0]
    data = np.load(output_dir / row.array_path, mmap_mode="r" if mmap else None, allow_pickle=False)
    table = pd.read_csv(output_dir / row.epoch_table_path, sep="\t")
    retained = table.loc[table.retained].reset_index(drop=True)
    if not np.array_equal(retained.array_index, np.arange(len(data))):
        raise ValueError("Metadata and array are not aligned")
    return data, retained, float(row.output_sfreq_hz)
