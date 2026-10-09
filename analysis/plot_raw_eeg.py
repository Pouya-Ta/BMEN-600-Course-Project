"""Download three original ds004504 recordings and plot unmodified EEG samples."""
from pathlib import Path
import csv
import hashlib
import json
import re
import requests
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'ds004504'
OUT = ROOT / 'plots'
VERSION = '1.0.9'
META = f'https://raw.githubusercontent.com/OpenNeuroDatasets/ds004504/{VERSION}/'
S3 = 'https://s3.amazonaws.com/openneuro.org/ds004504/'
SUBJECTS = [('sub-001', 'Alzheimer’s disease', 'A'),
            ('sub-037', 'Healthy control', 'C'),
            ('sub-066', 'Frontotemporal dementia', 'F')]

def fetch(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def metadata(relative):
    path = DATA / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch(META + relative))
    return path

def main():
    OUT.mkdir(exist_ok=True)
    for name in ['README', 'dataset_description.json', 'participants.tsv', 'participants.json']:
        metadata(name)
    with (DATA / 'participants.tsv').open() as f:
        participants = list(csv.DictReader(f, delimiter='\t'))
    rows = {p['participant_id']: p for p in participants}
    fig, axes = plt.subplots(1, 3, figsize=(18, 11), sharex=True, sharey=True)
    summaries = []
    display_limits = []
    spacing = 150  # Display offsets only; original samples are not centered or scaled.
    for ax, (subject, group, code) in zip(axes, SUBJECTS):
        relative = f'{subject}/eeg/{subject}_task-eyesclosed_eeg.set'
        path = DATA / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pointer = fetch(META + relative).decode()
        checksum = re.search(r'SHA256E-s(\d+)--([0-9a-f]{64})', pointer)
        if checksum is None:
            raise ValueError('Expected a git-annex SHA256 pointer')
        if not path.exists():
            print(f'Downloading {subject}', flush=True)
            content = fetch(S3 + relative)
            if len(content) != int(checksum[1]) or hashlib.sha256(content).hexdigest() != checksum[2]:
                raise ValueError(f'Integrity check failed for {subject}')
            path.write_bytes(content)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != checksum[2]:
            raise ValueError(f'Local file does not match version {VERSION}: {path}')
        metadata(relative.replace('.set', '.json'))
        eeg = loadmat(path, simplify_cells=True)
        eeg = eeg.get('EEG', eeg)
        samples = eeg['data']
        fs = float(eeg['srate'])
        labels = [c['labels'] for c in eeg['chanlocs']]
        assert samples.shape == (int(eeg['nbchan']), int(eeg['pnts']))
        assert int(eeg['trials']) == 1
        assert rows[subject]['Group'] == code
        start, stop = int(60 * fs), int(70 * fs)
        t = np.arange(start, stop) / fs
        offsets = np.arange(len(labels))[::-1] * spacing
        for i, offset in enumerate(offsets):
            ax.plot(t, samples[i, start:stop] + offset, linewidth=.55, color='#245c87')
        ax.set_yticks(offsets, labels)
        display_limits.append((float(np.min(samples[:, start:stop] + offsets[:, None])),
                               float(np.max(samples[:, start:stop] + offsets[:, None]))))
        ax.set_xlim(60, 70)
        ax.grid(axis='x', alpha=.2)
        duration = samples.shape[1] / fs
        p = rows[subject]
        ax.set_title(f'{group}\n{subject} · age {p["Age"]} · MMSE {p["MMSE"]}\nFull recording: {duration / 60:.1f} min', fontsize=11)
        ax.set_xlabel('Time from recording start (seconds)')
        ax.tick_params(labelleft=True)
        ax.plot([69.5, 69.5], [-100, -50], color='black', linewidth=2)
        ax.text(69.4, -75, '50 µV', ha='right', va='center', fontsize=9)
        summaries.append(dict(subject=subject, group=group, channels=labels, sampling_hz=fs,
                              samples_per_channel=samples.shape[1], duration_seconds=duration,
                              source=S3 + relative, sha256=digest))
    axes[0].set_ylim(min(v[0] for v in display_limits) - spacing,
                     max(v[1] for v in display_limits) + spacing)
    axes[0].set_ylabel('EEG channel (vertically offset for display)')
    fig.suptitle('OpenNeuro ds004504 — original eyes-closed resting EEG', fontsize=17)
    fig.text(.5, .025, '60–70 s excerpt · 19 channels · 500 Hz · same amplitude scale in all panels\nNo added filtering, re-referencing, resampling, normalization, or artifact removal. Display offsets: 150 µV.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .065, 1, .94])
    fig.savefig(OUT / 'raw_eeg_examples.png', dpi=160)
    fig.savefig(OUT / 'raw_eeg_examples.pdf')
    plt.close(fig)
    counts = {g: sum(p['Group'] == c for p in participants) for _, g, c in SUBJECTS}
    report = dict(dataset='ds004504', version=VERSION, participant_count=len(participants),
                  group_counts=counts, downloaded_recordings=summaries, plot_window_seconds=[60, 70],
                  preprocessing='None; only vertical offsets added for display')
    (OUT / 'dataset_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
