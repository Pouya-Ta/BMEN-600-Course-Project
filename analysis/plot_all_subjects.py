"""Checksum-verified downloads and raw amplitude heatmaps for all participants."""
import csv
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from plot_raw_eeg import DATA, OUT, META, S3, VERSION, fetch, metadata

GROUPS = {'A': 'Alzheimer’s disease', 'C': 'Healthy control', 'F': 'Frontotemporal dementia'}


def inspect_subject(row):
    subject = row['participant_id']
    relative = f'{subject}/eeg/{subject}_task-eyesclosed_eeg.set'
    path = DATA / relative
    for attempt in range(4):
        try:
            pointer = fetch(META + relative).decode()
            match = re.search(r'SHA256E-s(\d+)--([0-9a-f]{64})', pointer)
            if not match:
                raise ValueError(f'No checksum pointer: {subject}')
            size, sha = int(match[1]), match[2]
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                content = fetch(S3 + relative)
                if len(content) != size or hashlib.sha256(content).hexdigest() != sha:
                    raise ValueError(f'Download checksum mismatch: {subject}')
                temporary = path.with_suffix('.set.partial')
                temporary.write_bytes(content)
                temporary.replace(path)
                del content
            if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f'Cached checksum mismatch: {subject}')
            metadata(relative.replace('.set', '.json'))
            eeg = loadmat(path, simplify_cells=True)
            eeg = eeg.get('EEG', eeg)
            x = eeg['data']
            fs = float(eeg['srate'])
            if x.shape != (int(eeg['nbchan']), int(eeg['pnts'])) or int(eeg['trials']) != 1:
                raise ValueError(f'Unexpected recording shape: {subject}')
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite samples: {subject}')
            step = int(2 * fs)
            starts = np.arange(0, x.shape[1], step)
            amplitude = np.stack([np.ptp(x[:, s:min(s + step, x.shape[1])], axis=1)
                                  for s in starts], axis=1)
            edges = np.append(starts / fs, x.shape[1] / fs) / 60
            labels = [c['labels'] for c in eeg['chanlocs']]
            return dict(subject=subject, group=row['Group'], amplitude=amplitude,
                        edges=edges, labels=labels, sampling_hz=fs,
                        samples_per_channel=x.shape[1], sha256=sha, bytes=size,
                        source=S3 + relative)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    metadata('participants.tsv')
    with (DATA / 'participants.tsv').open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert len(rows) == 88 and len({r['participant_id'] for r in rows}) == 88
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(inspect_subject, row) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(f'{len(records)}/88 verified and measured: {result["subject"]}', flush=True)
    records.sort(key=lambda r: r['subject'])
    OUT.mkdir(exist_ok=True)
    all_values = np.concatenate([r['amplitude'].ravel() for r in records])
    positive = all_values[all_values > 0]
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    maxtime = max(r['edges'][-1] for r in records)
    # One row per participant; use the strongest channel in each window.
    fig, ax = plt.subplots(figsize=(16, 20), layout='constrained')
    for i, r in enumerate(records):
        mesh = ax.pcolormesh(r['edges'], [i, i + 1],
                            np.maximum(r['amplitude'].max(axis=0)[None, :], norm.vmin),
                            cmap='magma', norm=norm, shading='flat', rasterized=True)
    ax.set_yticks(np.arange(88) + .5,
                 [f'{r["subject"]} ({r["group"]})' for r in records], fontsize=7)
    ax.set_ylim(88, 0)
    ax.set_xlim(0, maxtime)
    for boundary in [36, 65]:
        ax.axhline(boundary, color='cyan', linewidth=1)
    ax.set_xlabel('Time from recording start (minutes)')
    ax.set_title('All 88 participants — raw EEG amplitude overview\n'
                 'Largest channel peak-to-peak amplitude per 2-second window', fontsize=15)
    fig.colorbar(mesh, ax=ax, fraction=.025, pad=.015).set_label('Peak-to-peak amplitude (µV; logarithmic scale)')
    fig.supxlabel('A = Alzheimer’s disease · C = healthy control · F = frontotemporal dementia\n'
                  'Brighter = larger excursions, not confirmed artifacts. Blank = recording ended. No preprocessing.', fontsize=10)
    fig.savefig(OUT / 'all_subjects_amplitude_overview.png', dpi=160)
    fig.savefig(OUT / 'all_subjects_amplitude_overview.pdf')
    plt.close(fig)
    # Detailed pages preserve exactly the channel × time measure from the earlier plot.
    with PdfPages(OUT / 'all_subjects_channel_heatmaps.pdf') as pdf:
        for page_start in range(0, len(records), 8):
            batch = records[page_start:page_start + 8]
            fig, axes = plt.subplots(4, 2, figsize=(18, 13), layout='constrained', sharex=True)
            for ax, r in zip(axes.ravel(), batch):
                mesh = ax.pcolormesh(r['edges'], np.arange(len(r['labels']) + 1),
                                    np.maximum(r['amplitude'], norm.vmin), cmap='magma',
                                    norm=norm, shading='flat', rasterized=True)
                ax.set_yticks(np.arange(len(r['labels'])) + .5, r['labels'], fontsize=6)
                ax.set_ylim(len(r['labels']), 0)
                ax.set_xlim(0, maxtime)
                ax.tick_params(labelbottom=True)
                ax.set_xlabel('Minutes', fontsize=8)
                ax.set_title(f'{r["subject"]} — {GROUPS[r["group"]]} · {r["edges"][-1]:.1f} min', fontsize=10)
            fig.colorbar(mesh, ax=list(axes.ravel()), fraction=.018, pad=.012).set_label('Peak-to-peak amplitude (µV; log scale)')
            fig.suptitle('Raw EEG: peak-to-peak amplitude per channel in 2-second windows', fontsize=15)
            fig.supxlabel('Shared color scale across all 88 subjects · no preprocessing · brighter ≠ confirmed artifact · blank = recording ended', fontsize=10)
            pdf.savefig(fig)
            fig.savefig(OUT / f'channel_heatmaps_page_{page_start // 8 + 1:02d}.png', dpi=130)
            plt.close(fig)
    np.savez_compressed(OUT / 'all_subjects_amplitude_metrics.npz',
                        **{f'{r["subject"]}_ptp_uv': r['amplitude'] for r in records},
                        **{f'{r["subject"]}_edges_minutes': r['edges'] for r in records})
    manifest = dict(dataset='ds004504', version=VERSION, participant_count=88,
                    window_seconds=2, final_partial_windows_included=True,
                    preprocessing='None', color_scale_uv=[norm.vmin, norm.vmax],
                    overview_aggregation='Maximum channel peak-to-peak amplitude per window',
                    recordings=[{k: v for k, v in r.items() if k not in ['amplitude', 'edges']} |
                                {'duration_seconds': float(r['edges'][-1] * 60),
                                 'windows': r['amplitude'].shape[1]} for r in records])
    (OUT / 'all_subjects_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Done: all 88 recordings verified; overview and 11 detailed pages saved.', flush=True)


if __name__ == '__main__':
    main()
