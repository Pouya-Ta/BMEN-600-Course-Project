"""Download a pinned ds004504 snapshot; verify annex SHA256 keys, including caches."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import re
import time

import requests

VERSION = "1.0.9"
REPO = "OpenNeuroDatasets/ds004504"
S3 = "https://s3.amazonaws.com/openneuro.org/ds004504/"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get(url):
    response = requests.get(url, timeout=(20, 120))
    response.raise_for_status()
    return response


def download(data_dir, version=VERSION, workers=4, raw_examples=True):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    old = root / "download_manifest.json"
    if old.exists() and json.loads(old.read_text())["version"] != version:
        raise ValueError("Use a separate directory for a different snapshot")
    commit = get(f"https://api.github.com/repos/{REPO}/commits/{version}").json()["sha"]
    tree = get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1").json()
    if tree.get("truncated"):
        raise ValueError("Incomplete remote tree")
    base = f"https://raw.githubusercontent.com/{REPO}/{commit}/"
    paths = [entry["path"] for entry in tree["tree"] if entry["type"] == "blob"]
    recordings = [p for p in paths if p.startswith("derivatives/") and p.endswith("_eeg.set")]
    metadata = [p for p in paths if p in ("README", "participants.tsv", "participants.json", "dataset_description.json", "CHANGES")
                or (not p.startswith("derivatives/") and p.endswith(("_channels.tsv", "_eeg.json")))]
    if raw_examples:
        recordings += [p for p in paths if p.endswith("_eeg.set") and p.split("/")[0] in ("sub-001", "sub-037", "sub-066")]

    def fetch_one(relative):
        for attempt in range(4):
            try:
                content = get(base + relative).content
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                key = re.search(rb"SHA256E-s(\d+)--([0-9a-f]{64})", content)
                expected = key[2].decode() if key else hashlib.sha256(content).hexdigest()
                size = int(key[1]) if key else len(content)
                if not path.exists() or path.stat().st_size != size or sha256(path) != expected:
                    temp = path.with_name(path.name + ".partial")
                    if key:
                        with requests.get(S3 + relative, stream=True, timeout=(20, 120)) as response:
                            response.raise_for_status()
                            with temp.open("wb") as stream:
                                for block in response.iter_content(2**20):
                                    stream.write(block)
                    else:
                        temp.write_bytes(content)
                    if temp.stat().st_size != size or sha256(temp) != expected:
                        raise ValueError(f"Snapshot checksum mismatch: {relative}")
                    temp.replace(path)
                return {"path": relative, "sha256": expected, "bytes": size,
                        "url": S3 + relative if key else base + relative}
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    files = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, p): p for p in metadata + recordings}
        for future in as_completed(futures):
            result = future.result()
            files.append(result)
            if result["path"].endswith(".set"):
                print(f"Verified {result['path']}", flush=True)
    provenance = {"dataset": "ds004504", "version": version, "git_commit": commit,
                  "access_date": datetime.now(ZoneInfo("America/Edmonton")).date().isoformat(),
                  "dataset_url": f"https://openneuro.org/datasets/ds004504/versions/{version}",
                  "recording_inventory": sorted(p for p in paths if p.endswith("_eeg.set")),
                  "files": sorted(files, key=lambda item: item["path"])}
    old.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {old}; {len(recordings)} checksum-verified recordings", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data/ds004504")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-raw-examples", action="store_true")
    args = parser.parse_args()
    download(args.data_dir, args.version, args.workers, not args.no_raw_examples)
