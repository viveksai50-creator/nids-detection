"""Download and extract the improved CIC-IDS2017 zip (resumable).

Run:
    python -m src.data.download
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from src.config import DATASET_URL, DATASET_ZIP, RAW_DIR


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:

    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            mode = "ab" if (existing and r.status == 206) else "wb"
            total = r.headers.get("Content-Length")
            print(f"downloading -> {dest} (mode={mode}, remaining={total} bytes)")
            with open(dest, mode) as f:
                while True:
                    block = r.read(chunk)
                    if not block:
                        break
                    f.write(block)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable: file already complete
            print("already complete")
        else:
            raise
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(zip_path: Path, out_dir: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        z.extractall(out_dir)
    return names


def main() -> int:
    download(DATASET_URL, DATASET_ZIP)
    print("sha256:", sha256(DATASET_ZIP))
    names = extract(DATASET_ZIP, RAW_DIR)
    print(f"extracted {len(names)} entries into {RAW_DIR}")
    for n in names:
        print("  ", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
