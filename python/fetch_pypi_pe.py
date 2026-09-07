#!/usr/bin/env python3
"""Benign real-PE corpus from PyPI Windows wheels: family = project, first_seen = release date.
Same layout as fetch_bazaar.py (samples/<sha256> + metadata.csv), for dry runs without malware.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = ["numpy", "pillow", "lxml", "pywin32", "aiohttp", "pyzmq", "psutil", "regex"]
STABLE = re.compile(r"^\d+(\.\d+)*$")
PE_EXT = (".pyd", ".dll", ".exe")


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ncd-threat"}), timeout=300) as r:
        return r.read()


def pick_wheel(files: list[dict]) -> dict | None:
    wheels = [f for f in files if f["filename"].endswith("win_amd64.whl") and not f.get("yanked")]
    if not wheels:
        return None
    return max(wheels, key=lambda f: [int(x) for x in re.findall(r"-cp(\d)(\d+)-", f["filename"])[:1][0]]
               if re.search(r"-cp\d+-", f["filename"]) else [0, 0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packages", nargs="+", default=DEFAULT)
    ap.add_argument("--releases", type=int, default=5, help="releases per project, spread over time")
    ap.add_argument("--per-wheel", type=int, default=4, help="PE members per wheel")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--until", default="9999", help="exclude releases uploaded on or after this date")
    ap.add_argument("--min-size", type=int, default=16 << 10)
    ap.add_argument("--max-size", type=int, default=8 << 20)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    store = args.out / "samples"
    store.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for pkg in args.packages:
        info = json.loads(get(f"https://pypi.org/pypi/{pkg}/json"))
        cands = []
        for ver, files in info["releases"].items():
            w = pick_wheel(files) if STABLE.match(ver) else None
            if w and args.since <= w["upload_time_iso_8601"] < args.until:
                cands.append((w["upload_time_iso_8601"], ver, w))
        cands.sort()
        if len(cands) > args.releases:
            idx = np.linspace(0, len(cands) - 1, args.releases).round().astype(int)
            cands = [cands[i] for i in sorted(set(idx))]
        n = 0
        for when, ver, w in cands:
            z = zipfile.ZipFile(io.BytesIO(get(w["url"])))
            members = sorted(m.filename for m in z.infolist()
                             if m.filename.lower().endswith(PE_EXT) and args.min_size <= m.file_size <= args.max_size)
            if len(members) > args.per_wheel:
                members = sorted(rng.choice(members, args.per_wheel, replace=False))
            for m in members:
                data = z.read(m)
                sha = hashlib.sha256(data).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                (store / sha).write_bytes(data)
                rows.append({"sha256": sha, "family": pkg, "first_seen": when, "version": ver,
                             "member": m, "file_size": len(data)})
                n += 1
        print(f"{pkg}: {len(cands)} releases, {n} binaries")
    pd.DataFrame(rows).to_csv(args.out / "metadata.csv", index=False)
    print(f"{len(rows)} samples in {store}")


if __name__ == "__main__":
    main()
