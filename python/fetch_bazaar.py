#!/usr/bin/env python3
"""Download labelled PE samples from MalwareBazaar into a flat store plus metadata.csv.
Needs an abuse.ch Auth-Key (--auth-key or ABUSECH_AUTH_KEY) and pyzipper (AES zips).
Samples are written read-only, named by SHA-256 and without extension. Never execute them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://mb-api.abuse.ch/api/v1/"
PASSWORD = b"infected"


class Api:
    def __init__(self, url: str, key: str, delay: float):
        self.url, self.key, self.delay = url, key, delay
        self.last = 0.0

    def post(self, **params) -> bytes:
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(self.url, data=urllib.parse.urlencode(params).encode(),
                                     headers={"Auth-Key": self.key, "User-Agent": "ncd-threat"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        self.last = time.monotonic()
        return body

    def siginfo(self, signature: str, limit: int) -> list[dict]:
        doc = json.loads(self.post(query="get_siginfo", signature=signature, limit=limit))
        status = doc.get("query_status")
        if status != "ok":
            print(f"  {signature}: {status}", file=sys.stderr)
            return []
        return doc.get("data") or []

    def file(self, sha256: str) -> bytes:
        import pyzipper

        body = self.post(query="get_file", sha256_hash=sha256)
        if not body.startswith(b"PK"):
            raise RuntimeError(json.loads(body).get("query_status", "unexpected response"))
        with pyzipper.AESZipFile(io.BytesIO(body)) as z:
            z.setpassword(PASSWORD)
            data = z.read(z.namelist()[0])
        if hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError("sha256 mismatch")
        return data


def spread(items: list[dict], n: int) -> list[dict]:
    """n items evenly spaced in first_seen order, so the time range is covered."""
    items = sorted(items, key=lambda d: d.get("first_seen") or "")
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", required=True, help="MalwareBazaar signatures, e.g. AgentTesla")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--file-types", nargs="+", default=["exe", "dll"])
    ap.add_argument("--max-size", type=int, default=16 << 20, help="skip larger samples (bytes)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--auth-key", default=os.environ.get("ABUSECH_AUTH_KEY"))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--api", default=API, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.auth_key:
        raise SystemExit("missing Auth-Key: get one at https://auth.abuse.ch/ and pass --auth-key or ABUSECH_AUTH_KEY")

    api = Api(args.api, args.auth_key, args.delay)
    store = args.out / "samples"
    store.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.csv"
    rows = pd.read_csv(meta_path).to_dict("records") if meta_path.exists() else []
    have = {r["sha256"] for r in rows}

    for fam in args.families:
        found = [d for d in api.siginfo(fam, 1000)
                 if d.get("file_type") in args.file_types and (d.get("file_size") or 0) <= args.max_size]
        picked = spread(found, args.per_family)
        print(f"{fam}: {len(found)} candidates, downloading {len(picked)}")
        for d in picked:
            sha = d["sha256_hash"].lower()
            if sha in have:
                continue
            try:
                data = api.file(sha)
            except Exception as e:  # noqa: BLE001
                print(f"  {sha[:16]}: {e}", file=sys.stderr)
                continue
            path = store / sha
            path.write_bytes(data)
            path.chmod(0o444)
            have.add(sha)
            rows.append({"sha256": sha, "family": fam, "first_seen": d.get("first_seen"),
                         "file_type": d.get("file_type"), "file_size": len(data)})
            pd.DataFrame(rows).to_csv(meta_path, index=False)
    print(f"{len(rows)} samples in {store}; metadata in {meta_path}")


if __name__ == "__main__":
    main()
