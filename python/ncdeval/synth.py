"""Fabricated CICFlowMeter flows for tests and the window ablation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fake_flows(path: Path, rng: np.random.Generator) -> None:
    """One capture: 3 benign hosts, DoS Hulk, port scan, web brute force."""
    rows = []
    t0 = pd.Timestamp("2017-07-05 09:00")

    def flows(src, label, n, dur, pkts, port_fn, iat):
        t = t0 + pd.Timedelta(minutes=int(rng.integers(0, 600)))
        for _ in range(n):
            t += pd.Timedelta(seconds=float(rng.exponential(iat)))
            fwd = max(1, int(rng.poisson(pkts)))
            rows.append({
                " Source IP": src, " Timestamp": t.strftime("%d/%m/%Y %H:%M:%S"),
                " Destination Port": port_fn(), " Flow Duration": float(rng.lognormal(dur, 1.0)),
                " Total Fwd Packets": fwd, " Total Backward Packets": max(0, fwd - int(rng.integers(0, 3))),
                "Total Length of Fwd Packets": fwd * float(rng.lognormal(5, 0.6)),
                " Fwd Header Length": fwd * 20, " Fwd Header Length.1": fwd * 20,
                " Flow IAT Mean": float(rng.lognormal(dur - 2, 1.0)),
                " SYN Flag Count": int(rng.random() < 0.5),
                " Init_Win_bytes_forward": int(rng.choice([8192, 29200, 65535])),
                "Flow Bytes/s": np.inf if rng.random() < 0.01 else float(rng.lognormal(8, 2)),
                " Label": label,
            })

    for h in range(3):
        flows(f"192.168.10.{h + 5}", "BENIGN", 900, 11, 12, lambda: int(rng.choice([80, 443, 53, 22, 8080])), 4)
    flows("172.16.0.1", "DoS Hulk", 500, 9, 6, lambda: 80, 0.05)
    flows("172.16.0.2", "PortScan", 500, 3, 1, lambda: int(rng.integers(1, 1024)), 0.01)
    flows("172.16.0.3", "Web Attack \x96 Brute Force", 400, 10, 9, lambda: 80, 0.5)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="latin-1")


def fake_captures(path: Path, n: int, rng: np.random.Generator) -> None:
    """n captures in one CSV, with source IPs made unique per capture."""
    frames = []
    for rep in range(n):
        fake_flows(path, rng)
        df = pd.read_csv(path, encoding="latin-1")
        df[" Source IP"] = df[" Source IP"] + f"_{rep}"
        frames.append(df)
    pd.concat(frames).to_csv(path, index=False, encoding="latin-1")
