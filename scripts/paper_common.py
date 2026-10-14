"""Shared helpers for the paper experiment scripts (P1-P4)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ivuq.uq import empirical_coverage, mean_width, moneyness_tenor_buckets

ROOT = Path(__file__).resolve().parents[1]
# v1 outputs by default; set IVUQ_PAPER_DIR (e.g. results/uq/paper_v2) to read/write another run.
PAPER = Path(os.environ["IVUQ_PAPER_DIR"]).resolve() if os.environ.get("IVUQ_PAPER_DIR") else ROOT / "results" / "uq" / "paper"
BUCKET_NAMES = {
    0: "m<0.9,tau<0.4",
    1: "m<0.9,tau>=0.4",
    2: "0.9-1.1,tau<0.4",
    3: "0.9-1.1,tau>=0.4",
    4: "m>1.1,tau<0.4",
    5: "m>1.1,tau>=0.4",
}


def interval_metrics(lo, hi, u, m, tau, scale: float = 1.0) -> dict:
    """Marginal and per-bucket coverage/width. `scale` converts u units to dollars."""
    lo, hi, u = (np.asarray(a, dtype=np.float64) for a in (lo, hi, u))
    finite = np.isfinite(lo) & np.isfinite(hi)
    out = {
        "coverage": empirical_coverage(lo, hi, u),
        "mean_width": scale * mean_width(lo[finite], hi[finite]) if finite.any() else np.nan,
        "median_width": scale * float(np.median(hi[finite] - lo[finite])) if finite.any() else np.nan,
        "n_infinite": int((~finite).sum()),
    }
    b = moneyness_tenor_buckets(m, tau)
    covs = []
    for k, name in BUCKET_NAMES.items():
        sel = b == k
        if sel.any():
            c = empirical_coverage(lo[sel], hi[sel], u[sel])
            w = scale * mean_width(lo[sel & finite], hi[sel & finite]) if (sel & finite).any() else np.nan
            covs.append(c)
        else:
            c = w = np.nan
        out[f"cov_b{k}"] = c
        out[f"width_b{k}"] = w
        out[f"n_b{k}"] = int(sel.sum())
    out["worst_bucket_cov"] = float(np.nanmin(covs)) if covs else np.nan
    return out


def summarise(df, keys, metrics, unit_col="seed"):
    """Mean over replicates within each unit (seed), then mean/std across units."""
    import pandas as pd

    per_unit = df.groupby(keys + [unit_col], as_index=False)[metrics].mean()
    g = per_unit.groupby(keys)
    mean = g[metrics].mean().add_suffix("_mean")
    std = g[metrics].std(ddof=1).add_suffix("_std")
    n = g[unit_col].nunique().rename("n_seeds")
    return pd.concat([n, mean, std], axis=1).reset_index()
