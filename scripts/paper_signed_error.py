"""Diagnostic: signed point error (PINN - reference) per model, seed and bucket, on the P1/P2 pools.

    python scripts/paper_signed_error.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ivuq.uq import moneyness_tenor_buckets

from paper_common import BUCKET_NAMES, PAPER
from paper_models import K, predictors

COL = {"n1": "u_eu_bs", "n0": "u_eu_bs", "n2": "u_am_crr500"}

torch.set_num_threads(4)
rows = []
for model, col in COL.items():
    for s in range(10):
        pool = pd.read_csv(PAPER / "pools" / f"pool_seed{s}.csv")
        m, tau, u = pool.m.to_numpy(), pool.tau.to_numpy(), pool[col].to_numpy()
        e = K * (predictors(model, s)["pde"].predict(m, tau) - u)
        b = moneyness_tenor_buckets(m, tau)
        rows.append({"model": model, "seed": s, "bucket": "all", "mean_signed_err_usd": e.mean(), "frac_over": (e > 0).mean()})
        for k, name in BUCKET_NAMES.items():
            rows.append({"model": model, "seed": s, "bucket": name, "mean_signed_err_usd": e[b == k].mean(), "frac_over": (e[b == k] > 0).mean()})
df = pd.DataFrame(rows)
df.to_csv(PAPER / "p1p2" / "p1p2_signed_error_raw.csv", index=False)
summ = df.groupby(["model", "bucket"]).agg(mean_signed_err_usd_mean=("mean_signed_err_usd", "mean"),
                                           mean_signed_err_usd_std=("mean_signed_err_usd", "std"),
                                           frac_over_mean=("frac_over", "mean")).reset_index()
summ.to_csv(PAPER / "p1p2" / "p1p2_signed_error_summary.csv", index=False)
print(summ.round(3).to_string(index=False))
