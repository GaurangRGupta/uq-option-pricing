"""Shared loading of the quote panel with every theo variant on a common set of quotes.

Used by the study, out-of-sample and signal scripts so they all see the same data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.calibration.heston_calib import error_scale
from ivuq.trading.online_conformal import bucket_ids

__all__ = ["KEY", "SLICES", "load_panel"]

KEY = ["date", "symbol", "expiry", "strike"]
SLICES = Path("data/dolthub/slices")


def load_panel(tag: str, symbols, with_bs: bool = True) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Quotes joined with the Heston theos of `tag` (full and slow modes, whichever exist) and, if present,
    the BS flat and smile theos, restricted to quotes that have every variant. Returns (panel, [(name, column)])."""
    d = pd.read_csv(SLICES / "puts_clean.csv", parse_dates=["date", "expiry"])
    d = d[d["symbol"].isin(symbols)].reset_index(drop=True)
    d["scale"] = error_scale(d)
    variants = []
    for mode in ("full", "slow"):
        parts = []
        for s in symbols:
            f = SLICES / f"theo_{s}_{tag}_{mode}.csv"
            if f.exists():
                t = pd.read_csv(f, parse_dates=["date", "expiry"])
                t = t[KEY + ["theo", "model_delta", "burn_in"]].rename(
                    columns={"theo": f"theo_h_{mode}", "model_delta": f"mdelta_{mode}", "burn_in": f"burn_{mode}"})
                parts.append(t)
        if parts:
            t = pd.concat(parts)
            d = d.merge(t, on=KEY, how="left")
            if mode == "slow":
                d.loc[d["burn_slow"].fillna(True).astype(bool), "theo_h_slow"] = np.nan
            variants.append(f"h_{mode}")
    f = SLICES / "bs_theos.csv"
    if with_bs and f.exists():
        b = pd.read_csv(f, parse_dates=["date", "expiry"])
        d = d.merge(b[KEY + ["theo_flat", "theo_smile"]], on=KEY, how="left")
        variants += ["flat", "smile"]
    cols = {"h_full": "theo_h_full", "h_slow": "theo_h_slow", "flat": "theo_flat", "smile": "theo_smile"}
    common = np.ones(len(d), dtype=bool)
    for v in variants:
        common &= d[cols[v]].notna().to_numpy()
    d = d[common].reset_index(drop=True)
    d["bucket"] = bucket_ids(d["m"].to_numpy(), d["dte"].to_numpy())
    return d, [(v, cols[v]) for v in variants]


