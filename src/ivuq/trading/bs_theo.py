"""Black-Scholes benchmark theos for American puts (the "constant volatility" baselines).

Both fit to the same day's quotes as the Heston calibration, so the comparison
is the dynamics (constant vs stochastic volatility), not the information used.

  flat   one sigma per (date, symbol, expiry): the implied vol (American, CRR)
         of the put nearest the money. Every strike of that expiry is priced at it.
  smile  per (date, symbol, expiry): own American implied vols regressed on
         log-moneyness, sigma(k) = a + b k + c k^2, weights 1/half-spread
         (clipped); theo is the American CRR price at the fitted sigma(k).
         This is the standard desk "relative value against the fitted smile".

CRR uses 200 steps (accurate to about a cent per $100 strike here); implied
vols are solved with `ivuq.pricing.iv_solver.implied_vol` (Brent on the tree).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivuq.pricing.binomial import crr_price
from ivuq.pricing.iv_solver import implied_vol

__all__ = ["bs_theos_for_day"]

STEPS = 200


def _iv(row) -> float:
    try:
        return implied_vol(row.mid, row.spot, row.strike, row.tau, row.r, row.q, "put", True, steps=STEPS)
    except (ValueError, RuntimeError):
        return np.nan


def bs_theos_for_day(df: pd.DataFrame) -> pd.DataFrame:
    """df: one symbol-date. Returns df with columns own_iv, theo_flat, theo_smile."""
    out = df.copy()
    out["own_iv"] = [_iv(r) for r in out.itertuples()]
    out["theo_flat"] = np.nan
    out["theo_smile"] = np.nan
    for _, g in out.groupby("expiry"):
        ok = g[g["own_iv"].notna()]
        if len(ok) < 4:
            continue
        atm = ok.loc[(ok["m"] - 1.0).abs().idxmin()]
        for i, r in g.iterrows():
            out.at[i, "theo_flat"] = crr_price(r.spot, r.strike, r.tau, r.r, r.q, atm["own_iv"], "put", True, STEPS)
        k = np.log(ok["strike"].to_numpy() / ok["spot"].to_numpy())
        w = 1.0 / np.clip(0.5 * ok["spread"].to_numpy(), 0.02, None)
        coef = np.polyfit(k, ok["own_iv"].to_numpy(), 2, w=w)
        for i, r in g.iterrows():
            kk = np.log(r.strike / r.spot)
            sig = float(np.clip(np.polyval(coef, kk), 0.03, 3.0))
            out.at[i, "theo_smile"] = crr_price(r.spot, r.strike, r.tau, r.r, r.q, sig, "put", True, STEPS)
    return out
