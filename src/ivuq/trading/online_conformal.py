"""Rolling, past-only conformal band around a theo, with an adaptive miscoverage level.

Score. For quote i on date t with theo_i, mid_i and error scale c_i
(`calibration.heston_calib.error_scale`):

    s_i = | mid_i - theo_i | / c_i

Calibration set for date t: the scores of the previous `window` trading days of
the same underlying (never date t itself or later), split into Mondrian buckets
(moneyness x tenor). The band half-width for a quote in bucket b is

    h_i = Q_b(1 - alpha_t) * c_i ,    Q_b = conformal_quantile(scores_b, alpha_t)

from `ivuq.uq.conformal.conformal_quantile` (the finite-sample quantile,
k = ceil((n+1)(1-alpha)); +inf when the bucket has fewer than ~1/alpha scores,
which the trade rule reads as "no trade"). If a bucket has fewer than
`min_cal` past scores it falls back to the pooled scores of all buckets.

Adaptive level (Gibbs and Candes, 2021). The scores of consecutive days are
not exchangeable (parameters drift, regimes change), so the fixed-alpha
guarantee does not hold. After date t the level is updated with the realised
miss rate err_t = mean_i 1[ s_i > Q_b(i)(1 - alpha_t) ] over that date's quotes:

    alpha_{t+1} = clip( alpha_t + gamma * (alpha - err_t), 0.01, 0.50 )

which makes the long-run miss rate track alpha without assuming
exchangeability. gamma = 0 gives the fixed-alpha rolling split conformal.
Everything uses information available at the close of date t only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivuq.uq.conformal import conformal_quantile

__all__ = ["bucket_ids", "add_rolling_conformal"]


def bucket_ids(m: np.ndarray, dte: np.ndarray) -> np.ndarray:
    """Moneyness (S/K < 0.95, 0.95 to 1.05, > 1.05) x tenor (DTE < 21, >= 21): ids 0..5."""
    mb = np.digitize(m, [0.95, 1.05])
    tb = (np.asarray(dte) >= 21).astype(int)
    return mb * 2 + tb


def add_rolling_conformal(
    panel: pd.DataFrame,
    theo_col: str,
    scale_col: str = "scale",
    alpha: float = 0.10,
    window: int = 20,
    gamma: float = 0.0,
    min_cal: int = 60,
) -> pd.DataFrame:
    """Returns panel with columns score, bucket, q, alpha_t, n_cal added (suffix-free, for one theo)."""
    out = panel.copy()
    out["score"] = (out["mid"] - out[theo_col]).abs() / out[scale_col]
    out["bucket"] = bucket_ids(out["m"].to_numpy(), out["dte"].to_numpy())
    out["q"] = np.nan
    out["alpha_t"] = np.nan
    out["n_cal"] = 0
    for sym, g in out.groupby("symbol"):
        dates = sorted(g["date"].unique())
        by_date = {d: g.loc[g["date"] == d] for d in dates}
        a_t = alpha
        for k, d in enumerate(dates):
            today = by_date[d]
            past = [by_date[x] for x in dates[max(0, k - window):k]]
            if past:
                pool = pd.concat(past)
                s_all, b_all = pool["score"].to_numpy(), pool["bucket"].to_numpy()
                s_all_valid = s_all[np.isfinite(s_all)]
            q_by_b, n_by_b = {}, {}
            for b in np.unique(today["bucket"]):
                if past:
                    s_b = s_all[(b_all == b) & np.isfinite(s_all)]
                    s_use = s_b if len(s_b) >= min_cal else s_all_valid
                else:
                    s_use = np.array([])
                n_by_b[b] = len(s_use)
                q_by_b[b] = conformal_quantile(s_use, a_t) if len(s_use) >= min_cal else np.inf
            q = today["bucket"].map(q_by_b).to_numpy()
            out.loc[today.index, "q"] = q
            out.loc[today.index, "alpha_t"] = a_t
            out.loc[today.index, "n_cal"] = today["bucket"].map(n_by_b).to_numpy()
            if gamma > 0 and np.isfinite(q).any():
                s_today = today["score"].to_numpy()
                ok = np.isfinite(q) & np.isfinite(s_today)
                err = float(np.mean(s_today[ok] > q[ok]))
                a_t = float(np.clip(a_t + gamma * (alpha - err), 0.01, 0.50))
    return out
