"""Tables behind the report figures: the ablation of theo x band rule, cost sensitivity, regimes, failures.

All numbers come from the executed-trade files written by `scripts/run_heston_study.py` (rolling past-only
conformal band, horizon 3 trading days, hedge with the vendor delta) and are split at `SPLIT` into a
development period and a test period. The trade rules are not re-tuned on the test period: horizon, alpha,
window and gamma were fixed before looking at it (see DEVLOG). Every P&L is in dollars per one-contract trade.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SPLIT", "THEO_NAMES", "load_trades", "period_stats", "ablation", "cost_grid", "monthly", "block_bootstrap_ci"]

SPLIT = pd.Timestamp("2026-05-01")
THEO_NAMES = {"pinn": "Heston PINN (single)", "cpl1000": "Heston PINN (coupled)", "fd": "Heston exact solver",
              "flat": "Black-Scholes flat vol", "smile": "Black-Scholes smile", "h_slow": "Heston PINN (slow params)"}
_PAT = re.compile(r"trades_(\w+?)_(h_full|h_slow|flat|smile)_(conformal|spread)_h3_l(\d)\.csv")


def load_trades(results: str | Path = "results/heston_trading") -> pd.DataFrame:
    """One frame of executed trades (status ok) with columns model (the theo variant name), rule, lag; the trade file's own `theo` column is the theo price. The BS rows are taken from the
    tag `pinn` only (they do not depend on the network)."""
    parts = []
    for f in sorted(Path(results).glob("trades_*_h3_l*.csv")):
        m = _PAT.match(f.name)
        if not m:
            continue
        tag, var, rule, lag = m.groups()
        if var in ("flat", "smile") and tag != "pinn":
            continue
        theo = var if var in ("flat", "smile") else ("h_slow" if var == "h_slow" else tag)
        t = pd.read_csv(f, parse_dates=["date", "entry_date", "exit_date", "expiry"])
        t = t[t["status"] == "ok"].copy()
        t["model"], t["rule"], t["lag"] = theo, rule, int(lag)
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d["period"] = np.where(d["date"] < SPLIT, "development", "test")
    return d


def block_bootstrap_ci(daily: np.ndarray, n_boot: int = 2000, block: int = 5, seed: int = 0) -> tuple[float, float]:
    n = len(daily)
    if n < 8:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, max(n - block + 1, 1), nb)
        means[i] = np.concatenate([daily[a:a + block] for a in s])[:n].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def period_stats(g: pd.DataFrame) -> dict:
    if not len(g):
        return dict(n=0, mean=np.nan, total=0.0, hit=np.nan, sharpe=np.nan, ci_lo=np.nan, ci_hi=np.nan, days=0)
    daily = g.groupby("exit_date")["net_pnl"].sum().sort_index()
    sd = daily.std(ddof=1)
    lo, hi = block_bootstrap_ci(daily.to_numpy())
    return dict(n=len(g), mean=float(g["net_pnl"].mean()), total=float(g["net_pnl"].sum()), hit=float((g["net_pnl"] > 0).mean()),
                sharpe=float(daily.mean() / sd * np.sqrt(252)) if len(daily) > 2 and sd > 0 else np.nan,
                ci_lo=lo, ci_hi=hi, days=len(daily))


def ablation(trades: pd.DataFrame) -> pd.DataFrame:
    """Per (model, rule, lag, period): trades, mean and total net P&L, hit rate, Sharpe, 95% CI of mean daily P&L."""
    rows = []
    for (model, rule, lag, per), g in trades.groupby(["model", "rule", "lag", "period"]):
        rows.append(dict(model=model, rule=rule, lag=lag, period=per, **period_stats(g)))
    return pd.DataFrame(rows)


def cost_grid(trades: pd.DataFrame, commissions=(0.0, 0.65, 1.3, 2.6), hedge_mult=(0.0, 1.0, 2.0, 4.0)) -> pd.DataFrame:
    """Total net P&L as the per-side commission ($) and the stock-trading cost (multiple of the 1bp assumption) vary."""
    rows = []
    for (model, rule, lag, per), g in trades.groupby(["model", "rule", "lag", "period"]):
        gross = (g["opt_pnl"] + g["hedge_pnl"]).sum()
        for c in commissions:
            for h in hedge_mult:
                tot = gross - h * g["hedge_cost"].sum() - 2 * c * len(g)
                rows.append(dict(model=model, rule=rule, lag=lag, period=per, commission=c, hedge_mult=h, total=tot,
                                 mean=tot / len(g) if len(g) else np.nan, n=len(g)))
    return pd.DataFrame(rows)


def monthly(trades: pd.DataFrame) -> pd.DataFrame:
    t = trades.copy()
    t["month"] = t["exit_date"].dt.to_period("M").astype(str)
    return t.groupby(["model", "rule", "lag", "month"]).agg(n=("net_pnl", "size"), total=("net_pnl", "sum"),
                                                          mean=("net_pnl", "mean")).reset_index()
