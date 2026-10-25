"""Random-signal control for the backtest.

For every (symbol, date) where a strategy placed n trades, draw n quotes
uniformly from that date's quotes and a random side. The control keeps the
strategy's trade count per day, the contract mix of the universe, the horizon
and all costs, and removes the information in the theo. A strategy only has
evidence of skill if it beats the distribution of this control's P&L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["random_signals"]


def random_signals(panel: pd.DataFrame, template: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """`template`: the signal frame of the real strategy (columns symbol, date). Returns a signal frame
    in the format `run_backtest(signals=...)` expects."""
    rng = np.random.default_rng(seed)
    counts = template.groupby(["symbol", "date"]).size()
    parts = []
    for (sym, d), n in counts.items():
        g = panel[(panel["symbol"] == sym) & (panel["date"] == d)]
        if not len(g):
            continue
        pick = g.sample(n=min(n, len(g)), random_state=int(rng.integers(0, 2**31 - 1)))
        s = pick[["date", "symbol", "expiry", "strike", "scale", "bucket"]].copy()
        s["side"] = rng.choice([-1, 1], size=len(s))
        s["edge"] = 0.0
        s["excess"] = 0.0
        s["theo"] = pick["mid"].to_numpy()
        parts.append(s)
    return pd.concat(parts) if parts else pd.DataFrame()
