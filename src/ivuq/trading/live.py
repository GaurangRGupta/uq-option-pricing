"""Forward test on live snapshots: turn saved signals into trades and settle them against later snapshots.

A signal row (BUY or SELL, from `scripts/live_signals.py`) becomes the same one-contract, delta-hedged trade
as in the backtest (`trading.backtest.run_backtest`): filled at the snapshot's ask (buy) or bid (sell) on the
signal date (lag 0) or the next snapshot date (lag 1), closed `horizon` snapshot dates later at the opposite
side of that day's quote, hedged with the Black-Scholes delta of the snapshot. A signal is settled once the
exit snapshot exists; until then it is pending. Snapshot dates are the trading days on which a snapshot was
taken, so a missed day shortens the calendar horizon: take one snapshot every trading day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivuq.trading.backtest import BacktestConfig, run_backtest

__all__ = ["signals_to_trades", "settle"]


def signals_to_trades(sig: pd.DataFrame, max_per_day: int = 10) -> pd.DataFrame:
    """BUY/SELL rows of the live signal files in the layout `run_backtest` expects, best `max_per_day` per underlying and date."""
    s = sig[sig["action"].isin(["BUY", "SELL"])].copy()
    s["side"] = np.where(s["action"] == "BUY", 1, -1)
    s["edge"] = np.where(s["side"] == 1, s["theo"] - s["ask"], s["bid"] - s["theo"])
    s = s.sort_values(["symbol", "date", "excess"], ascending=[True, True, False])
    s = s.groupby(["symbol", "date"], sort=False).head(max_per_day)
    return s[["date", "symbol", "expiry", "strike", "scale", "bucket", "side", "edge", "excess", "theo"]].reset_index(drop=True)


def settle(snapshots: pd.DataFrame, signals: pd.DataFrame, horizon: int = 3, lag: int = 0,
           dividends: pd.DataFrame | None = None, max_per_day: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (ledger of settled trades, pending signals). `snapshots` needs symbol, date, expiry, strike, bid, ask, delta, spot."""
    trades_in = signals_to_trades(signals, max_per_day)
    cfg = BacktestConfig(rule="conformal", horizon=horizon, lag=lag, hedge="vendor")
    ledger = run_backtest(snapshots, "theo", cfg, dividends=dividends, signals=trades_in) if len(trades_in) else pd.DataFrame()
    done = set(zip(ledger["date"], ledger["symbol"], ledger["expiry"], ledger["strike"])) if len(ledger) and "strike" in ledger else set()
    mask = [(d, s, e, k) not in done for d, s, e, k in zip(trades_in["date"], trades_in["symbol"], trades_in["expiry"], trades_in["strike"])]
    return ledger, trades_in[mask].reset_index(drop=True)
