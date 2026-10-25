"""Live put-chain snapshots from Yahoo Finance (yfinance), in the same schema as `dolthub.clean_puts`.

Yahoo serves only the current chain (no history), so a history is built by taking one snapshot per
trading day after the close and saving it under data/live/snapshots/. The snapshot carries real
bid/ask, which is what the take-the-quote rules need; Yahoo's own implied vol and Greeks are not
used because their provenance is undocumented.

Derived inputs follow the DoltHub loader so the model sees the same definitions:
  spot  S   = the spot implied by put-call parity of the near-the-money quotes of the near expiries
              (`parity_spot`); `spot_close` is Yahoo's last close. Option quotes keep updating after the
              4pm close (SPY options trade to 4:15pm), so the close is stale against them: on 2026-10-23
              it was about $1.1 below the parity spot for SPY, enough to move an at-the-money put by half a dollar.
  mid   P   = (bid + ask) / 2
  tau       = (expiry - quote_date) / 365
  m         = S / K
  r         = ln(1 + y(tau) / 100) from the Treasury curve of the DoltHub rates repo (latest print held flat)
  q         = cash dividends with ex-date in the 365 days up to the quote date, over S
  vendor_iv = Black-Scholes implied vol of the mid (European inversion, with q), used only for the
              starting point of the calibration and the error scale; delta, gamma, vega, theta, rho
              are the Black-Scholes Greeks at that vol (the hedge ratio the backtest calls "vendor").

Raw market data is never committed (gitignored).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.data.sources.dolthub import trailing_dividend_yield, treasury_rate
from ivuq.pricing import black_scholes as bs

__all__ = ["SNAP_DIR", "fetch_puts", "parity_spot", "clean_snapshot", "take_snapshot", "load_snapshots"]

SNAP_DIR = Path("data/live/snapshots")


def _spot_and_date(tk) -> tuple[float, pd.Timestamp]:
    h = tk.history(period="5d", auto_adjust=False)
    if not len(h):
        raise RuntimeError(f"no price history for {tk.ticker}")
    d = pd.Timestamp(h.index[-1]).tz_localize(None).normalize()
    return float(h["Close"].iloc[-1]), d


def fetch_puts(symbol: str, dte_min: int = 7, dte_max: int = 70) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw put rows for expiries in the DTE window, with spot and quote date, the dividend history and the
    raw call rows (used only to infer the spot from put-call parity)."""
    import yfinance as yf

    tk = yf.Ticker(symbol)
    spot, qdate = _spot_and_date(tk)
    rows, calls = [], []
    for ex in tk.options:
        dte = (pd.Timestamp(ex) - qdate).days
        if dte < dte_min or dte > dte_max:
            continue
        ch = tk.option_chain(ex)
        p = ch.puts.copy()
        p["expiry"] = pd.Timestamp(ex)
        rows.append(p)
        c = ch.calls.copy()
        c["expiry"] = pd.Timestamp(ex)
        calls.append(c)
    if not rows:
        raise RuntimeError(f"no expiries in the window for {symbol}")
    raw = pd.concat(rows, ignore_index=True)
    raw["symbol"], raw["spot_close"], raw["date"] = symbol, spot, qdate
    calls = pd.concat(calls, ignore_index=True)
    calls["symbol"], calls["date"] = symbol, qdate
    div = tk.dividends.reset_index()
    div.columns = ["ex_date", "amount"]
    div["ex_date"] = pd.to_datetime(div["ex_date"]).dt.tz_localize(None).dt.normalize()
    div["act_symbol"] = symbol
    return raw, div, calls


def parity_spot(puts: pd.DataFrame, calls: pd.DataFrame, spot_close: float, r_of_tau, q: float, qdate: pd.Timestamp,
                band: float = 0.04, dte_lo: int = 7, dte_hi: int = 45) -> float:
    """Spot implied by European put-call parity, F = K + (C - P) e^{rT}, S = F e^{-(r-q)T}, taken as the median
    over near-the-money strikes (|K/S - 1| < band) of the expiries with DTE in [dte_lo, dte_hi], then the median
    over expiries. American early exercise adds a small premium to near-the-money puts and calls; it biases the
    estimate by cents, far below the quote-timing gap this corrects. Falls back to the close if too few pairs."""
    m = puts.merge(calls, on=["expiry", "strike"], suffixes=("_p", "_c"))
    m = m[(m["bid_p"] > 0) & (m["ask_p"] > m["bid_p"]) & (m["bid_c"] > 0) & (m["ask_c"] > m["bid_c"])]
    m = m[(m["strike"] / spot_close - 1).abs() < band]
    est = []
    for ex, g in m.groupby("expiry"):
        dte = (ex - qdate).days
        if dte < dte_lo or dte > dte_hi or len(g) < 3:
            continue
        T = dte / 365.0
        r = float(r_of_tau(T))
        C, P = 0.5 * (g["bid_c"] + g["ask_c"]), 0.5 * (g["bid_p"] + g["ask_p"])
        F = g["strike"] + (C - P) * np.exp(r * T)
        est.append(float(np.median(F * np.exp(-(r - q) * T))))
    return float(np.median(est)) if est else float(spot_close)


def clean_snapshot(raw: pd.DataFrame, div: pd.DataFrame, treasury: pd.DataFrame, m_lo: float = 0.80, m_hi: float = 1.25,
                   dte_min: int = 7, dte_max: int = 70) -> pd.DataFrame:
    """Same filters as `dolthub.clean_puts`: bid > 0, ask > bid, DTE and moneyness windows, above intrinsic."""
    d = raw.copy()
    if "spot" not in d:
        d["spot"] = d["spot_close"]
    for c in ("strike", "bid", "ask"):
        d[c] = d[c].astype(float)
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])].copy()
    d["dte"] = (d["expiry"] - d["date"]).dt.days
    d = d[(d["dte"] >= dte_min) & (d["dte"] <= dte_max)]
    d["m"] = d["spot"] / d["strike"]
    d = d[(d["m"] >= m_lo) & (d["m"] <= m_hi)].copy()
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    d["spread"] = d["ask"] - d["bid"]
    d = d[d["mid"] + 0.5 * d["spread"] >= np.maximum(d["strike"] - d["spot"], 0.0)].copy()
    d["tau"] = d["dte"] / 365.0
    d["r"] = treasury_rate(treasury, d["date"], d["tau"].to_numpy())
    q = {}
    for (s, dt), g in d.groupby(["symbol", "date"]):
        q[(s, dt)] = trailing_dividend_yield(div, s, dt, float(g["spot"].iloc[0]))
    d["q"] = [q[(s, dt)] for s, dt in zip(d["symbol"], d["date"])]

    iv = np.full(len(d), np.nan)
    for i, t in enumerate(d.itertuples()):
        try:
            iv[i] = bs.implied_vol(t.mid, t.spot, t.strike, t.tau, t.r, t.q, "put")
        except ValueError:
            pass
    d["vendor_iv"] = iv
    d = d[np.isfinite(d["vendor_iv"]) & (d["vendor_iv"] > 0)].copy()
    S, K, T, r, qq, s = (d[c].to_numpy(float) for c in ("spot", "strike", "tau", "r", "q", "vendor_iv"))
    d["delta"] = [bs.delta(*a, "put") for a in zip(S, K, T, r, qq, s)]
    d["gamma"] = [bs.gamma(*a) for a in zip(S, K, T, r, qq, s)]
    d["theta"] = [bs.theta(*a, "put") for a in zip(S, K, T, r, qq, s)]
    d["vega"] = [bs.vega(*a) for a in zip(S, K, T, r, qq, s)]
    d["rho"] = [bs.rho(*a, "put") for a in zip(S, K, T, r, qq, s)]
    d = d.rename(columns={"impliedVolatility": "yahoo_iv", "lastPrice": "last", "openInterest": "open_interest"})
    cols = ["date", "symbol", "expiry", "strike", "spot", "spot_close", "bid", "ask", "mid", "spread", "dte", "tau", "m", "r", "q",
            "vendor_iv", "delta", "gamma", "theta", "vega", "rho", "last", "volume", "open_interest", "yahoo_iv"]
    return d[cols].sort_values(["date", "symbol", "expiry", "strike"]).reset_index(drop=True)


def take_snapshot(symbols=("SPY", "AAPL", "NVDA"), treasury_csv: str | Path = "data/dolthub/slices/raw/us_treasury.csv",
                  out_dir: str | Path = SNAP_DIR, stamp: str | None = None) -> pd.DataFrame:
    """Fetch, clean and save one snapshot per symbol: <out_dir>/<quote_date>_<symbol>.csv (overwritten if retaken
    the same day, so the file is the last snapshot of that date). Returns the cleaned frame."""
    ust = pd.read_csv(treasury_csv, parse_dates=["date"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for s in symbols:
        raw, div, calls = fetch_puts(s)
        qd = raw["date"].iloc[0]
        q = trailing_dividend_yield(div, s, qd, float(raw["spot_close"].iloc[0]))
        r_of = lambda T, qd=qd: treasury_rate(ust, pd.Series([qd]), np.array([T]))[0]
        raw["spot"] = parity_spot(raw, calls, float(raw["spot_close"].iloc[0]), r_of, q, qd)
        c = clean_snapshot(raw, div, ust)
        c["snapshot_time"] = stamp or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        c.to_csv(out / f"{c['date'].iloc[0].date()}_{s}.csv", index=False)
        parts.append(c)
    return pd.concat(parts, ignore_index=True)


def load_snapshots(snap_dir: str | Path = SNAP_DIR) -> pd.DataFrame:
    fs = sorted(Path(snap_dir).glob("*.csv"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, parse_dates=["date", "expiry"]) for f in fs], ignore_index=True)
