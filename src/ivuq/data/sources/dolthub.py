"""DoltHub `post-no-preference` options, stocks and rates -> clean American put quotes.

Source. Three public DoltHub repos by the same maintainer (CC BY-SA 4.0):
  options.option_chain  end-of-day bid/ask, vendor vol and Greeks, per contract
  stocks.ohlcv          end-of-day close (not split-adjusted, matching the strikes)
  stocks.dividend       cash dividends with ex-dates
  rates.us_treasury     constant-maturity yields in percent, 1m to 30y

The SQL API is used one date at a time. A single-date query is fast; a
symbol-range query over the 120M-row option_chain times out (see
data/dolthub/ASSESSMENT.md). `pull_raw` therefore loops over weekdays, keeps
one CSV per date and skips dates already on disk, so it can be stopped and
resumed, and rerun daily to extend the history.

Derived inputs (these definitions are what the pricer and the backtest use):
  spot      S      = ohlcv close on the quote date
  mid       P      = (bid + ask) / 2
  tau              = (expiry - quote_date) / 365 in years
  moneyness m      = S / K
  rate      r      = ln(1 + y(tau) / 100), y linear in tenor between the
                     1m, 2m, 3m, 6m, 1y Treasury points, flat outside
  yield     q      = (cash dividends with ex-date in the 365 days up to the
                     quote date) / S
Quotes are taken to be end-of-day of the stamped date: the earlier assessment
showed put-call parity spots matching that date's close, an inference and not
documented by the maintainer.

Never committed: the pulled CSVs (licence ShareAlike and size). Attribution is
in data/dolthub/ASSESSMENT.md.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

__all__ = [
    "API",
    "DEFAULT_SYMBOLS",
    "query",
    "pull_raw",
    "load_raw",
    "treasury_rate",
    "trailing_dividend_yield",
    "clean_puts",
]

API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/{repo}/master"
DEFAULT_SYMBOLS = ("AAPL", "NVDA", "SPY")

_TENOR_YEARS = {"1_month": 1 / 12, "2_month": 2 / 12, "3_month": 3 / 12, "6_month": 0.5, "1_year": 1.0}


def query(repo: str, sql: str, retries: int = 5, pause: float = 2.0) -> list[dict]:
    """Run one SQL query against a DoltHub repo and return the rows.

    Retries on transport errors and on 'context deadline exceeded'. Raises
    RuntimeError if the query keeps failing, so a partial pull never looks
    like an empty (holiday) date."""
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(API.format(repo=repo), params={"q": sql}, timeout=120)
            j = r.json()
            if j.get("query_execution_status") == "Success":
                return j["rows"]
            last = str(j.get("query_execution_message"))
        except (requests.RequestException, ValueError) as exc:
            last = repr(exc)
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"DoltHub query failed after {retries} tries: {sql!r}: {last}")


def _in(symbols) -> str:
    return ",".join(f"'{s}'" for s in symbols)


def pull_raw(
    start: str,
    end: str,
    out_dir: str | Path,
    symbols=DEFAULT_SYMBOLS,
    verbose: bool = True,
) -> Path:
    """Pull raw chain, closes, dividends and rates for [start, end] into
    `out_dir`. Layout: chain/<date>.csv, ohlcv/<date>.csv (only for dates
    where the market was open), dividend.csv, us_treasury.csv.

    A weekday with no ohlcv rows is a market holiday: its chain is not pulled
    (DoltHub stamps holidays with copies of the previous day's quotes)."""
    out = Path(out_dir)
    (out / "chain").mkdir(parents=True, exist_ok=True)
    (out / "ohlcv").mkdir(parents=True, exist_ok=True)
    sym = _in(symbols)

    for d in pd.bdate_range(start, end):
        ds = d.strftime("%Y-%m-%d")
        f_px, f_ch = out / "ohlcv" / f"{ds}.csv", out / "chain" / f"{ds}.csv"
        if f_px.exists() and f_ch.exists():
            continue
        px = query("stocks", f"select * from ohlcv where date='{ds}' and act_symbol in ({sym})")
        if not px:
            pd.DataFrame({"date": []}).to_csv(f_px, index=False)
            pd.DataFrame({"date": []}).to_csv(f_ch, index=False)
            if verbose:
                print(f"{ds} holiday/no data", flush=True)
            continue
        ch = query("options", f"select * from option_chain where date='{ds}' and act_symbol in ({sym})")
        pd.DataFrame(px).to_csv(f_px, index=False)
        pd.DataFrame(ch).to_csv(f_ch, index=False)
        if verbose:
            print(f"{ds} px={len(px)} chain={len(ch)}", flush=True)

    # dividends reach back a year before `start` so the trailing yield is defined on day one
    back = (pd.Timestamp(start) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    div = query("stocks", f"select * from dividend where act_symbol in ({sym}) and ex_date>='{back}'")
    pd.DataFrame(div).to_csv(out / "dividend.csv", index=False)
    ust = query("rates", f"select * from us_treasury where date>='{back}'")
    pd.DataFrame(ust).to_csv(out / "us_treasury.csv", index=False)
    return out


def load_raw(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read what `pull_raw` wrote into typed frames."""
    raw = Path(raw_dir)

    def _cat(sub: str) -> pd.DataFrame:
        parts = [pd.read_csv(f) for f in sorted((raw / sub).glob("*.csv")) if f.stat().st_size > 8]
        parts = [p for p in parts if len(p)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    chain, px = _cat("chain"), _cat("ohlcv")
    for c in ("date", "expiration"):
        chain[c] = pd.to_datetime(chain[c])
    px["date"] = pd.to_datetime(px["date"])
    div = pd.read_csv(raw / "dividend.csv", parse_dates=["ex_date"])
    ust = pd.read_csv(raw / "us_treasury.csv", parse_dates=["date"])
    return {"chain": chain, "ohlcv": px, "dividend": div, "treasury": ust}


def treasury_rate(ust: pd.DataFrame, dates: pd.Series, tau: np.ndarray) -> np.ndarray:
    """Continuously compounded risk-free rate for each (date, tau) pair.

    r = ln(1 + y/100) with y interpolated linearly in tenor on the
    1m/2m/3m/6m/1y points and held flat outside them. Dates without a
    Treasury print (bond-market closures) take the previous available print."""
    cols = list(_TENOR_YEARS)
    t = np.array([_TENOR_YEARS[c] for c in cols])
    curve = ust.sort_values("date").set_index("date")[cols].astype(float)
    curve = curve.reindex(curve.index.union(pd.DatetimeIndex(dates.unique()))).sort_index().ffill()
    y = curve.loc[pd.DatetimeIndex(dates)].to_numpy()
    tau = np.asarray(tau, dtype=float)
    out = np.empty(len(tau))
    for i in range(len(tau)):
        out[i] = np.interp(tau[i], t, y[i])
    return np.log1p(out / 100.0)


def trailing_dividend_yield(div: pd.DataFrame, symbol: str, date: pd.Timestamp, spot: float) -> float:
    """Sum of cash dividends with ex-date in (date - 365d, date], divided by spot."""
    d = div[(div["act_symbol"] == symbol) & (div["ex_date"] <= date) & (div["ex_date"] > date - pd.Timedelta(days=365))]
    return float(d["amount"].astype(float).sum() / spot)


def clean_puts(
    raw: dict[str, pd.DataFrame],
    dte_min: int = 7,
    dte_max: int = 70,
    m_lo: float = 0.80,
    m_hi: float = 1.25,
) -> pd.DataFrame:
    """Tidy American-put quotes, one row per (date, symbol, expiry, strike).

    Drops, in order: calls; dates with no same-day close (holidays); bid <= 0;
    ask < bid; locked quotes (bid == ask); vendor vol <= 0; DTE outside
    [dte_min, dte_max]; moneyness S/K outside [m_lo, m_hi]. Also drops quotes
    priced below intrinsic value minus half a spread, which cannot be a
    sensible American put quote. `drop_log` (DataFrame.attrs) holds the count
    removed at each step."""
    ch = raw["chain"].copy()
    px = raw["ohlcv"][["date", "act_symbol", "close"]].rename(columns={"close": "spot"})
    log: dict[str, int] = {"rows_in": len(ch)}

    for c in ("strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"):
        ch[c] = ch[c].astype(float)
    px["spot"] = px["spot"].astype(float)

    ch = ch[ch["call_put"] == "Put"]
    log["puts"] = len(ch)
    ch = ch.merge(px, on=["date", "act_symbol"], how="inner")
    log["with_same_day_close"] = len(ch)
    ch = ch[ch["bid"] > 0]
    log["bid_positive"] = len(ch)
    ch = ch[ch["ask"] >= ch["bid"]]
    ch = ch[ch["ask"] > ch["bid"]]
    log["not_crossed_or_locked"] = len(ch)
    ch = ch[ch["vol"] > 0]
    log["vol_positive"] = len(ch)

    ch["dte"] = (ch["expiration"] - ch["date"]).dt.days
    ch = ch[(ch["dte"] >= dte_min) & (ch["dte"] <= dte_max)]
    log["dte_window"] = len(ch)
    ch["m"] = ch["spot"] / ch["strike"]
    ch = ch[(ch["m"] >= m_lo) & (ch["m"] <= m_hi)]
    log["moneyness_window"] = len(ch)

    ch["mid"] = 0.5 * (ch["bid"] + ch["ask"])
    ch["spread"] = ch["ask"] - ch["bid"]
    intrinsic = np.maximum(ch["strike"] - ch["spot"], 0.0)
    ch = ch[ch["mid"] + 0.5 * ch["spread"] >= intrinsic]
    log["above_intrinsic"] = len(ch)

    ch["tau"] = ch["dte"] / 365.0
    ch["r"] = treasury_rate(raw["treasury"], ch["date"], ch["tau"].to_numpy())
    q = {}
    for (s, d), g in ch.groupby(["act_symbol", "date"]):
        q[(s, d)] = trailing_dividend_yield(raw["dividend"], s, d, float(g["spot"].iloc[0]))
    ch["q"] = [q[(s, d)] for s, d in zip(ch["act_symbol"], ch["date"])]

    ch = ch.rename(columns={"act_symbol": "symbol", "expiration": "expiry", "vol": "vendor_iv"})
    cols = ["date", "symbol", "expiry", "strike", "spot", "bid", "ask", "mid", "spread",
            "dte", "tau", "m", "r", "q", "vendor_iv", "delta", "gamma", "theta", "vega", "rho"]
    out = ch[cols].sort_values(["date", "symbol", "expiry", "strike"]).reset_index(drop=True)
    out.attrs["drop_log"] = log
    return out
