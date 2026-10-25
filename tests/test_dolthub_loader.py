"""Cleaning rules and derived inputs of the DoltHub loader, on a tiny hand-made fixture
(no network)."""

import numpy as np
import pandas as pd

from ivuq.data.sources.dolthub import clean_puts, trailing_dividend_yield, treasury_rate


def _chain(rows):
    cols = ["date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"]); df["expiration"] = pd.to_datetime(df["expiration"])
    return df


def _raw(rows, closes):
    ust = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-05"]), "1_month": [4.0, 4.0], "2_month": [4.1, 4.1],
                        "3_month": [4.2, 4.2], "6_month": [4.3, 4.3], "1_year": [4.4, 4.4]})
    div = pd.DataFrame({"act_symbol": ["SPY"], "ex_date": pd.to_datetime(["2025-12-19"]), "amount": [1.8]})
    px = pd.DataFrame(closes, columns=["date", "act_symbol", "close"]); px["date"] = pd.to_datetime(px["date"])
    return {"chain": _chain(rows), "ohlcv": px, "dividend": div, "treasury": ust}


Z = [0.0] * 5


def test_filters_and_derived_columns():
    good = ["2026-01-05", "SPY", "2026-02-06", 90.0, "Put", 0.40, 0.50, 0.30] + Z
    rows = [
        good,
        ["2026-01-05", "SPY", "2026-02-06", 95.0, "Call", 6.0, 6.2, 0.3] + Z,    # call
        ["2026-01-05", "SPY", "2026-02-06", 91.0, "Put", 0.0, 0.5, 0.3] + Z,     # zero bid
        ["2026-01-05", "SPY", "2026-02-06", 92.0, "Put", 0.6, 0.6, 0.3] + Z,     # locked
        ["2026-01-05", "SPY", "2026-02-06", 93.0, "Put", 0.7, 0.9, 0.0] + Z,     # vol <= 0
        ["2026-01-05", "SPY", "2026-01-08", 90.0, "Put", 0.1, 0.2, 0.3] + Z,     # DTE 3
        ["2026-01-05", "SPY", "2026-02-06", 60.0, "Put", 0.1, 0.2, 0.3] + Z,     # K/S far from 1
        ["2026-01-01", "SPY", "2026-02-06", 90.0, "Put", 0.4, 0.5, 0.3] + Z,     # holiday: no close that day
        ["2026-01-05", "SPY", "2026-02-06", 120.0, "Put", 1.0, 1.2, 0.3] + Z,    # below intrinsic (100 spot -> 20)
    ]
    raw = _raw(rows, [("2026-01-05", "SPY", 100.0)])
    out = clean_puts(raw)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["mid"] == 0.45 and np.isclose(row["spread"], 0.10)
    assert row["dte"] == 32 and np.isclose(row["tau"], 32 / 365)
    assert np.isclose(row["m"], 100 / 90)
    assert np.isclose(row["q"], 0.018)                       # 1.8 / 100
    assert 0.0392 < row["r"] < 0.0400                        # y between 4.0 and 4.1 percent, continuous
    assert out.attrs["drop_log"]["rows_in"] == len(rows)


def test_treasury_previous_day_fill_and_flat_extrapolation():
    ust = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "1_month": [4.0], "2_month": [4.0], "3_month": [4.0],
                        "6_month": [5.0], "1_year": [6.0]})
    dates = pd.Series(pd.to_datetime(["2026-01-03", "2026-01-03"]))          # no print that day -> previous day
    r = treasury_rate(ust, dates, np.array([0.01, 0.75]))
    assert np.isclose(r[0], np.log1p(0.04))                                  # below 1m: flat
    assert np.isclose(r[1], np.log1p(0.055))                                 # halfway 6m..1y


def test_trailing_dividend_window():
    div = pd.DataFrame({"act_symbol": ["X", "X"], "ex_date": pd.to_datetime(["2024-09-01", "2025-09-01"]), "amount": [1.0, 2.0]})
    assert np.isclose(trailing_dividend_yield(div, "X", pd.Timestamp("2025-10-01"), 100.0), 0.02)   # first one aged out
