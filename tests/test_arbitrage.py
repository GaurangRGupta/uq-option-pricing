from datetime import date, timedelta

import numpy as np
import pandas as pd

from ivuq.arbitrage.checker import (
    calendar_spread_violations,
    check_arbitrage,
    put_call_parity_violations,
    strike_convexity_violations,
    strike_monotonicity_violations,
)


def _row(strike, option_type, price, expiry_days=30, underlying="TEST", S=100.0, r=0.04, q=0.01, quote=None):
    quote = quote or date.today()
    return {
        "underlying": underlying,
        "option_type": option_type,
        "strike": strike,
        "quote_date": pd.Timestamp(quote),
        "expiry_date": pd.Timestamp(quote + timedelta(days=expiry_days)),
        "underlying_price": S,
        "risk_free_rate": r,
        "dividend_yield": q,
        "market_option_price": price,
    }


def test_put_call_parity_clean():
    S, K, r, q, T = 100.0, 100.0, 0.04, 0.01, 30 / 365
    call = 5.0
    put = call - (S * np.exp(-q * T) - K * np.exp(-r * T))
    df = pd.DataFrame([_row(K, "call", call), _row(K, "put", put)])
    assert put_call_parity_violations(df, tol=0.01).empty


def test_put_call_parity_violation_detected():
    df = pd.DataFrame([_row(100.0, "call", 5.0), _row(100.0, "put", 50.0)])
    violations = put_call_parity_violations(df, tol=0.5)
    assert len(violations) == 1


def test_strike_monotonicity_clean_calls():
    df = pd.DataFrame([_row(k, "call", p) for k, p in [(90, 12.0), (100, 6.0), (110, 2.0)]])
    assert strike_monotonicity_violations(df).empty


def test_strike_monotonicity_violation_detected():
    df = pd.DataFrame([_row(k, "call", p) for k, p in [(90, 6.0), (100, 12.0), (110, 2.0)]])
    violations = strike_monotonicity_violations(df)
    assert len(violations) == 1


def test_strike_convexity_clean():
    df = pd.DataFrame([_row(k, "call", p) for k, p in [(90, 12.0), (100, 6.0), (110, 2.0)]])
    assert strike_convexity_violations(df).empty


def test_strike_convexity_violation_detected():
    df = pd.DataFrame([_row(k, "call", p) for k, p in [(90, 2.0), (100, 20.0), (110, 2.0)]])
    violations = strike_convexity_violations(df)
    assert len(violations) == 1


def test_calendar_spread_clean():
    df = pd.DataFrame([
        _row(100.0, "call", 5.0, expiry_days=30),
        _row(100.0, "call", 7.0, expiry_days=60),
    ])
    assert calendar_spread_violations(df).empty


def test_calendar_spread_violation_detected():
    df = pd.DataFrame([
        _row(100.0, "call", 7.0, expiry_days=30),
        _row(100.0, "call", 5.0, expiry_days=60),
    ])
    violations = calendar_spread_violations(df)
    assert len(violations) == 1


def test_check_arbitrage_aggregates_all_four():
    df = pd.DataFrame([_row(100.0, "call", 5.0), _row(100.0, "put", 50.0)])
    report = check_arbitrage(df)
    assert report.n_rows == 2
    assert report.put_call_parity >= 1
    assert "put_call_parity" in report.details
