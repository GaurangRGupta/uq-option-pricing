from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ivuq.pricing import black_scholes
from ivuq.pricing.binomial import crr_price
from ivuq.pricing.comparison import compare_pricers


def _row(option_style, option_type, S=100.0, K=100.0, r=0.04, q=0.01, sigma=0.25, days=30):
    quote_date = pd.Timestamp(date.today())
    expiry_date = pd.Timestamp(date.today() + timedelta(days=days))
    T = days / 365.0
    is_american = option_style == "american"
    price = (
        crr_price(S, K, T, r, q, sigma, option_type, is_american=True)
        if is_american
        else black_scholes.price(S, K, T, r, q, sigma, option_type)
    )
    return {
        "underlying": "TEST",
        "option_style": option_style,
        "option_type": option_type,
        "quote_date": quote_date,
        "expiry_date": expiry_date,
        "underlying_price": S,
        "strike": K,
        "risk_free_rate": r,
        "dividend_yield": q,
        "market_option_price": price,
        "implied_volatility": sigma,
    }


def test_compare_pricers_requires_iv_column():
    df = pd.DataFrame([_row("european", "call")]).drop(columns=["implied_volatility"])
    with pytest.raises(ValueError):
        compare_pricers(df)


def test_compare_pricers_european_matches_bs_and_tree():
    df = pd.DataFrame([_row("european", "call"), _row("european", "put")])
    out = compare_pricers(df)
    assert np.allclose(out["bs_price"], out["market_option_price"], atol=1e-4)
    assert np.allclose(out["bs_vs_tree_diff"], 0.0, atol=0.05)
    assert out["baw_price"].isna().all()


def test_compare_pricers_american_tree_matches_market_baw_close():
    df = pd.DataFrame([_row("american", "call", q=0.03), _row("american", "put", q=0.03)])
    out = compare_pricers(df)
    assert np.allclose(out["tree_price"], out["market_option_price"], atol=1e-4)
    assert (out["baw_vs_tree_diff"].abs() < 0.1).all()


def test_compare_pricers_skips_unsolvable_rows():
    df = pd.DataFrame([_row("european", "call")])
    df.loc[0, "implied_volatility"] = np.nan
    out = compare_pricers(df)
    assert out.empty
