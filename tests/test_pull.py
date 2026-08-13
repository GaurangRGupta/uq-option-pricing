from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ivuq.data import pull
from ivuq.data.sources.yahoo import YahooMarketDataCollector
from ivuq.pricing.binomial import crr_price


def _mock_chain(expiry, S=100.0, r=0.04, q=0.01, sigma=0.25, strikes=(90.0, 100.0, 110.0)):
    quote_date = date.today()
    T = (expiry - quote_date).days / 365.0
    rows = []
    for k in strikes:
        for opt in ("call", "put"):
            price = crr_price(S, k, T, r, q, sigma, opt, is_american=True)
            rows.append({
                "underlying": "NVDA",
                "option_style": "american",
                "contract_symbol": f"NVDA{k}{opt}",
                "option_type": opt,
                "quote_date": pd.Timestamp(quote_date),
                "underlying_price": S,
                "strike": k,
                "expiry_date": pd.Timestamp(expiry),
                "risk_free_rate": r,
                "dividend_yield": q,
                "vendor_implied_volatility": sigma,
                "market_option_price": price,
                "bid": price - 0.05,
                "ask": price + 0.05,
                "bid_ask_spread": 0.1,
                "volume": 100,
                "open_interest": 500,
                "data_source": "yahoo",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def two_expiries():
    today = date.today()
    return [today + timedelta(days=30), today + timedelta(days=60)]


def _patch_collector(monkeypatch, expiries, chain_fn=None, fail_on=None):
    monkeypatch.setattr(YahooMarketDataCollector, "get_available_expiries", lambda self: expiries)

    def default_chain_fn(self, expiry, valuation_date=None):
        if fail_on is not None and expiry == fail_on:
            raise RuntimeError("Yahoo Finance missing columns")
        return _mock_chain(expiry), {"underlying_price": 100.0, "risk_free_rate": 0.04, "dividend_yield": 0.01}

    monkeypatch.setattr(YahooMarketDataCollector, "get_option_chain", chain_fn or default_chain_fn)
    monkeypatch.setattr(YahooMarketDataCollector, "clean_option_chain", lambda self, df, **kw: df)


def test_pull_full_snapshot_aggregates_expiries(monkeypatch, two_expiries):
    _patch_collector(monkeypatch, two_expiries)

    result = pull.pull_full_snapshot("NVDA")
    assert result.n_expiries_pulled == 2
    assert result.n_expiries_failed == 0
    assert result.n_rows_iv_solved == len(result.df)
    assert "implied_volatility" in result.df.columns
    assert np.allclose(result.df["implied_volatility"], 0.25, atol=1e-3)


def test_pull_full_snapshot_skips_failed_expiry(monkeypatch, two_expiries):
    _patch_collector(monkeypatch, two_expiries, fail_on=two_expiries[0])

    result = pull.pull_full_snapshot("NVDA")
    assert result.n_expiries_pulled == 1
    assert result.n_expiries_failed == 1


def test_pull_rejects_out_of_scope_underlying():
    with pytest.raises(ValueError):
        pull.pull_full_snapshot("SPY")


def test_save_snapshot_writes_csv(tmp_path, monkeypatch, two_expiries):
    _patch_collector(monkeypatch, two_expiries[:1])

    result = pull.pull_full_snapshot("NVDA")
    path = pull.save_snapshot(result, out_dir=str(tmp_path))
    assert path.exists()
    saved = pd.read_csv(path)
    assert len(saved) == len(result.df)
