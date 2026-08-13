from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ivuq.data.schema import validate_option_chain
from ivuq.data.sources.yahoo import YahooMarketDataCollector, option_style_for


def _fake_chain_frame(strikes, price_base=10.0):
    return pd.DataFrame({
        "contractSymbol": [f"NVDA{i}" for i in range(len(strikes))],
        "strike": strikes,
        "lastPrice": [price_base] * len(strikes),
        "bid": [price_base - 0.1] * len(strikes),
        "ask": [price_base + 0.1] * len(strikes),
        "volume": [100] * len(strikes),
        "openInterest": [500] * len(strikes),
        "impliedVolatility": [0.3] * len(strikes),
    })


class _FakeTickerObject:
    def __init__(self, expiry: date):
        self._expiry = expiry
        self.fast_info = {"last_price": 120.0}
        self.info = {"dividendYield": 0.0}
        self.options = [expiry.strftime("%Y-%m-%d")]

    def history(self, period="5d", auto_adjust=False):
        return pd.DataFrame({"Close": [120.0, 121.0]})

    def option_chain(self, expiry_str):
        calls = _fake_chain_frame([100.0, 110.0, 120.0, 130.0])
        puts = _fake_chain_frame([100.0, 110.0, 120.0, 130.0])
        return SimpleNamespace(calls=calls, puts=puts)


@pytest.fixture
def collector(monkeypatch):
    c = YahooMarketDataCollector("NVDA", manual_risk_free_rate=0.05, manual_dividend_yield=0.0)
    expiry = date.today() + timedelta(days=30)
    fake_obj = _FakeTickerObject(expiry)
    monkeypatch.setattr(YahooMarketDataCollector, "ticker_object", property(lambda self: fake_obj))
    return c, expiry


def test_get_option_chain_matches_schema(collector):
    c, expiry = collector
    df, meta = c.get_option_chain(expiry)

    report = validate_option_chain(df)
    assert not report.missing_required_columns
    assert report.rows_dropped_for_missing_required_values == 0
    assert (df["option_style"] == "american").all()
    assert (df["underlying"] == "NVDA").all()
    assert (df["data_source"] == "yahoo").all()
    assert meta["underlying_price"] == 120.0


def test_clean_option_chain_drops_wide_spreads_and_illiquid_rows(collector):
    c, expiry = collector
    df, meta = c.get_option_chain(expiry)
    df.loc[0, "volume"] = 0  # illiquid, should be dropped
    T = (expiry - date.today()).days / 365.0

    cleaned = c.clean_option_chain(
        df, S=meta["underlying_price"], r=meta["risk_free_rate"], q=meta["dividend_yield"], T=T
    )
    assert len(cleaned) < len(df)
    assert (cleaned["volume"] >= 5).all()


def test_option_style_rejects_unknown_underlying():
    with pytest.raises(ValueError):
        option_style_for("SPY")  # ETF, not the index — deliberately not in scope


@pytest.mark.parametrize("ticker,expected", [("SPX", "european"), ("NDX", "european"), ("NVDA", "american"), ("AAPL", "american")])
def test_option_style_for_in_scope_underlyings(ticker, expected):
    assert option_style_for(ticker) == expected
