from datetime import date
from unittest.mock import MagicMock

import pytest

from ivuq.data.schema import validate_option_chain
from ivuq.data.sources.tradier import TradierClient


def _fake_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def client(monkeypatch):
    c = TradierClient(api_token="fake-sandbox-token")

    def fake_get(url, params=None, timeout=None):
        if "quotes" in url:
            return _fake_response({"quotes": {"quote": {"last": 120.0, "close": 119.5}}})
        if "expirations" in url:
            return _fake_response({"expirations": {"date": ["2026-06-19"]}})
        if "chains" in url:
            return _fake_response({
                "options": {
                    "option": [
                        {
                            "symbol": "NVDA260619C00120000",
                            "option_type": "call",
                            "strike": 120.0,
                            "bid": 9.8,
                            "ask": 10.2,
                            "last": 10.0,
                            "volume": 150,
                            "open_interest": 900,
                            "greeks": {"mid_iv": 0.31, "delta": 0.55, "gamma": 0.02},
                        },
                        {
                            "symbol": "NVDA260619P00120000",
                            "option_type": "put",
                            "strike": 120.0,
                            "bid": 8.7,
                            "ask": 9.1,
                            "last": 8.9,
                            "volume": 90,
                            "open_interest": 400,
                            "greeks": {"mid_iv": 0.30, "delta": -0.45, "gamma": 0.02},
                        },
                    ]
                }
            })
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(c._session, "get", fake_get)
    return c


def test_get_underlying_price(client):
    assert client.get_underlying_price("NVDA") == 120.0


def test_get_expirations(client):
    assert client.get_expirations("NVDA") == [date(2026, 6, 19)]


def test_get_option_chain_matches_schema(client):
    df, meta = client.get_option_chain(
        "NVDA", date(2026, 6, 19), underlying_price=120.0, risk_free_rate=0.05, dividend_yield=0.005
    )
    report = validate_option_chain(df)
    assert not report.missing_required_columns
    assert report.rows_dropped_for_missing_required_values == 0
    assert (df["option_style"] == "american").all()
    assert (df["data_source"] == "tradier").all()
    assert set(df["option_type"]) == {"call", "put"}
    # Greeks/IV are present here (Tradier gives them), unlike a bid/ask-free source.
    assert df["vendor_implied_volatility"].notna().all()


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("TRADIER_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        TradierClient(api_token=None, base_url="https://example.invalid")
