from unittest.mock import MagicMock

import pytest

from ivuq.data.sources import lse_data


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("LSE_DATA_API_KEY", raising=False)
    # Simulate the lse-data package being installed but no key configured.
    monkeypatch.setitem(__import__("sys").modules, "lse", MagicMock(LSE=MagicMock()))
    with pytest.raises(ValueError):
        lse_data._load_lse_client(api_key=None)


def test_row_to_schema_maps_confirmed_fields():
    # Field names confirmed against a real live options() response (2026-08-13, NVDA).
    row = {
        "ticker": "NVDA260619C00120000",
        "underlying": "NVDA",
        "strike": 120.0,
        "expiry": "2026-06-19",
        "contract_type": "call",
        "last_price": 10.0,
        "volume_today": 150,
        "premium_today": 150000,
        "underlying_price": 120.0,
        "dte": 5,
        "iv": 0.31,
        "delta": 0.55,
        "gamma": 0.02,
        "theta": None,
        "vega": None,
        "rho": None,
        "last_trade_at": "2026-01-05T19:24:12Z",
        "updated_at": "2026-01-05T19:24:12Z",
    }
    mapped = lse_data._row_to_schema(row, "NVDA", "2026-01-05")

    assert mapped["option_type"] == "call"
    assert mapped["option_style"] == "american"
    assert mapped["bid"] is None and mapped["ask"] is None  # confirmed: field doesn't exist in the response
    assert mapped["open_interest"] is None  # confirmed: field doesn't exist in the response
    assert mapped["data_source"] == "lse_data"
    assert mapped["market_option_price"] == 10.0
    assert mapped["volume"] == 150
    assert mapped["contract_symbol"] == "NVDA260619C00120000"


def test_get_option_chain_uses_options_client(monkeypatch):
    fake_client = MagicMock()
    fake_client.options.return_value = [
        {
            "ticker": "SPXW260619C05000000", "underlying": "SPX", "contract_type": "call",
            "underlying_price": 5000.0, "strike": 5000.0, "expiry": "2026-06-19", "dte": 30,
            "iv": 0.18, "delta": 0.5, "gamma": 0.001, "theta": None, "vega": None, "rho": None,
            "last_price": 120.0, "volume_today": 10, "premium_today": 1200,
        }
    ]
    monkeypatch.setattr(lse_data, "_load_lse_client", lambda api_key: fake_client)

    df = lse_data.get_option_chain("SPX", api_key="fake-key")
    assert len(df) == 1
    assert df.iloc[0]["underlying"] == "SPX"
    assert df.iloc[0]["option_style"] == "european"
    fake_client.options.assert_called_once()


def test_stream_option_chain_yields_raw_ticks(monkeypatch):
    fake_client = MagicMock()
    fake_client.stream.return_value = iter([{"symbol": "NVDA", "price": 121.0}])
    monkeypatch.setattr(lse_data, "_load_lse_client", lambda api_key: fake_client)

    ticks = list(lse_data.stream_option_chain(["NVDA"], api_key="fake-key"))
    assert ticks == [{"symbol": "NVDA", "price": 121.0}]
    fake_client.stream.assert_called_once_with(["NVDA"])
