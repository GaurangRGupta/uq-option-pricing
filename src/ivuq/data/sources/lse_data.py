"""London Strategic Edge — free market-data platform (londonstrategicedge.com).

This is a REST-pull-plus-WebSocket-stream SDK (`pip install lse-data`), not the
London Stock Exchange Group. Confirmed from their GitHub (londonstrategicedge/lse-data):
  - `client.options(ticker, type=..., max_dte=...)`      -> REST pull, one snapshot
  - `client.stream([...])` / `client.subscribe_options()` -> live push over a
    WebSocket under the hood

Per the project decision: use `options()` for historical/batch pulls and
`stream_option_chain()` (below, wrapping `subscribe_options`/`stream_async`) for
anything live — that's the "websocket, not the plain API" distinction.

STATUS: field mapping confirmed against a real, live options() response
(2026-08-13, NVDA). Actual keys returned: ticker, underlying, strike, expiry,
contract_type, last_price, volume_today, premium_today, underlying_price,
dte, iv, delta, gamma, theta, vega, rho, last_trade_at, updated_at. No bid,
ask, or open_interest field exists at all (not just null) — confirms the
user's original note that this source doesn't carry them. theta/vega/rho are
a bonus beyond what was assumed but aren't mapped into the shared schema
(only vendor_implied_volatility/delta/gamma exist there) — add
vendor_theta/vega/rho columns if that turns out to be worth having.

Their terms prohibit redistributing the raw data to third parties — fine for a
capstone, just don't publish raw dumps (only derived numbers/figures).
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Iterator, Optional

import pandas as pd

from ivuq.data.schema import option_style_for


def _load_lse_client(api_key: Optional[str]):
    try:
        from lse import LSE  # package name per their README: pip install lse-data
    except ImportError as exc:
        raise ImportError(
            "lse-data is not installed. Run: pip install lse-data"
        ) from exc
    key = api_key or os.environ.get("LSE_DATA_API_KEY")
    if not key:
        raise ValueError(
            "No London Strategic Edge API key. Pass api_key=... or set the "
            "LSE_DATA_API_KEY environment variable. Get one free at "
            "https://londonstrategicedge.com/data"
        )
    return LSE(api_key=key)


def _row_to_schema(row: dict, underlying: str, quote_date) -> dict:
    """Map one lse-data options() record onto the shared schema.

    Field names confirmed against a real live response (see module docstring).
    """
    style = option_style_for(underlying)
    return {
        "underlying": underlying,
        "option_style": style,
        "contract_symbol": row.get("ticker"),
        "option_type": (row.get("contract_type") or "").lower(),
        "quote_date": pd.Timestamp(quote_date),
        "underlying_price": row.get("underlying_price"),
        "strike": row.get("strike"),
        "expiry_date": pd.Timestamp(row["expiry"]) if row.get("expiry") else None,
        "risk_free_rate": None,   # joined separately, not provided by this source
        "dividend_yield": None,   # joined separately, not provided by this source
        "vendor_implied_volatility": row.get("iv"),
        "vendor_delta": row.get("delta"),
        "vendor_gamma": row.get("gamma"),
        "market_option_price": row.get("last_price"),
        "bid": None,   # confirmed: no bid/ask field in the response at all
        "ask": None,   # confirmed: no bid/ask field in the response at all
        "last_price": row.get("last_price"),
        "volume": row.get("volume_today"),
        "open_interest": row.get("open_interest"),  # confirmed: field doesn't exist, always None
        "data_source": "lse_data",
    }


def get_option_chain(
    underlying: str,
    option_type: Optional[str] = None,
    max_dte: Optional[int] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """One-shot REST pull via client.options() — the non-streaming path."""
    client = _load_lse_client(api_key)
    rows = client.options(underlying.lower(), type=option_type, max_dte=max_dte)
    quote_date = datetime.now().date()
    records = [_row_to_schema(dict(row), underlying, quote_date) for row in rows]
    return pd.DataFrame.from_records(records)


def stream_option_chain(
    underlyings: Iterable[str],
    api_key: Optional[str] = None,
) -> Iterator[dict]:
    """Live push updates via the WebSocket-backed stream() method.

    Yields raw ticks (not yet mapped to the schema) — the caller decides how to
    buffer/aggregate before writing to disk, since a live feed has no natural
    "one row per day" boundary the way a REST snapshot does.
    """
    client = _load_lse_client(api_key)
    for tick in client.stream(list(underlyings)):
        yield tick
