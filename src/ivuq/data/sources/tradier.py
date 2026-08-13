"""Tradier sandbox — free fallback US option-chain source.

Used when Yahoo Finance breaks, rate-limits, or changes shape (yfinance is
unofficial and unsupported; BUILD_PLAN.md 3.1 flags this as a real risk).

Getting a key (free, no brokerage account or deposit needed):
  1. Register at https://developer.tradier.com
  2. Open a "sandbox" account from the developer dashboard
  3. Copy the sandbox bearer token, pass it as `api_token` or set
     the TRADIER_API_TOKEN environment variable

Sandbox data is delayed, not real-time — fine for this project, since nothing
here trades live. It gives real bid/ask/open interest and hourly-updated
Greeks, which is more complete than some other free sources.
"""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from ivuq.data.schema import option_style_for

_SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"


class TradierClient:
    def __init__(self, api_token: Optional[str] = None, base_url: str = _SANDBOX_BASE_URL) -> None:
        self.api_token = api_token or os.environ.get("TRADIER_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "No Tradier API token. Pass api_token=... or set the TRADIER_API_TOKEN "
                "environment variable. See module docstring for how to get a free one."
            )
        self.base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        })

    def _get(self, path: str, params: dict) -> dict:
        response = self._session.get(f"{self.base_url}{path}", params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_underlying_price(self, symbol: str) -> float:
        data = self._get("/markets/quotes", {"symbols": symbol})
        quote = data["quotes"]["quote"]
        price = float(quote["last"] if quote.get("last") is not None else quote["close"])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"Tradier returned an invalid price for {symbol}: {price}")
        return price

    def get_expirations(self, symbol: str) -> list[date]:
        data = self._get("/markets/options/expirations", {"symbol": symbol})
        raw = data["expirations"]["date"]
        raw = raw if isinstance(raw, list) else [raw]
        return [datetime.strptime(d, "%Y-%m-%d").date() for d in raw]

    def get_option_chain(
        self,
        symbol: str,
        expiration: date,
        underlying_price: Optional[float] = None,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        include_greeks: bool = True,
    ) -> tuple[pd.DataFrame, dict]:
        data = self._get(
            "/markets/options/chains",
            {"symbol": symbol, "expiration": expiration.strftime("%Y-%m-%d"), "greeks": str(include_greeks).lower()},
        )
        options = data.get("options")
        if not options or not options.get("option"):
            raise RuntimeError(f"Tradier returned no option chain for {symbol} {expiration}")
        rows = options["option"]

        underlying_price = underlying_price if underlying_price is not None else self.get_underlying_price(symbol)
        style = option_style_for(symbol)
        quote_date = datetime.now().date()

        records = []
        for row in rows:
            greeks = row.get("greeks") or {}
            bid = row.get("bid")
            ask = row.get("ask")
            valid_bid_ask = bid is not None and ask is not None and bid > 0 and ask >= bid
            mid = (bid + ask) / 2.0 if valid_bid_ask else row.get("last")
            records.append({
                "underlying": symbol,
                "option_style": style,
                "contract_symbol": row.get("symbol"),
                "option_type": row.get("option_type"),
                "quote_date": pd.Timestamp(quote_date),
                "underlying_price": underlying_price,
                "strike": float(row["strike"]),
                "expiry_date": pd.Timestamp(expiration),
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "vendor_implied_volatility": greeks.get("mid_iv"),
                "vendor_delta": greeks.get("delta"),
                "vendor_gamma": greeks.get("gamma"),
                "market_option_price": mid,
                "bid": bid,
                "ask": ask,
                "last_price": row.get("last"),
                "volume": row.get("volume"),
                "open_interest": row.get("open_interest"),
                "data_source": "tradier",
            })

        df = pd.DataFrame.from_records(records)
        metadata = {
            "underlying": symbol,
            "valuation_date": quote_date,
            "expiry_date": expiration,
            "underlying_price": underlying_price,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "data_source": "tradier",
            "downloaded_at_utc": datetime.utcnow().isoformat(),
        }
        return df, metadata
