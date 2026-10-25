"""London Strategic Edge (LSE Data 3.0) REST client for the US options tape.

What the free plan carries for options (docs: github.com/londonstrategicedge/lse-data, docs/api.md): OPRA trades since
October 2025, candles at every timeframe built from the trades, and per-minute Greeks (implied vol, delta, ...)
computed twice a day. It does NOT carry bid/ask quotes, so it cannot drive the take-the-quote backtest; it is used
for independent price checks (traded prices against our quotes and theos) and for implied-vol history.

The key is read from the environment variable LSE_API_KEY or from a `.env` file in the working directory
(`LSE_API_KEY=...`, gitignored). It is sent in the X-API-Key header only and never printed or stored.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

__all__ = ["BASE", "load_key", "occ_symbol", "LSEClient"]

BASE = "https://download.londonstrategicedge.com"


def load_key(env_file: str | Path = ".env") -> str:
    key = os.environ.get("LSE_API_KEY")
    if not key and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LSE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("no LSE key: set LSE_API_KEY or put LSE_API_KEY=... in .env")
    return key


def occ_symbol(root: str, expiry, right: str, strike: float) -> str:
    """OCC contract name with the root padded to six characters, e.g. `AAPL  261120P00300000`."""
    e = pd.Timestamp(expiry).strftime("%y%m%d")
    return f"{root.upper():<6}{e}{right.upper()[0]}{int(round(strike * 1000)):08d}"


class LSEClient:
    def __init__(self, key: str | None = None, timeout: float = 120.0):
        self.key = key or load_key()
        self.timeout = timeout

    def _get(self, path: str, **params):
        r = requests.get(BASE + path, params={k: v for k, v in params.items() if v is not None},
                         headers={"X-API-Key": self.key}, timeout=self.timeout)
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {"message": r.text[:200]}
            raise RuntimeError(f"LSE {path} -> {r.status_code} {body.get('code')}: {body.get('message')}")
        return r

    def usage(self) -> dict:
        return self._get("/v3/usage").json()

    def chain(self, underlying: str, type: str | None = None, max_dte: int | None = None, expiry: str | None = None,
              limit: int = 1000) -> pd.DataFrame:
        """Unexpired contracts with the last trade, today's volume and the newest computed Greeks (no bid/ask)."""
        rows, offset = [], 0
        while True:
            j = self._get("/v3/options/chain", underlying=underlying, type=type, max_dte=max_dte, expiry=expiry,
                          limit=limit, offset=offset).json()
            rows += j["rows"]
            offset += j["returned"]
            if j["returned"] == 0 or offset >= j["contracts"]:
                break
        d = pd.DataFrame(rows)
        d.attrs.update(underlying_price=j.get("underlying_price"), greeks_as_of=j.get("greeks_as_of"))
        return d

    def _csv(self, **params) -> pd.DataFrame:
        r = self._get("/v3/export", format="csv", **params)
        return pd.read_csv(io.BytesIO(r.content))

    def greeks(self, symbol: str, start: str, end: str, timeframe: str = "1m") -> pd.DataFrame:
        """One row per traded minute: price_at, iv, delta, gamma, theta, vega, option_price, underlying_price, ..."""
        return self._csv(market="us_options", symbol=symbol, data="greeks", timeframe=timeframe, **{"from": start, "to": end})

    def candles(self, symbol: str, timeframe: str, start: str, end: str, market: str = "us_options") -> pd.DataFrame:
        return self._csv(market=market, symbol=symbol, timeframe=timeframe, data="candles", **{"from": start, "to": end})

    def trades(self, symbol: str, day: str) -> pd.DataFrame:
        """The trade slice of one contract on one day: time, price, size, kind."""
        return self._csv(market="us_options", symbol=symbol, timeframe="tick", data="candles", **{"from": day, "to": day})
