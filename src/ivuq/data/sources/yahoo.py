"""Yahoo Finance loader — primary US option-chain source (Phase 2).

Fixes applied to the original draft (data.py):
  - Added the missing imports (numpy, pandas, datetime, typing) — the draft had none.
  - `MarketInputs` was referenced but never defined; it's a dataclass now.
  - `load_market_data_from_csv` was defined twice, the first copy a dead stub with
    a "# ... existing code unchanged ..." placeholder that was never executed.
  - The CSV loader hardcoded ticker="AAPL", r=0.045, q=0.0032 for every file; these
    are now parameters, since the project needs four different underlyings.
  - Output columns are aligned to the schema in ivuq/data/schema.py (`underlying`,
    `option_style`, `data_source`, `market_option_price` instead of `ticker`, etc.).

Yahoo has no official support and can rate-limit or change shape without notice
(BUILD_PLAN.md 3.1 / the group table). ivuq.data.sources.tradier is the fallback
when this breaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ivuq.data.schema import option_style_for


@dataclass
class MarketInputs:
    underlying_price: float
    risk_free_rate: float
    dividend_yield: float
    valuation_date: date


class YahooMarketDataCollector:
    """Collects the Phase-1 variables from Yahoo Finance via yfinance."""

    def __init__(
        self,
        ticker: str,
        manual_risk_free_rate: float,
        manual_dividend_yield: float,
        canonical_underlying: Optional[str] = None,
    ) -> None:
        """`ticker` is the symbol yfinance is queried with. `canonical_underlying`
        is the schema-facing name (see ivuq.data.schema) — they differ for the
        index groups: Yahoo lists SPX/NDX option chains under "^SPX"/"^NDX", but
        the schema's underlying label (and option_style_for lookup) is "SPX"/"NDX"
        with no caret. Defaults to `ticker` itself, which is correct for NVDA/AAPL.
        """
        self.ticker = ticker.upper().strip()
        self.canonical_underlying = (canonical_underlying or self.ticker).upper().strip()
        self.manual_risk_free_rate = float(manual_risk_free_rate)
        self.manual_dividend_yield = float(manual_dividend_yield)
        self._ticker_object = None

    @staticmethod
    def _load_yfinance():
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ImportError("yfinance is not installed. Run: pip install yfinance") from exc
        return yf

    @property
    def ticker_object(self):
        if self._ticker_object is None:
            yf = self._load_yfinance()
            self._ticker_object = yf.Ticker(self.ticker)
        return self._ticker_object

    def get_underlying_price(self) -> float:
        try:
            value = self.ticker_object.fast_info["last_price"]
            if value is not None and np.isfinite(value) and value > 0:
                return float(value)
        except Exception:
            pass
        history = self.ticker_object.history(period="5d", auto_adjust=False)
        if history.empty or "Close" not in history:
            raise RuntimeError(f"Unable to retrieve underlying price for {self.ticker}.")
        price = float(history["Close"].dropna().iloc[-1])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError("The retrieved underlying price is invalid.")
        return price

    def get_risk_free_rate(self) -> float:
        try:
            yf = self._load_yfinance()
            treasury = yf.Ticker("^IRX").history(period="5d", auto_adjust=False)
            quoted_rate = float(treasury["Close"].dropna().iloc[-1])
            if np.isfinite(quoted_rate) and quoted_rate >= 0:
                return quoted_rate / 100.0
        except Exception:
            pass
        return self.manual_risk_free_rate

    def get_dividend_yield(self) -> float:
        """Prefer `trailingAnnualDividendYield` — it's already a clean decimal
        fraction (e.g. 0.0034 for AAPL). `dividendYield` is unreliable across
        yfinance versions: it used to be a fraction, but as of yfinance>=0.2 it
        is reported as a percent-style number (0.35 meaning "0.35%", not "35%"),
        which silently broke a naive `/100 if value > 1.0` heuristic — a value
        like 0.35 or 0.46 sits below 1.0 and was returned unscaled, producing a
        ~35-46% "dividend yield" instead of ~0.3-0.5%. Confirmed against live
        AAPL/NVDA/MSFT data. Always divide `dividendYield` by 100.
        """
        try:
            information = self.ticker_object.info
            trailing = information.get("trailingAnnualDividendYield")
            if trailing is not None and np.isfinite(trailing) and trailing >= 0:
                return float(trailing)
            percent_style = information.get("dividendYield")
            if percent_style is not None and np.isfinite(percent_style) and percent_style >= 0:
                return float(percent_style) / 100.0
        except Exception:
            pass
        return self.manual_dividend_yield

    def get_available_expiries(self) -> list[date]:
        raw_expiries = list(self.ticker_object.options)
        if not raw_expiries:
            raise RuntimeError(f"No listed option expiries returned for {self.ticker}.")
        return [datetime.strptime(expiry, "%Y-%m-%d").date() for expiry in raw_expiries]

    def select_expiry(self, target_days_to_expiry: int, valuation_date: Optional[date] = None) -> date:
        valuation_date = valuation_date or datetime.now(timezone.utc).date()
        future_expiries = [expiry for expiry in self.get_available_expiries() if expiry > valuation_date]
        if not future_expiries:
            raise RuntimeError("No future option expiries available.")
        return min(future_expiries, key=lambda expiry: abs((expiry - valuation_date).days - target_days_to_expiry))

    def get_option_chain(self, expiry: date, valuation_date: Optional[date] = None) -> tuple[pd.DataFrame, dict]:
        valuation_date = valuation_date or datetime.now(timezone.utc).date()
        underlying_price = self.get_underlying_price()
        risk_free_rate = self.get_risk_free_rate()
        dividend_yield = self.get_dividend_yield()
        chain = self.ticker_object.option_chain(expiry.strftime("%Y-%m-%d"))

        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        option_data = pd.concat([calls, puts], ignore_index=True)

        required_cols = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]
        missing = [c for c in required_cols if c not in option_data.columns]
        if missing:
            raise RuntimeError(f"Yahoo Finance missing columns: {missing}")

        for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
            option_data[c] = pd.to_numeric(option_data[c], errors="coerce")

        option_data = option_data.rename(columns={
            "contractSymbol": "contract_symbol",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "impliedVolatility": "vendor_implied_volatility",
        })

        valid_bid_ask = option_data["bid"].gt(0) & option_data["ask"].ge(option_data["bid"])
        option_data["market_option_price"] = np.where(
            valid_bid_ask, (option_data["bid"] + option_data["ask"]) / 2.0, option_data["last_price"]
        )
        option_data["bid_ask_spread"] = option_data["ask"] - option_data["bid"]
        option_data["underlying"] = self.canonical_underlying
        option_data["option_style"] = option_style_for(self.canonical_underlying)
        option_data["data_source"] = "yahoo"
        option_data["quote_date"] = pd.Timestamp(valuation_date)
        option_data["expiry_date"] = pd.Timestamp(expiry)
        option_data["underlying_price"] = underlying_price
        option_data["risk_free_rate"] = risk_free_rate
        option_data["dividend_yield"] = dividend_yield

        final_cols = [
            "underlying", "option_style", "contract_symbol", "option_type", "quote_date",
            "underlying_price", "strike", "expiry_date", "risk_free_rate",
            "dividend_yield", "vendor_implied_volatility", "market_option_price",
            "bid", "ask", "bid_ask_spread", "volume", "open_interest", "data_source",
        ]
        option_data = option_data[final_cols].sort_values(["option_type", "strike"]).reset_index(drop=True)

        metadata = {
            "underlying": self.canonical_underlying,
            "valuation_date": valuation_date,
            "expiry_date": expiry,
            "underlying_price": underlying_price,
            "risk_free_rate": risk_free_rate,
            "dividend_yield": dividend_yield,
            "data_source": "yahoo",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return option_data, metadata

    def _validate_price_bounds(self, row: pd.Series, S: float, r: float, q: float, T: float) -> bool:
        """Check that market price respects basic European no-arbitrage bounds."""
        K = float(row["strike"])
        price = float(row["market_option_price"])
        is_call = row["option_type"] == "call"

        df_r = np.exp(-r * T)
        df_q = np.exp(-q * T)

        if is_call:
            intrinsic = max(S * df_q - K * df_r, 0.0)
            upper_bound = S * df_q * 1.05  # tolerance for American early-exercise premium
        else:
            intrinsic = max(K * df_r - S * df_q, 0.0)
            upper_bound = K * df_r * 1.05

        return intrinsic <= price <= upper_bound

    def clean_option_chain(
        self,
        df: pd.DataFrame,
        S: float,
        r: float,
        q: float,
        T: float,
        min_volume: int = 5,
        min_oi: int = 0,
        max_spread_pct: float = 0.15,
        moneyness_min: float = 0.60,
        moneyness_max: float = 1.40,
    ) -> pd.DataFrame:
        """Full cleaning pipeline for a raw Yahoo Finance option chain.

        Steps: dedup by contract -> drop missing critical fields -> basic price
        validity -> liquidity filter -> moneyness filter -> no-arbitrage bounds.

        `min_oi` defaults to 0 (not enforced): confirmed against live data that
        Yahoo's `openInterest` field is unreliable across the board right now
        (0/392 populated for a real SPX pull, 2/189 for NVDA) rather than a
        SPX-specific gap. open_interest is an OPTIONAL column in the schema for
        exactly this reason — liquidity filtering here leans on volume and
        spread instead. Raise `min_oi` explicitly if a source's OI is verified
        trustworthy.
        """
        df = df.copy()

        df = df.drop_duplicates(subset=["contract_symbol"], keep="first")

        required = ["strike", "market_option_price", "bid", "ask", "option_type"]
        df = df.dropna(subset=required)

        df = df[df["market_option_price"] > 0.01]
        df = df[df["bid"] >= 0]
        df = df[df["ask"] >= df["bid"]]

        df["spread_pct"] = (df["ask"] - df["bid"]) / df["market_option_price"].clip(lower=0.01)
        df = df[
            (df["volume"].fillna(0).ge(min_volume))
            & (df["open_interest"].fillna(0).ge(min_oi))
            & (df["spread_pct"].le(max_spread_pct))
        ]

        df = df[
            (df["strike"].ge(moneyness_min * S))
            & (df["strike"].le(moneyness_max * S))
        ]

        valid_mask = df.apply(lambda row: self._validate_price_bounds(row, S, r, q, T), axis=1)
        df = df[valid_mask]

        return df.drop(columns=["spread_pct"], errors="ignore")


def load_market_data_from_csv(
    info_path: str,
    history_path: str,
    calls_path: str,
    puts_path: str,
    ticker: str,
    risk_free_rate: float,
    dividend_yield: float,
    valuation_date: date,
    expiry_date: date,
) -> tuple[MarketInputs, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-fetched Yahoo Finance CSVs and build a MarketInputs object.

    `ticker`, `risk_free_rate`, and `dividend_yield` are parameters, not hardcoded,
    since this loader is shared across four different underlyings (SPX, NDX, NVDA, AAPL).
    """
    info_df = pd.read_csv(info_path)
    hist_df = pd.read_csv(history_path)
    calls_df = pd.read_csv(calls_path)
    puts_df = pd.read_csv(puts_path)

    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df = hist_df.sort_values("Date").reset_index(drop=True)
    hist_df["log_return"] = np.log(hist_df["Close"] / hist_df["Close"].shift(1))

    S = float(info_df["regularMarketPrice"].values[0])

    calls_df = calls_df[calls_df["impliedVolatility"] > 0.001].copy()
    puts_df = puts_df[puts_df["impliedVolatility"] > 0.001].copy()
    calls_df["midPrice"] = (calls_df["bid"] + calls_df["ask"]) / 2
    puts_df["midPrice"] = (puts_df["bid"] + puts_df["ask"]) / 2
    calls_df["option_type"] = "call"
    puts_df["option_type"] = "put"

    style = option_style_for(ticker)
    for df_ in (calls_df, puts_df):
        df_.rename(columns={
            "contractSymbol": "contract_symbol",
            "lastPrice": "last_price",
            "openInterest": "open_interest",
            "impliedVolatility": "vendor_implied_volatility",
        }, inplace=True)
        df_["market_option_price"] = df_["midPrice"]
        df_["bid_ask_spread"] = df_["ask"] - df_["bid"]
        df_["underlying"] = ticker
        df_["option_style"] = style
        df_["data_source"] = "yahoo"
        df_["quote_date"] = pd.Timestamp(valuation_date)
        df_["expiry_date"] = pd.Timestamp(expiry_date)
        df_["underlying_price"] = S
        df_["risk_free_rate"] = risk_free_rate
        df_["dividend_yield"] = dividend_yield

    market_inputs = MarketInputs(
        underlying_price=S,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        valuation_date=valuation_date,
    )
    return market_inputs, calls_df, puts_df, hist_df
