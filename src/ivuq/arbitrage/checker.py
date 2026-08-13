"""No-arbitrage checks on real market option chains.

Four classic static checks, each on the raw `market_option_price` (not our
solved IV, since the whole point is to catch prices a rational market
shouldn't produce in the first place):

  - put-call parity      : C - P == S*exp(-qT) - K*exp(-rT), same (underlying, expiry, strike)
  - strike monotonicity   : calls non-increasing in K, puts non-decreasing in K, same expiry
  - strike convexity      : butterfly spread can't be negative, same expiry
  - calendar spread       : price non-decreasing in T, same (underlying, option_type, strike)

These operate on the "law of one price" a snapshot should obey; violations in
real, noisy market quotes are expected at the margin (stale quotes, wide
spreads) and are reported as counts + the offending rows, not raised as
errors — the caller decides what to do with them (BUILD_PLAN.md's own
arbitrage-checker task treats this as a diagnostic, not a hard filter).

Every check's tolerance is `tol + spread_multiplier * (the relevant quotes'
own bid-ask spread(s))`. Without the spread term, a fixed near-zero `tol`
flags ordinary bid-ask noise as "arbitrage" — confirmed against a real SPX
pull, where a near-zero tolerance flagged ~30% of rows as convexity
violations, almost all just adjacent strikes quoted independently with a few
cents/dollars of spread, not real mispricing. Falls back to `tol` alone if
`bid_ask_spread` isn't in the frame (e.g. a source that never has bid/ask).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "put_call_parity_violations",
    "strike_monotonicity_violations",
    "strike_convexity_violations",
    "calendar_spread_violations",
    "check_arbitrage",
    "ArbitrageReport",
]


def _year_fraction(expiry: pd.Series, quote: pd.Series) -> pd.Series:
    return (expiry - quote).dt.days / 365.0


def put_call_parity_violations(df: pd.DataFrame, tol: float = 0.05, spread_multiplier: float = 1.0) -> pd.DataFrame:
    """C - P vs S*exp(-qT) - K*exp(-rT), matched on (underlying, expiry_date, strike)."""
    calls = df[df["option_type"] == "call"]
    puts = df[df["option_type"] == "put"]
    keys = ["underlying", "expiry_date", "strike"]
    merged = calls.merge(puts, on=keys, suffixes=("_call", "_put"))
    if merged.empty:
        return merged

    T = _year_fraction(merged["expiry_date"], merged["quote_date_call"])
    S = merged["underlying_price_call"]
    K = merged["strike"]
    r = merged["risk_free_rate_call"]
    q = merged["dividend_yield_call"]

    if "bid_ask_spread_call" in merged.columns and "bid_ask_spread_put" in merged.columns:
        adaptive = spread_multiplier * (
            merged["bid_ask_spread_call"].fillna(0) + merged["bid_ask_spread_put"].fillna(0)
        )
    else:
        adaptive = 0.0
    threshold = tol + adaptive

    lhs = merged["market_option_price_call"] - merged["market_option_price_put"]
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    merged["parity_gap"] = lhs - rhs
    violations = merged[merged["parity_gap"].abs() > threshold]
    return violations[keys + ["parity_gap", "market_option_price_call", "market_option_price_put"]]


def strike_monotonicity_violations(df: pd.DataFrame, tol: float = 1e-6, spread_multiplier: float = 1.0) -> pd.DataFrame:
    """Calls must be non-increasing in strike; puts non-decreasing; per (underlying, expiry, type)."""
    rows = []
    for (underlying, expiry, opt_type), group in df.groupby(["underlying", "expiry_date", "option_type"]):
        group = group.sort_values("strike")
        diffs = group["market_option_price"].diff()
        if "bid_ask_spread" in group.columns:
            adaptive = spread_multiplier * (group["bid_ask_spread"].fillna(0) + group["bid_ask_spread"].shift(1).fillna(0))
        else:
            adaptive = 0.0
        threshold = tol + adaptive
        bad = diffs > threshold if opt_type == "call" else -diffs > threshold
        bad_rows = group[bad.fillna(False)]
        rows.append(bad_rows)
    if not rows:
        return df.iloc[0:0]
    return pd.concat(rows)


def strike_convexity_violations(df: pd.DataFrame, tol: float = 1e-6, spread_multiplier: float = 1.0) -> pd.DataFrame:
    """Butterfly no-arbitrage: for consecutive strikes K1<K2<K3 (same underlying,
    expiry, option_type), price(K2) must not exceed the K1-K3 chord value.
    """
    rows = []
    for (underlying, expiry, opt_type), group in df.groupby(["underlying", "expiry_date", "option_type"]):
        group = group.sort_values("strike").reset_index()
        k = group["strike"].to_numpy()
        p = group["market_option_price"].to_numpy()
        has_spread = "bid_ask_spread" in group.columns
        spreads = group["bid_ask_spread"].fillna(0).to_numpy() if has_spread else np.zeros(len(group))
        for i in range(1, len(group) - 1):
            k1, k2, k3 = k[i - 1], k[i], k[i + 1]
            if k3 == k1:
                continue
            weight = (k3 - k2) / (k3 - k1)
            chord = weight * p[i - 1] + (1 - weight) * p[i + 1]
            threshold = tol + spread_multiplier * (spreads[i - 1] + spreads[i] + spreads[i + 1])
            if p[i] - chord > threshold:
                rows.append(group.iloc[[i]])
    if not rows:
        return df.iloc[0:0]
    return pd.concat(rows).drop(columns=["index"], errors="ignore")


def calendar_spread_violations(df: pd.DataFrame, tol: float = 1e-6, spread_multiplier: float = 1.0) -> pd.DataFrame:
    """Same underlying/type/strike, price must be non-decreasing as expiry lengthens."""
    rows = []
    for (underlying, opt_type, strike), group in df.groupby(["underlying", "option_type", "strike"]):
        group = group.sort_values("expiry_date")
        diffs = group["market_option_price"].diff()
        if "bid_ask_spread" in group.columns:
            adaptive = spread_multiplier * (group["bid_ask_spread"].fillna(0) + group["bid_ask_spread"].shift(1).fillna(0))
        else:
            adaptive = 0.0
        threshold = tol + adaptive
        bad_rows = group[(diffs < -threshold).fillna(False)]
        rows.append(bad_rows)
    if not rows:
        return df.iloc[0:0]
    return pd.concat(rows)


@dataclass
class ArbitrageReport:
    n_rows: int
    put_call_parity: int
    strike_monotonicity: int
    strike_convexity: int
    calendar_spread: int
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ArbitrageReport: {self.n_rows} rows checked — "
            f"parity={self.put_call_parity}, monotonicity={self.strike_monotonicity}, "
            f"convexity={self.strike_convexity}, calendar={self.calendar_spread}"
        )


def check_arbitrage(df: pd.DataFrame) -> ArbitrageReport:
    parity = put_call_parity_violations(df)
    monotonicity = strike_monotonicity_violations(df)
    convexity = strike_convexity_violations(df)
    calendar = calendar_spread_violations(df)
    return ArbitrageReport(
        n_rows=len(df),
        put_call_parity=len(parity),
        strike_monotonicity=len(monotonicity),
        strike_convexity=len(convexity),
        calendar_spread=len(calendar),
        details={
            "put_call_parity": parity,
            "strike_monotonicity": monotonicity,
            "strike_convexity": convexity,
            "calendar_spread": calendar,
        },
    )
