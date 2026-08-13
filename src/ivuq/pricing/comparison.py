"""Pricer comparison table (BUILD_PLAN.md Weeks 5-9 "Done when" artifact).

Takes a chain that already has our own solved `implied_volatility` attached
(see ivuq.data.pull) and reprices every row with the other pricers built in
this project, so the three implementations can be compared against real
market data, not just against each other on synthetic test cases.

  - European rows: BS price (should equal market price almost exactly, since
    IV was solved from it) alongside the tree run in European mode — this is
    a live cross-check that the tree agrees with the closed form.
  - American rows: tree price (the one IV was actually solved against) is the
    reference; BAW is the fast approximation being checked against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivuq.pricing import black_scholes
from ivuq.pricing.baw import baw_price
from ivuq.pricing.binomial import crr_price

__all__ = ["compare_pricers"]


def _year_fraction(row) -> float:
    return max((row.expiry_date - row.quote_date).days / 365.0, 1e-6)


def compare_pricers(df: pd.DataFrame) -> pd.DataFrame:
    """Requires `implied_volatility` (see ivuq.data.pull._attach_implied_vol).
    Rows where IV wasn't solvable are skipped (nothing to reprice with).
    """
    if "implied_volatility" not in df.columns:
        raise ValueError("df must have implied_volatility attached first (see ivuq.data.pull)")

    df = df[df["implied_volatility"].notna()].copy()
    bs_prices, tree_prices, baw_prices = [], [], []

    for row in df.itertuples(index=False):
        T = _year_fraction(row)
        args = (row.underlying_price, row.strike, T, row.risk_free_rate, row.dividend_yield, row.implied_volatility)
        is_american = row.option_style == "american"

        bs_prices.append(black_scholes.price(*args, row.option_type))
        tree_prices.append(crr_price(*args, row.option_type, is_american=is_american))
        baw_prices.append(baw_price(*args, row.option_type) if is_american else np.nan)

    df["bs_price"] = bs_prices
    df["tree_price"] = tree_prices
    df["baw_price"] = baw_prices
    df["tree_vs_market_diff"] = df["tree_price"] - df["market_option_price"]
    df["bs_vs_tree_diff"] = df["bs_price"] - df["tree_price"]
    df["baw_vs_tree_diff"] = df["baw_price"] - df["tree_price"]
    return df
