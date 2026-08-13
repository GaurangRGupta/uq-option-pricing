"""The option-chain column contract every data source must fill in.

Scope is US-only, four underlyings (see DECISIONS.md and PROJECT_PLAN.md):
  - European, cash-settled index options: SPX, NDX
  - American, physically-settled single-stock options: NVDA, AAPL

Two tiers of columns:
  REQUIRED   - every loader must produce these, non-null, or the row is unusable.
  OPTIONAL   - nice to have. Yahoo usually has all of these; a fallback source
               (Tradier, London Strategic Edge, NSE-style feeds) may be missing
               some of them. That is an accepted tradeoff, not a blocker — the
               project only ever computes its own implied volatility and Greeks
               from price + rate + dividend yield, so bid/ask/open interest are
               used for liquidity filtering and cross-checks, not as required inputs.

Never trust a vendor's implied volatility or Greeks as the target. They live in
`vendor_implied_volatility` / `vendor_delta` etc., kept separate, used only to
sanity-check our own solver (see pricing/iv_solver.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

EUROPEAN_UNDERLYINGS = ("SPX", "NDX")
AMERICAN_UNDERLYINGS = ("NVDA", "AAPL")
ALL_UNDERLYINGS = EUROPEAN_UNDERLYINGS + AMERICAN_UNDERLYINGS

REQUIRED_COLUMNS = (
    "underlying",       # SPX, NDX, NVDA, AAPL
    "option_style",     # "european" | "american" — derived from underlying, never fetched
    "option_type",      # "call" | "put"
    "strike",
    "expiry_date",
    "quote_date",
    "underlying_price",     # spot at quote_date
    "market_option_price",  # mid(bid,ask) if available, else last_price/settle
    "risk_free_rate",       # continuously compounded, interpolated to this option's expiry
    "dividend_yield",       # continuous yield
    "data_source",          # "yahoo" | "tradier" | "lse_data"
)

OPTIONAL_COLUMNS = (
    "contract_symbol",
    "bid",
    "ask",
    "last_price",
    "volume",
    "open_interest",
    "vendor_implied_volatility",
    "vendor_delta",
    "vendor_gamma",
)

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def option_style_for(underlying: str) -> str:
    underlying = underlying.upper()
    if underlying in EUROPEAN_UNDERLYINGS:
        return "european"
    if underlying in AMERICAN_UNDERLYINGS:
        return "american"
    raise ValueError(
        f"Unrecognized underlying {underlying!r}. In-scope underlyings are "
        f"{ALL_UNDERLYINGS}. Index options (SPX, NDX) are European and cash-settled; "
        f"the corresponding ETF options (SPY, QQQ) are American and NOT interchangeable "
        f"with them — see DECISIONS.md."
    )


@dataclass
class ValidationReport:
    n_rows: int
    missing_required_columns: tuple = field(default_factory=tuple)
    rows_dropped_for_missing_required_values: int = 0
    optional_column_null_pct: dict = field(default_factory=dict)
    ok: bool = True

    def __str__(self) -> str:
        lines = [f"ValidationReport: {self.n_rows} rows, ok={self.ok}"]
        if self.missing_required_columns:
            lines.append(f"  MISSING REQUIRED COLUMNS: {self.missing_required_columns}")
        if self.rows_dropped_for_missing_required_values:
            lines.append(
                f"  {self.rows_dropped_for_missing_required_values} rows have a null "
                "in a required column"
            )
        for col, pct in self.optional_column_null_pct.items():
            lines.append(f"  optional '{col}': {pct:.1f}% null")
        return "\n".join(lines)


def validate_option_chain(df: pd.DataFrame) -> ValidationReport:
    """Check a loader's output against the column contract.

    Missing REQUIRED columns, or nulls within them, are reported but do not raise —
    callers decide whether to drop rows or reject the whole batch. Missing OPTIONAL
    columns are expected for some sources (e.g. NSE-style feeds have no bid/ask) and
    are reported purely as an information line.
    """
    missing_required = tuple(c for c in REQUIRED_COLUMNS if c not in df.columns)
    present_required = [c for c in REQUIRED_COLUMNS if c in df.columns]
    rows_with_null_required = int(df[present_required].isnull().any(axis=1).sum()) if present_required else len(df)

    optional_null_pct = {}
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            optional_null_pct[col] = 100.0 * df[col].isnull().mean()
        else:
            optional_null_pct[col] = 100.0

    return ValidationReport(
        n_rows=len(df),
        missing_required_columns=missing_required,
        rows_dropped_for_missing_required_values=rows_with_null_required,
        optional_column_null_pct=optional_null_pct,
        ok=not missing_required,
    )
