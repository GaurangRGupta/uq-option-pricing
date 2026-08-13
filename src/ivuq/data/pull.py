"""Pull a full option-chain snapshot (all listed expiries) for every in-scope
underlying and attach our own solved implied volatility.

yfinance only exposes the *current* option chain — there is no free source of
historical option chain snapshots (see the project's own scoping note in
claude-context.md). So "6 months to 1 year of data" cannot be backfilled; the
realistic plan is:
  1. Pull everything currently listed (all expiries) right now -> a full
     cross-sectional smile/term-structure snapshot, usable immediately.
  2. Re-run this module daily, saving a new dated file each time, so real
     historical depth accumulates going forward.

Ticker-vs-underlying note: Yahoo lists SPX/NDX option chains under "^SPX" /
"^NDX", but the schema's canonical underlying label is "SPX" / "NDX" (see
ivuq.data.schema). YahooMarketDataCollector's `canonical_underlying` param
handles that split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ivuq.data.schema import ALL_UNDERLYINGS, option_style_for, validate_option_chain
from ivuq.data.sources.yahoo import YahooMarketDataCollector
from ivuq.pricing.iv_solver import implied_vol

logger = logging.getLogger(__name__)

# Yahoo query symbol, and manual (r, q) fallbacks used only if the live lookup
# fails (^IRX history for r; index dividend yield is never in `.info`, so the
# manual value is what indices actually use). Estimates as of 2026, update if
# markedly stale.
_UNDERLYING_CONFIG = {
    "SPX": {"yahoo_ticker": "^SPX", "manual_rf": 0.04, "manual_div": 0.013},
    "NDX": {"yahoo_ticker": "^NDX", "manual_rf": 0.04, "manual_div": 0.007},
    "NVDA": {"yahoo_ticker": "NVDA", "manual_rf": 0.04, "manual_div": 0.0002},
    "AAPL": {"yahoo_ticker": "AAPL", "manual_rf": 0.04, "manual_div": 0.0034},
}


@dataclass
class SnapshotResult:
    underlying: str
    quote_date: date
    df: pd.DataFrame
    n_expiries_pulled: int
    n_expiries_failed: int
    n_rows_before_iv: int
    n_rows_iv_solved: int
    n_rows_iv_unsolvable: int


def _attach_implied_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Solve our own IV per row; rows whose market price is outside the
    no-arbitrage bracket get `implied_volatility = NaN` and `iv_solvable =
    False` rather than being silently dropped, so the caller can see exactly
    how much of the chain that was.
    """
    df = df.copy()
    ivs = []
    solvable = []
    for row in df.itertuples(index=False):
        is_american = row.option_style == "american"
        T = max((row.expiry_date - row.quote_date).days / 365.0, 1e-6)
        try:
            iv = implied_vol(
                market_price=float(row.market_option_price),
                S=float(row.underlying_price),
                K=float(row.strike),
                T=T,
                r=float(row.risk_free_rate),
                q=float(row.dividend_yield),
                option_type=row.option_type,
                is_american=is_american,
            )
            ivs.append(iv)
            solvable.append(True)
        except ValueError:
            ivs.append(float("nan"))
            solvable.append(False)
    df["implied_volatility"] = ivs
    df["iv_solvable"] = solvable
    return df


def pull_full_snapshot(
    underlying: str,
    max_expiries: Optional[int] = None,
    valuation_date: Optional[date] = None,
) -> SnapshotResult:
    """Pull every currently-listed expiry for one underlying, clean each
    expiry's chain, concatenate, and attach our own solved IV.

    `max_expiries` caps how many of the nearest expiries are pulled (None =
    all of them). Deep, far-dated, illiquid expiries are still requested by
    default since the cleaning step's liquidity filter will drop them anyway
    if nothing traded.
    """
    underlying = underlying.upper().strip()
    if underlying not in ALL_UNDERLYINGS:
        raise ValueError(f"{underlying!r} is not in scope: {ALL_UNDERLYINGS}")
    config = _UNDERLYING_CONFIG[underlying]

    valuation_date = valuation_date or datetime.now(timezone.utc).date()
    collector = YahooMarketDataCollector(
        ticker=config["yahoo_ticker"],
        manual_risk_free_rate=config["manual_rf"],
        manual_dividend_yield=config["manual_div"],
        canonical_underlying=underlying,
    )

    expiries = collector.get_available_expiries()
    expiries = [e for e in expiries if e > valuation_date]
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    frames = []
    n_failed = 0
    for expiry in expiries:
        try:
            raw_df, meta = collector.get_option_chain(expiry, valuation_date)
        except Exception as exc:  # yfinance can 404/empty-chain on thinly listed expiries
            logger.warning("skipping %s %s: %s", underlying, expiry, exc)
            n_failed += 1
            continue
        T = (expiry - valuation_date).days / 365.0
        cleaned = collector.clean_option_chain(
            raw_df, S=meta["underlying_price"], r=meta["risk_free_rate"], q=meta["dividend_yield"], T=T
        )
        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        raise RuntimeError(f"No usable option chain rows for {underlying} on {valuation_date}")

    combined = pd.concat(frames, ignore_index=True)
    n_before_iv = len(combined)
    combined = _attach_implied_vol(combined)
    n_solved = int(combined["iv_solvable"].sum())

    report = validate_option_chain(combined)
    if not report.ok:
        logger.warning("schema validation issue for %s: %s", underlying, report)

    return SnapshotResult(
        underlying=underlying,
        quote_date=valuation_date,
        df=combined,
        n_expiries_pulled=len(frames),
        n_expiries_failed=n_failed,
        n_rows_before_iv=n_before_iv,
        n_rows_iv_solved=n_solved,
        n_rows_iv_unsolvable=n_before_iv - n_solved,
    )


def save_snapshot(result: SnapshotResult, out_dir: str = "data/raw") -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"{result.underlying}_{result.quote_date.isoformat()}.csv"
    result.df.to_csv(file_path, index=False)
    return file_path


def pull_all(
    underlyings: tuple = ALL_UNDERLYINGS,
    max_expiries: Optional[int] = None,
    out_dir: str = "data/raw",
) -> list[SnapshotResult]:
    results = []
    for underlying in underlyings:
        result = pull_full_snapshot(underlying, max_expiries=max_expiries)
        path = save_snapshot(result, out_dir=out_dir)
        logger.info("saved %s -> %s (%d rows)", underlying, path, len(result.df))
        results.append(result)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for res in pull_all():
        print(
            f"{res.underlying}: {res.n_expiries_pulled} expiries pulled "
            f"({res.n_expiries_failed} failed), {len(res.df)} rows after cleaning, "
            f"{res.n_rows_iv_solved} IV-solvable, {res.n_rows_iv_unsolvable} unsolvable"
        )
