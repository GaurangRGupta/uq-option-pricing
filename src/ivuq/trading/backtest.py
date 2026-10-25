"""Take-the-quote backtest of a theo-vs-market signal, delta hedged with the underlying.

Inputs are daily end-of-day panels (`dolthub.clean_puts` plus a theo column).
There is no intraday book, so the only fill model is to cross the spread at the
stamped end-of-day quote: buy at the ask, sell at the bid. No market-making
fills are simulated.

Signal for quote i (c_i the error scale, `calibration.heston_calib.error_scale`,
thr_i the trade threshold below):

    buy   if  theo_i - ask_i > thr_i       (market offered below theo by more than thr)
    sell  if  bid_i - theo_i > thr_i       (market bid above theo by more than thr)

Rules for thr_i:
  conformal   thr_i = q_i * c_i, q_i the past-only conformal quantile of |mid - theo| / c
              (`online_conformal.add_rolling_conformal`), so a trade needs the whole
              spread-crossing edge to clear the band
  spread      thr_i = 0, any positive edge after crossing the spread
  fixed_k     thr_i = k * c_i

At most `max_per_day` trades per underlying and date, ranked by (edge - thr) / c.

Trade. One contract (100 options) of side s = +1 (long) or -1 (short), entered at
trading day e = t + lag (lag 0: same close as the signal, optimistic; lag 1: next
close) and closed `horizon` trading days later at the opposite side of that day's
quote. A contract that expires on or before the exit date settles at intrinsic
value.

    option P&L = 100 s (exit - entry) - 2 * commission
    hedge P&L  = sum_{d=e}^{x-1} h_d (S_{d+1} + D_{d,d+1} - S_d) ,   h_d = -100 s Delta_d
                 less stock_cost_bp on every share traded (opening, daily rebalance, closing)

Delta_d is the vendor delta of that contract on day d ("vendor", the same hedge for
every theo being compared) or the model's own delta ("model"). D is the cash
dividend with ex-date in (d, d+1], so the hedge earns the total return of the stock.
The hedge removes the first-order exposure to the underlying, which leaves the P&L
from the market quote moving towards or away from theo, plus vega, gamma and costs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["BacktestConfig", "build_signals", "run_backtest", "summarise", "bootstrap_ci"]


@dataclass
class BacktestConfig:
    rule: str = "conformal"          # conformal | spread | fixed_k
    k: float = 1.0                   # for fixed_k
    horizon: int = 3                 # trading days
    lag: int = 0                     # 0 same close, 1 next close
    hedge: str = "vendor"            # vendor | model | none
    max_per_day: int = 10
    commission: float = 0.65         # dollars per contract per side
    stock_cost_bp: float = 1.0
    min_edge: float = 0.0            # absolute dollar floor on (edge - thr)
    dte_min_entry: int = 0


def _thr(df: pd.DataFrame, cfg: BacktestConfig) -> np.ndarray:
    if cfg.rule == "conformal":
        return (df["q"] * df["scale"]).to_numpy()
    if cfg.rule == "spread":
        return np.zeros(len(df))
    if cfg.rule == "fixed_k":
        return cfg.k * df["scale"].to_numpy()
    raise ValueError(cfg.rule)


def _signals(df: pd.DataFrame, theo_col: str, cfg: BacktestConfig) -> pd.DataFrame:
    thr = _thr(df, cfg)
    buy_edge = df[theo_col].to_numpy() - df["ask"].to_numpy()
    sell_edge = df["bid"].to_numpy() - df[theo_col].to_numpy()
    side = np.where((buy_edge > thr) & (buy_edge >= sell_edge), 1, np.where(sell_edge > thr, -1, 0))
    edge = np.where(side == 1, buy_edge, np.where(side == -1, sell_edge, 0.0))
    ok = (side != 0) & np.isfinite(edge) & (edge - thr > cfg.min_edge) & (df["dte"].to_numpy() >= cfg.dte_min_entry)
    s = df.loc[ok, ["date", "symbol", "expiry", "strike", "scale", "bucket"]].copy()
    s["side"] = side[ok]
    s["edge"] = edge[ok]
    s["excess"] = (edge[ok] - thr[ok]) / df.loc[ok, "scale"].to_numpy()
    s["theo"] = df.loc[ok, theo_col].to_numpy()
    s = s.sort_values(["symbol", "date", "excess"], ascending=[True, True, False])
    return s.groupby(["symbol", "date"], sort=False).head(cfg.max_per_day)


def build_signals(df: pd.DataFrame, theo_col: str, cfg: BacktestConfig) -> pd.DataFrame:
    """The trade list the rule would place, before any fills (public wrapper)."""
    return _signals(df, theo_col, cfg)


def run_backtest(panel: pd.DataFrame, theo_col: str, cfg: BacktestConfig, dividends: pd.DataFrame | None = None,
                 delta_model_col: str = "model_delta", signals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per trade (status ok, or no_exit when the exit quote is missing).
    `signals` overrides the rule (used by the random control)."""
    sig = _signals(panel, theo_col, cfg) if signals is None else signals
    dates = {s: np.array(sorted(g["date"].unique())) for s, g in panel.groupby("symbol")}
    spot = {s: dict(zip(g["date"], g["spot"])) for s, g in panel.groupby("symbol")}
    delta_col = "delta" if cfg.hedge == "vendor" else delta_model_col
    cols = ["bid", "ask", delta_col]
    quotes = {k: dict(zip(g["date"], g[cols].to_numpy())) for k, g in panel.groupby(["symbol", "expiry", "strike"])}
    div = {}
    if dividends is not None:
        for s, g in dividends.groupby("act_symbol"):
            div[s] = g.set_index("ex_date")["amount"].astype(float)
    div_cache: dict = {}

    def div_between(sym, d0, d1) -> float:
        if sym not in div:
            return 0.0
        key = (sym, d0, d1)
        if key not in div_cache:
            x = div[sym]
            div_cache[key] = float(x[(x.index > d0) & (x.index <= d1)].sum())
        return div_cache[key]

    rows = []
    for t in sig.itertuples():
        d_arr = dates[t.symbol]
        i0 = int(np.searchsorted(d_arr, t.date))
        e, x = i0 + cfg.lag, i0 + cfg.lag + cfg.horizon
        if x >= len(d_arr):
            continue
        qk = quotes[(t.symbol, t.expiry, t.strike)]
        d_e, d_x = d_arr[e], d_arr[x]
        if d_e not in qk:
            continue
        qe = qk[d_e]                                    # (bid, ask, delta)
        entry = qe[1] if t.side == 1 else qe[0]
        expired = t.expiry <= d_x
        if expired:
            exit_px = max(t.strike - float(spot[t.symbol].get(d_x, np.nan)), 0.0)
        elif d_x in qk:
            qx = qk[d_x]
            exit_px = qx[0] if t.side == 1 else qx[1]
        else:
            rows.append(dict(date=t.date, symbol=t.symbol, status="no_exit"))
            continue
        opt_pnl = 100.0 * t.side * (exit_px - entry)

        hedge_pnl, hedge_cost, prev_sh = 0.0, 0.0, 0.0
        if cfg.hedge != "none":
            last_delta = 0.0
            sp = spot[t.symbol]
            for d in range(e, x):
                dd = d_arr[d]
                if dd in qk and np.isfinite(qk[dd][2]):
                    last_delta = float(qk[dd][2])
                sh = -100.0 * t.side * last_delta
                S0, S1 = float(sp[dd]), float(sp[d_arr[d + 1]])
                hedge_pnl += sh * (S1 + div_between(t.symbol, dd, d_arr[d + 1]) - S0)
                hedge_cost += abs(sh - prev_sh) * S0 * cfg.stock_cost_bp * 1e-4
                prev_sh = sh
            hedge_cost += abs(prev_sh) * float(sp[d_arr[x - 1]]) * cfg.stock_cost_bp * 1e-4
        comm = 2.0 * cfg.commission
        rows.append(dict(date=t.date, symbol=t.symbol, expiry=t.expiry, strike=t.strike, side=t.side, bucket=t.bucket,
                         entry_date=d_e, exit_date=d_x, entry=entry, exit=exit_px, edge=t.edge, excess=t.excess,
                         theo=t.theo, scale=t.scale, opt_pnl=opt_pnl, hedge_pnl=hedge_pnl, hedge_cost=hedge_cost,
                         commission=comm, net_pnl=opt_pnl + hedge_pnl - hedge_cost - comm, expired=expired,
                         status="ok"))
    return pd.DataFrame(rows)


def summarise(trades: pd.DataFrame) -> dict:
    if not len(trades):
        return {"n_trades": 0, "n_no_exit": 0}
    ok = trades[trades["status"] == "ok"]
    n_no_exit = int((trades["status"] == "no_exit").sum())
    if not len(ok):
        return {"n_trades": 0, "n_no_exit": n_no_exit}
    daily = ok.groupby("exit_date")["net_pnl"].sum()
    sd = daily.std(ddof=1) if len(daily) > 2 else np.nan
    sharpe = float(daily.mean() / sd * np.sqrt(252)) if sd and sd > 0 else float("nan")
    cum = daily.sort_index().cumsum()
    return {
        "n_trades": int(len(ok)), "n_no_exit": n_no_exit,
        "hit_rate": float((ok["net_pnl"] > 0).mean()),
        "mean_net": float(ok["net_pnl"].mean()), "median_net": float(ok["net_pnl"].median()),
        "total_net": float(ok["net_pnl"].sum()),
        "mean_gross_hedged": float((ok["opt_pnl"] + ok["hedge_pnl"]).mean()),
        "mean_edge_at_entry": float(100 * ok["edge"].mean()),
        "mean_costs": float((ok["hedge_cost"] + ok["commission"]).mean()),
        "sharpe": sharpe, "max_drawdown": float((cum - cum.cummax()).min()), "days_traded": int(len(daily)),
        "total_premium": float((100 * ok["entry"]).sum()),
    }


def bootstrap_ci(trades: pd.DataFrame, n_boot: int = 2000, block: int = 5, seed: int = 0) -> dict:
    """Moving-block bootstrap over exit days of the daily net P&L: 95% CI of the mean daily P&L and of Sharpe."""
    ok = trades[trades["status"] == "ok"] if len(trades) else trades
    if not len(ok):
        return {}
    daily = ok.groupby("exit_date")["net_pnl"].sum().sort_index().to_numpy()
    n = len(daily)
    rng = np.random.default_rng(seed)
    means, sharpes = [], []
    nb = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block + 1, 1), nb)
        s = np.concatenate([daily[a:a + block] for a in starts])[:n]
        means.append(s.mean())
        sharpes.append(s.mean() / s.std(ddof=1) * np.sqrt(252) if s.std(ddof=1) > 0 else np.nan)

    def ci(a):
        return [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]

    return {"mean_daily_pnl_ci": ci(means), "sharpe_ci": ci(sharpes)}
