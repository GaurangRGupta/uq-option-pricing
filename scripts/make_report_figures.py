"""Build every report figure and table from the study results.

    PYTHONPATH=src python scripts/make_report_figures.py

Needs the outputs of the study scripts (results/heston_trading/trades_*, conformal_diagnostics_*, random_control_*,
cond_eval_*), the theo files and puts_clean.csv under data/dolthub/slices/, and optionally the live signals under
data/live/signals/. Writes PNGs to results/heston_trading/figures/ and CSV tables beside them.
"""
import json
from pathlib import Path

import pandas as pd

from ivuq.pinn.heston_boxes import box_for
from ivuq.reporting import figures as F
from ivuq.reporting.analysis import ablation, cost_grid, load_trades, monthly
from ivuq.trading.online_conformal import add_rolling_conformal
from ivuq.trading.study import SLICES, load_panel

RES = Path("results/heston_trading")
OUT = RES / "figures"
SYMS = ["SPY", "AAPL", "NVDA"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    puts = pd.read_csv(SLICES / "puts_clean.csv", parse_dates=["date", "expiry"])

    # market: implied volatility surfaces
    for s in SYMS:
        F.iv_time_surface(puts, s, OUT)
        last = puts[(puts.symbol == s) & (puts.date == puts.date.max())]
        F.iv_term_surface(last, s, OUT, "dolthub_2026-10-21")
    live_f = sorted(Path("data/live/signals").glob("*_pinn.csv"))
    live = pd.read_csv(live_f[-1], parse_dates=["date", "expiry"]) if live_f else None
    if live is not None:
        for s in SYMS:
            F.iv_term_surface(live[live.symbol == s], s, OUT, f"yahoo_live_{live['date'].iloc[0].date()}")

    # network accuracy
    ev = {}
    for s in SYMS:
        for tag in ("cv_anch1000", "cpl1000"):
            ev[f"{s}_{tag}"] = json.load(open(RES / f"cond_eval_{s}_{tag}.json"))
    F.network_accuracy(ev, OUT)

    # model against market
    panel, _ = load_panel("pinn", SYMS)
    pc = add_rolling_conformal(panel, "theo_h_full", alpha=0.10, window=20, gamma=0.0)
    # add_rolling_conformal overwrites the dividend-yield column q with the conformal quantile: keep both apart
    band = pc[["date", "symbol", "expiry", "strike", "q"]].rename(columns={"q": "cq"})
    pb = panel.merge(band, on=["date", "symbol", "expiry", "strike"])
    pb["theo"] = pb["theo_h_full"]
    pb["half_width"] = pb["cq"] * pb["scale"]
    for s in SYMS:
        day = pb[(pb.symbol == s) & (pb.date == pb.date.max())]
        F.smile_fit(day, s, OUT, "dolthub_2026-10-21")
    F.price_error_map(panel, "theo_h_full", OUT, "error_map_heston.png", "Heston PINN: median |mid - theo| / c")
    F.price_error_map(panel, "theo_flat", OUT, "error_map_bs_flat.png", "Black-Scholes flat vol: median |mid - theo| / c")
    boxes = {}
    theo = {}
    for s in SYMS:
        b = box_for(s)
        boxes[s] = dict(kappa=b.kappa, theta=b.theta, xi=b.xi, rho=b.rho)
        theo[s] = pd.read_csv(SLICES / f"theo_{s}_pinn_full.csv", parse_dates=["date"])
    F.params_time(theo, boxes, OUT, "single_network")

    # trading
    tr = load_trades(RES)
    ab = ablation(tr)
    ab.to_csv(RES / "ablation_table.csv", index=False)
    cg = cost_grid(tr)
    cg.to_csv(RES / "cost_grid.csv", index=False)
    monthly(tr).to_csv(RES / "monthly_pnl.csv", index=False)
    F.ablation_bars(ab, OUT)
    for lag in (0, 1):
        F.equity_curves(tr, OUT, lag)
    F.lag_decay(ab, OUT)
    for t in ("pinn", "cpl1000"):
        F.breakdown(tr, OUT, t, 0)
        F.cost_heatmap(cg, OUT, t, 0)
        F.failure_panel(tr, OUT, t, 0)
    F.failure_panel(tr, OUT, "pinn", 1)
    ctrl = pd.concat([pd.read_csv(RES / f"random_control_{t}.csv").assign(tag=t) for t in ("pinn", "cpl1000", "fd")])
    F.random_vs_model(tr, ctrl, OUT)
    diag = pd.read_csv(RES / "conformal_diagnostics_pinn.csv")
    F.conformal_coverage(diag, OUT)
    slopes = F.gap_closure(panel, {"pinn": "theo_h_full", "flat": "theo_flat", "smile": "theo_smile"}, OUT)
    json.dump(slopes, open(RES / "gap_closure_slopes.json", "w"), indent=1)

    for w in ("best", "worst"):
        F.trade_example(tr, puts, OUT, "pinn", w)

    # live
    lt = {t: pd.read_csv(f"data/live/signals/{live['date'].iloc[0].date()}_{t}.csv") for t in ("pinn", "cpl1000", "fd")
          if live is not None and Path(f"data/live/signals/{live['date'].iloc[0].date()}_{t}.csv").exists()}
    if lt:
        F.live_tags(lt, OUT)
    if live is not None:
        F.signal_map(live, OUT, f"yahoo_{live['date'].iloc[0].date()}_pinn", "Live signals, Heston PINN with band")
        F.live_shift(puts, live, OUT)
    print("figures:", len(list(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
