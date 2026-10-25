"""Out-of-sample test of the conformal trading rule.

The theos and the conformal band are walk-forward (every fit and every quantile uses past dates only), so
the only thing that can be tuned on the same data it is scored on is the settings of the rule itself.
Protocol:

  development = dates before --split, test = dates from --split on.
  For each theo variant and entry lag, pick (alpha, horizon, adaptive gamma) with the best development
  Sharpe (at least --min-trades trades), freeze it, and report the test-period result of that frozen choice
  with a block-bootstrap interval and the random-signal control (same trade counts per day).

    PYTHONPATH=src python scripts/run_oos_test.py --tag fd --split 2026-05-01

Writes results/heston_trading/oos_summary_<tag>.csv and oos_trades_<tag>_<variant>_lag<k>.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.trading.backtest import BacktestConfig, bootstrap_ci, build_signals, run_backtest, summarise
from ivuq.trading.controls import random_signals
from ivuq.trading.online_conformal import add_rolling_conformal
from ivuq.trading.study import SLICES, load_panel

OUT = Path("results/heston_trading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fd")
    ap.add_argument("--split", default="2026-05-01")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL", "NVDA"])
    ap.add_argument("--min-trades", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(a.split)
    panel, variants = load_panel(a.tag, a.symbols)
    div = pd.read_csv(SLICES / "raw" / "dividend.csv", parse_dates=["ex_date"])
    print(f"{len(panel)} quotes, {panel['date'].nunique()} dates, split {split.date()}, variants {[v for v, _ in variants]}", flush=True)

    rows = []
    for name, col in variants:
        mcol = {"h_full": "mdelta_full", "h_slow": "mdelta_slow"}.get(name, "delta")
        p = panel.copy()
        p["model_delta"] = p[mcol] if mcol in p else p["delta"]
        bands = {}
        for alpha in (0.05, 0.10, 0.20):
            for gamma in (0.0, 0.02):
                bands[(alpha, gamma)] = add_rolling_conformal(p, col, alpha=alpha, window=a.window, gamma=gamma)
        for lag in (0, 1):
            grid = []
            for (alpha, gamma), pc in bands.items():
                for H in (1, 3, 5):
                    cfg = BacktestConfig(rule="conformal", horizon=H, lag=lag, hedge="vendor")
                    tr = run_backtest(pc, col, cfg, dividends=div)
                    dev = tr[(tr["status"] == "ok") & (tr["date"] < split)]
                    sdev = summarise(dev) if len(dev) else {}
                    grid.append(dict(alpha=alpha, gamma=gamma, horizon=H, n_dev=len(dev),
                                     sharpe_dev=sdev.get("sharpe", np.nan), total_dev=sdev.get("total_net", np.nan), trades=tr))
            ok = [g for g in grid if g["n_dev"] >= a.min_trades and np.isfinite(g["sharpe_dev"])]
            best = max(ok, key=lambda g: g["sharpe_dev"]) if ok else None
            if best is None:
                rows.append(dict(variant=name, lag=lag, note="no development configuration with enough trades"))
                continue
            tr = best["trades"]
            test = tr[tr["date"] >= split]
            st = summarise(test)
            st.update({f"ci_{k}": v for k, v in bootstrap_ci(test).items()})
            pc = bands[(best["alpha"], best["gamma"])]
            cfg = BacktestConfig(rule="conformal", horizon=best["horizon"], lag=lag, hedge="vendor")
            tmpl = build_signals(pc, col, cfg)
            tmpl = tmpl[tmpl["date"] >= split]
            tots = []
            for sd in range(a.seeds):
                rs = random_signals(pc[pc["date"] >= split], tmpl, seed=sd)
                tots.append(summarise(run_backtest(pc, col, cfg, dividends=div, signals=rs)).get("total_net", np.nan))
            st.update(variant=name, lag=lag, alpha=best["alpha"], gamma=best["gamma"], horizon=best["horizon"],
                      sharpe_dev=best["sharpe_dev"], total_dev=best["total_dev"], n_dev=best["n_dev"],
                      control_total_mean=float(np.nanmean(tots)), control_total_p05=float(np.nanquantile(tots, 0.05)),
                      control_total_p95=float(np.nanquantile(tots, 0.95)))
            rows.append(st)
            test.to_csv(OUT / f"oos_trades_{a.tag}_{name}_lag{lag}.csv", index=False)
            keep = ("alpha", "gamma", "horizon", "n_trades", "total_net", "sharpe", "hit_rate", "control_total_mean")
            print(name, lag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items() if k in keep}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / f"oos_summary_{a.tag}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
