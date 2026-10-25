"""Backtest every theo variant on the same quotes and write the summary tables.

Needs, from earlier steps (not committed, under data/dolthub/slices/):
  puts_clean.csv                         scripts/pull_dolthub.py
  theo_<SYM>_<tag>_full.csv / _slow.csv  scripts/run_heston_theo.py   (Heston-PINN theos)
  bs_theos.csv                           scripts/run_bs_theos.py      (BS flat and smile theos)

    PYTHONPATH=src python scripts/run_heston_study.py --tag phys

Variants are compared on the common set of quotes: every variant has a theo there, burn-in
dates of the slow mode are removed for all of them. Output (results/heston_trading/):
  backtest_summary_<tag>.csv    one row per (variant, rule, horizon, lag)
  random_control_<tag>.csv      same trade counts, random contracts and sides, 30 seeds
  conformal_diagnostics_<tag>.csv   realised miss rate of the band by symbol / month / bucket
  trades_<tag>_<variant>_<rule>_h<H>_l<lag>.csv   executed trades for the headline settings
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.trading.study import KEY, SLICES, load_panel
from ivuq.trading.backtest import BacktestConfig, bootstrap_ci, build_signals, run_backtest, summarise
from ivuq.trading.controls import random_signals
from ivuq.trading.online_conformal import add_rolling_conformal

OUT = Path("results/heston_trading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phys")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL", "NVDA"])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    panel, variants = load_panel(a.tag, a.symbols)
    div = pd.read_csv(SLICES / "raw" / "dividend.csv", parse_dates=["ex_date"])
    print(f"common quotes: {len(panel)} over {panel['date'].nunique()} dates; variants: {[v for v, _ in variants]}", flush=True)

    rows, diag, ctrl = [], [], []
    for name, col in variants:
        mcol = {"h_full": "mdelta_full", "h_slow": "mdelta_slow"}.get(name, "delta")
        p = panel.copy()
        p["model_delta"] = p[mcol] if mcol in p else p["delta"]
        for gamma, label in ((0.0, "conformal"), (0.02, "conformal_aci")):
            pc = add_rolling_conformal(p, col, alpha=a.alpha, window=a.window, gamma=gamma)
            fin = pc[np.isfinite(pc["q"])]
            for key, g in fin.groupby([fin["symbol"], fin["date"].dt.to_period("M"), fin["bucket"]]):
                diag.append(dict(variant=name, band=label, symbol=key[0], month=str(key[1]), bucket=key[2], n=len(g),
                                 miss_rate=float((g["score"] > g["q"]).mean())))
            for rule, k, pp in (("conformal", 0, pc), ("spread", 0, pc), ("fixed_k", 1.0, pc), ("fixed_k", 2.0, pc)):
                if label == "conformal_aci" and rule != "conformal":
                    continue
                rname = label if rule == "conformal" else (rule if rule == "spread" else f"fixed_k{k:g}")
                for H in (1, 3, 5):
                    for lag in (0, 1):
                        cfg = BacktestConfig(rule=rule, k=k, horizon=H, lag=lag, hedge="vendor")
                        tr = run_backtest(pp, col, cfg, dividends=div)
                        s = summarise(tr)
                        s.update(variant=name, rule=rname, horizon=H, lag=lag, hedge="vendor")
                        if H == 3 and rname in ("conformal", "spread"):
                            s.update({f"ci_{kk}": vv for kk, vv in bootstrap_ci(tr).items()})
                            tr.to_csv(OUT / f"trades_{a.tag}_{name}_{rname}_h{H}_l{lag}.csv", index=False)
                        rows.append(s)
                        print(name, rname, H, lag, s.get("n_trades"), round(s.get("total_net", 0.0), 1), flush=True)
                if rule == "conformal" and label == "conformal":
                    for lag in (0, 1):
                        cfg = BacktestConfig(rule="conformal", horizon=3, lag=lag, hedge="vendor")
                        tmpl = build_signals(pp, col, cfg)
                        tots = []
                        for sd in range(a.seeds):
                            rs = random_signals(pp, tmpl, seed=sd)
                            tots.append(summarise(run_backtest(pp, col, cfg, dividends=div, signals=rs)))
                        ctrl.append(dict(variant=name, lag=lag, n_seeds=a.seeds,
                                         total_net_mean=float(np.mean([t.get("total_net", np.nan) for t in tots])),
                                         total_net_p05=float(np.nanquantile([t.get("total_net", np.nan) for t in tots], 0.05)),
                                         total_net_p95=float(np.nanquantile([t.get("total_net", np.nan) for t in tots], 0.95)),
                                         mean_net_mean=float(np.mean([t.get("mean_net", np.nan) for t in tots]))))
        pd.DataFrame(rows).to_csv(OUT / f"backtest_summary_{a.tag}.csv", index=False)
        pd.DataFrame(ctrl).to_csv(OUT / f"random_control_{a.tag}.csv", index=False)
        pd.DataFrame(diag).to_csv(OUT / f"conformal_diagnostics_{a.tag}.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
