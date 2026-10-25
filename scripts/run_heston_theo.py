"""Daily walk-forward Heston theo for one underlying.

PINN theo (direct, from a trained conditioned network):
    PYTHONPATH=src python scripts/run_heston_theo.py --symbol SPY --pricer pinn --model models/heston_cond/SPY_x.pt --tag x --mode full
FD theo (calibrate with the mid-grid finite-difference solver, reprice with the fine grid):
    PYTHONPATH=src python scripts/run_heston_theo.py --symbol SPY --pricer fd --tag fd --mode full --start 2025-10-16 --end 2026-01-31

--start/--end select a date range (rows from --warmup earlier dates are fitted but dropped), so a long
run can be split into chunks and run in parallel; chunk files are merged by scripts/merge_theo_chunks.py.
Writes data/dolthub/slices/theo_<symbol>_<tag>_<mode>[_<start>].csv.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from ivuq.calibration.heston_calib import FDPricer, PINNPricer, error_scale
from ivuq.pinn.heston_boxes import box_for
from ivuq.trading.theo_pipeline import run_theo


def load_model(path):
    blob = torch.load(path, weights_only=False)
    if blob.get("kind") == "coupled":
        from ivuq.pinn.heston_cond_coupled import HestonCoupledPINN
        return HestonCoupledPINN.load(path)
    from ivuq.pinn.heston_cond import HestonCondPINN
    return HestonCondPINN.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--pricer", default="pinn", choices=["pinn", "fd"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--mode", default="full", choices=["full", "slow"])
    ap.add_argument("--src", default="data/dolthub/slices/puts_clean.csv")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-days", type=int, default=0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-nfev", type=int, default=25)
    ap.add_argument("--refit-every", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = pd.read_csv(a.src, parse_dates=["date", "expiry"])
    d = d[d["symbol"] == a.symbol].reset_index(drop=True)
    dates = sorted(d["date"].unique())
    lo = 0 if a.start is None else max(0, int(pd.Index(dates).searchsorted(pd.Timestamp(a.start))))
    hi = len(dates) if a.end is None else int(pd.Index(dates).searchsorted(pd.Timestamp(a.end), side="right"))
    if a.max_days:
        hi = min(hi, lo + a.max_days)
    first = max(0, lo - a.warmup) if lo > 0 else 0
    warm = lo - first
    d = d[d["date"].isin(dates[first:hi])].reset_index(drop=True)
    d["scale"] = error_scale(d)
    box = box_for(a.symbol, wide=(a.pricer == "fd"))
    final = None
    if a.pricer == "pinn":
        pricer = PINNPricer(load_model(a.model))
        tag = a.tag or Path(a.model).stem
    else:
        pricer = FDPricer(box, grid="mid")
        final = FDPricer(box, grid="fine")
        tag = a.tag or "fd"
    out = run_theo(pricer, d, mode=a.mode, verbose=True, final_pricer=final, warmup_dates=warm, max_nfev=a.max_nfev,
                    refit_every=a.refit_every)
    suffix = "" if a.start is None else f"_{a.start}"
    path = Path(f"data/dolthub/slices/theo_{a.symbol}_{tag}_{a.mode}{suffix}.csv")
    out.to_csv(path, index=False)
    e = (out["theo"] - out["mid"]).abs()
    print(f"{path}: n={len(out)} MAE ${e.mean():.4f} median ${e.median():.4f} mean fit_rmse {out['fit_rmse'].mean():.2f}")


if __name__ == "__main__":
    main()
