"""Run the PINN vs ML/DL comparison and write results to disk.

  python scripts/run_baseline_comparison.py --out results/baselines
  python scripts/run_baseline_comparison.py --fast --styles european --regimes full   # smoke test

Trains each PINN once per (style, option_type) with no labelled prices, then
sweeps the number of labels given to every ML/DL baseline in each regime (see
ivuq.baselines.experiments). One CSV per (style, option type, regime) is
written as soon as it finishes, so a long run can be stopped and resumed
without losing completed parts; existing CSVs are skipped unless --force.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import argparse
import time
from pathlib import Path

import pandas as pd

from ivuq.baselines import MarketParams, pinn_predictor, run_regime
from ivuq.pinn import AmericanPINN, AmericanPINNConfig, EuropeanPINN, EuropeanPINNConfig

_FAST_PINN = dict(hidden_layers=3, hidden_width=24, epochs=1500, n_interior=1000, n_terminal=200, n_boundary=100, lr_decay_every=700)


def train_pinns(style: str, option_type: str, params: MarketParams, fast: bool) -> dict:
    kw = dict(r=params.r, q=params.q, sigma=params.sigma, **(_FAST_PINN if fast else {}))
    out = {}
    if style == "european":
        for name, lam in (("pinn_n1", 1.0), ("pinn_n0", 0.0)):
            t0 = time.perf_counter()
            model = EuropeanPINN(EuropeanPINNConfig(option_type=option_type, lambda_pde=lam, **kw)).fit()
            out[name] = (pinn_predictor(model), time.perf_counter() - t0)
            print(f"  trained {name} in {out[name][1]:.0f}s", flush=True)
    else:
        t0 = time.perf_counter()
        model = AmericanPINN(AmericanPINNConfig(option_type=option_type, weighting_scheme="fixed", **kw)).fit()
        out["pinn_n2"] = (pinn_predictor(model), time.perf_counter() - t0)
        print(f"  trained pinn_n2 in {out['pinn_n2'][1]:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--fast", action="store_true", help="small training budgets, for smoke tests only")
    ap.add_argument("--styles", nargs="+", default=["american", "european"], help="American first: it is the project focus")
    ap.add_argument("--option-types", nargs="+", default=["put", "call"])
    ap.add_argument("--regimes", nargs="+", default=["full", "extrapolate", "noisy"])
    ap.add_argument("--n-list", nargs="+", type=int, default=[50, 200, 1000, 3000])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = MarketParams()

    for style in args.styles:
        for option_type in args.option_types:
            if style == "american" and option_type == "call" and params.q == 0.0:
                continue  # with q = 0 an American call equals the European call: no early exercise to study
            todo = [g for g in args.regimes if args.force or not (out_dir / f"{style}_{option_type}_{g}.csv").exists()]
            if not todo:
                continue
            print(f"[{style} {option_type}] training PINNs", flush=True)
            pinns = train_pinns(style, option_type, params, args.fast)
            for regime in todo:
                t0 = time.perf_counter()
                df = run_regime(
                    regime, style, option_type, params,
                    n_list=args.n_list, seeds=list(range(args.seeds)), external=pinns, fast=args.fast,
                )
                df.to_csv(out_dir / f"{style}_{option_type}_{regime}.csv", index=False)
                print(f"[{style} {option_type} {regime}] done in {time.perf_counter() - t0:.0f}s", flush=True)

    frames = [pd.read_csv(p) for p in sorted(out_dir.glob("*_*_*.csv")) if p.name != "all.csv"]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "all.csv", index=False)
        print(f"combined results: {out_dir / 'all.csv'}")


if __name__ == "__main__":
    main()
