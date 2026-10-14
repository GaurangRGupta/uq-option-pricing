"""P4: run the existing static-arbitrage checker (ivuq.arbitrage.checker) on
PINN point prices and on conformal interval edges (lo, hi).

Synthetic put chain, S = 100, strikes 60..140 step 2.5, expiries
tau = days/365 with days = round(365 * {0.1, ..., 1.0}). Intervals: split r = 0,
n_cal = 800, alpha = 0.10, calibrated on the same pools as P1/P2. Put-only, so
put-call parity is not applicable. No projection or repair is applied.

    python scripts/paper_p4_arbitrage.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from ivuq.arbitrage.checker import (
    calendar_spread_violations,
    strike_convexity_violations,
    strike_monotonicity_violations,
)
from ivuq.baselines.data import MarketParams, reference_u
from ivuq.uq import mondrian_split_conformal_interval, split_conformal_interval

from paper_common import PAPER
from paper_models import predictors

SEEDS = list(range(10))
S0 = 100.0
STRIKES = np.arange(60.0, 140.001, 2.5)
DAYS = np.array([int(round(365 * t)) for t in np.linspace(0.1, 1.0, 10)])
QUOTE = pd.Timestamp("2026-01-01")
ALPHA, N_CAL, N_TEST, SPLIT = 0.10, 800, 2000, 0
TOLS = [1e-6, 0.01]
MODELS = {"n1": ("european", "u_eu_bs"), "n0": ("european", "u_eu_bs"), "n2": ("american", "u_am_crr500")}


def chain_grid():
    kk, dd = np.meshgrid(STRIKES, DAYS)
    K, days = kk.ravel(), dd.ravel()
    return K, days, S0 / K, days / 365.0


def check(prices: np.ndarray, K: np.ndarray, days: np.ndarray) -> list[dict]:
    df = pd.DataFrame({
        "underlying": "SYN", "option_type": "put", "strike": K,
        "expiry_date": QUOTE + pd.to_timedelta(days, unit="D"), "quote_date": QUOTE,
        "market_option_price": prices,
    })
    n_exp, n_k = len(DAYS), len(STRIKES)
    denom = {"strike_monotonicity": n_exp * (n_k - 1), "strike_convexity": n_exp * (n_k - 2), "calendar_spread": n_k * (n_exp - 1)}
    out = []
    for tol in TOLS:
        counts = {
            "strike_monotonicity": len(strike_monotonicity_violations(df, tol=tol)),
            "strike_convexity": len(strike_convexity_violations(df, tol=tol)),
            "calendar_spread": len(calendar_spread_violations(df, tol=tol)),
        }
        for k, c in counts.items():
            out.append({"tol": tol, "check": k, "violations": c, "checkable": denom[k], "rate": c / denom[k]})
    out_neg = float(np.mean(prices < 0))
    for d in out:
        d["frac_negative_price"] = out_neg
    return out


def main() -> None:
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    K, days, m_c, tau_c = chain_grid()
    params = MarketParams()
    refs = {style: K * reference_u(style, "put", m_c, tau_c, params, tree_steps=500) for style in ("european", "american")}
    rows = []
    for style, ref in refs.items():
        for d in check(ref, K, days):
            rows.append(dict(model=f"reference_{style}", seed=-1, method="reference", edge="point", **d))
    for model, (style, col) in MODELS.items():
        for s in SEEDS:
            pool = pd.read_csv(PAPER / "pools" / f"pool_seed{s}.csv")
            perm = np.random.default_rng(200000 + 100 * s + SPLIT).permutation(len(pool))
            ci = perm[:N_CAL]
            mc, tc, uc = pool["m"].to_numpy()[ci], pool["tau"].to_numpy()[ci], pool[col].to_numpy()[ci]
            preds = predictors(model, s)
            pde = preds["pde"]
            res = {
                "vanilla": split_conformal_interval(pde, mc, tc, uc, m_c, tau_c, alpha=ALPHA, kappa=0.0),
                "normalised_pde": split_conformal_interval(pde, mc, tc, uc, m_c, tau_c, alpha=ALPHA, scale_kappa=True),
                "mondrian": mondrian_split_conformal_interval(pde, mc, tc, uc, m_c, tau_c, alpha=ALPHA, kappa=0.0),
            }
            if "lcp" in preds:
                res["normalised_lcp"] = split_conformal_interval(preds["lcp"], mc, tc, uc, m_c, tau_c, alpha=ALPHA, scale_kappa=True)
            for d in check(K * res["vanilla"].u_hat, K, days):
                rows.append(dict(model=model, seed=s, method="pinn", edge="point", **d))
            for name, rr in res.items():
                for edge, arr in (("lo", rr.lo), ("hi", rr.hi)):
                    for d in check(K * arr, K, days):
                        rows.append(dict(model=model, seed=s, method=name, edge=edge, **d))
            print(f"{model} seed {s}", flush=True)
    raw = pd.DataFrame(rows)
    out = PAPER / "p4"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "p4_raw.csv", index=False)
    g = raw.groupby(["model", "method", "edge", "tol", "check"])
    summ = g.agg(n_seeds=("seed", "nunique"), rate_mean=("rate", "mean"), rate_std=("rate", "std"),
                 violations_mean=("violations", "mean"), checkable=("checkable", "first"),
                 frac_negative_price_mean=("frac_negative_price", "mean")).reset_index()
    summ.to_csv(out / "p4_summary.csv", index=False)
    wall = time.perf_counter() - t0
    (out / "p4_meta.json").write_text(json.dumps({"seeds": SEEDS, "alpha": ALPHA, "n_cal": N_CAL, "split": SPLIT,
                                                   "strikes": STRIKES.tolist(), "days": DAYS.tolist(), "tols": TOLS,
                                                   "wall_seconds": wall}, indent=2))
    pd.set_option("display.width", 250)
    print(summ[summ.tol == 1e-6].to_string(index=False))
    print(f"wall {wall:.1f}s")


if __name__ == "__main__":
    main()
