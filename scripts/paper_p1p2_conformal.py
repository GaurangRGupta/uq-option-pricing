"""P1 (European N1 and N0 put vs Black-Scholes) and P2 (American N2 put vs
500-step CRR): split / residual-normalised / Mondrian conformal intervals.

    python scripts/paper_p1p2_conformal.py

Needs results/uq/paper/models/*.pt (paper_train_pinns.py) and
results/uq/paper/pools/*.csv (paper_make_pools.py). Protocol: PROTOCOL.md.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ivuq.uq import mondrian_split_conformal_interval, split_conformal_interval

from paper_common import PAPER, interval_metrics, summarise
from paper_models import K, predictors

SEEDS = list(range(10))
SPLITS = list(range(20))
N_CALS = [400, 800]
N_TEST = 2000
ALPHAS = [0.10, 0.05]
MODELS = {"n1": "u_eu_bs", "n0": "u_eu_bs", "n2": "u_am_crr500"}


def run_one(model: str, seed: int) -> tuple[list[dict], dict]:
    pool = pd.read_csv(PAPER / "pools" / f"pool_seed{seed}.csv")
    m_all, tau_all, u_all = pool["m"].to_numpy(), pool["tau"].to_numpy(), pool[MODELS[model]].to_numpy()
    preds = predictors(model, seed)
    pde = preds["pde"]
    u_hat_all = pde.predict(m_all, tau_all)
    err = np.abs(u_hat_all - u_all)
    diag = {
        "model": model,
        "seed": seed,
        "pool_mae_usd": K * float(err.mean()),
        "pool_rmse_usd": K * float(np.sqrt(np.mean(err**2))),
        "pool_max_usd": K * float(err.max()),
        "spearman_abs_pde_resid_vs_abs_err": float(spearmanr(np.abs(pde.constraint_residual(m_all, tau_all)), err).statistic),
    }
    if "lcp" in preds:
        diag["spearman_abs_lcp_resid_vs_abs_err"] = float(
            spearmanr(np.abs(preds["lcp"].constraint_residual(m_all, tau_all)), err).statistic
        )
    rows = []
    for r in SPLITS:
        perm = np.random.default_rng(200000 + 100 * seed + r).permutation(len(pool))
        for n_cal in N_CALS:
            ci, ti = perm[:n_cal], perm[n_cal : n_cal + N_TEST]
            mc, tc, uc = m_all[ci], tau_all[ci], u_all[ci]
            mt, tt, ut = m_all[ti], tau_all[ti], u_all[ti]
            for alpha in ALPHAS:
                res = {
                    "vanilla": split_conformal_interval(pde, mc, tc, uc, mt, tt, alpha=alpha, kappa=0.0),
                    "normalised_pde": split_conformal_interval(pde, mc, tc, uc, mt, tt, alpha=alpha, scale_kappa=True),
                    "mondrian": mondrian_split_conformal_interval(pde, mc, tc, uc, mt, tt, alpha=alpha, kappa=0.0),
                }
                if "lcp" in preds:
                    res["normalised_lcp"] = split_conformal_interval(
                        preds["lcp"], mc, tc, uc, mt, tt, alpha=alpha, scale_kappa=True
                    )
                mae = K * float(np.mean(np.abs(res["vanilla"].u_hat - ut)))
                for name, rr in res.items():
                    d = dict(model=model, seed=seed, split=r, n_cal=n_cal, n_test=N_TEST, alpha=alpha, method=name,
                             kappa=rr.kappa, q_usd=K * rr.q if np.isscalar(rr.q) else np.nan, test_mae_usd=mae)
                    d.update(interval_metrics(rr.lo, rr.hi, ut, mt, tt, scale=K))
                    rows.append(d)
    return rows, diag


def main() -> None:
    torch.set_num_threads(4)
    out = PAPER / "p1p2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows, diags, timings = [], [], []
    for model in MODELS:
        for s in SEEDS:
            t1 = time.perf_counter()
            r, d = run_one(model, s)
            rows += r
            diags.append(d)
            timings.append({"model": model, "seed": s, "eval_seconds": time.perf_counter() - t1})
            print(f"{model} seed {s} done", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "p1p2_raw.csv", index=False)
    train = []
    for model in MODELS:
        for s in SEEDS:
            meta = json.loads((PAPER / "models" / f"{model}_seed{s}.json").read_text())
            train.append({"model": model, "seed": s, "train_wall_seconds": meta["wall_seconds"], **{f"final_loss_{k}": v for k, v in meta["final_losses"].items()}})
    diag = pd.DataFrame(diags).merge(pd.DataFrame(train), on=["model", "seed"]).merge(pd.DataFrame(timings), on=["model", "seed"])
    diag.to_csv(out / "p1p2_model_diagnostics.csv", index=False)
    metrics = ["coverage", "mean_width", "median_width", "worst_bucket_cov", "test_mae_usd", "kappa", "q_usd", "n_infinite"] + \
        [f"cov_b{k}" for k in range(6)] + [f"width_b{k}" for k in range(6)]
    summ = summarise(raw, ["model", "n_cal", "alpha", "method"], metrics)
    summ.to_csv(out / "p1p2_summary.csv", index=False)
    dcols = [c for c in diag.columns if c not in ("model", "seed")]
    dsum = diag.groupby("model")[dcols].agg(["mean", "std"])
    dsum.columns = [f"{a}_{b}" for a, b in dsum.columns]
    dsum.reset_index().to_csv(out / "p1p2_model_diagnostics_summary.csv", index=False)
    # Split-level spread (all seeds x splits) for the coverage histogram.
    spread = raw.groupby(["model", "n_cal", "alpha", "method"])["coverage"].agg(["mean", "std", "min", "max", "count"]).reset_index()
    spread.to_csv(out / "p1p2_coverage_split_spread.csv", index=False)
    wall = time.perf_counter() - t0
    (out / "p1p2_meta.json").write_text(json.dumps({"seeds": SEEDS, "splits": SPLITS, "n_cals": N_CALS, "n_test": N_TEST,
                                                     "alphas": ALPHAS, "eval_wall_seconds": wall}, indent=2))
    pd.set_option("display.width", 250)
    print(summ[["model", "n_cal", "alpha", "method", "coverage_mean", "coverage_std", "mean_width_mean", "worst_bucket_cov_mean", "test_mae_usd_mean"]].to_string(index=False))
    print(dsum.reset_index().to_string())
    print(f"wall {wall:.1f}s")


if __name__ == "__main__":
    main()
