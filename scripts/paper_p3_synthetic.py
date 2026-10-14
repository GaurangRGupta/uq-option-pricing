"""P3: synthetic conformal study (cheap, no PINN).

Same data-generating process as scripts/run_conformal_sandbox.py, run over
50 seeds and alpha in {0.10, 0.05}, with per-bucket coverage. See
results/uq/paper/PROTOCOL.md.

    python scripts/paper_p3_synthetic.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ivuq.uq import mondrian_split_conformal_interval, split_conformal_interval

from paper_common import PAPER, interval_metrics, summarise

SEEDS = list(range(50))
ALPHAS = [0.10, 0.05]


class Oracle:
    name = "oracle"

    def __init__(self, residual: bool) -> None:
        self.residual = residual

    def predict(self, m, tau):
        return _mean(np.asarray(m, float), np.asarray(tau, float))

    def constraint_residual(self, m, tau):
        return np.abs(np.asarray(m, float) - 1.0) if self.residual else None


def _mean(m, tau):
    return 0.08 + 0.25 * np.maximum(1.0 - m, 0.0) * np.sqrt(np.maximum(tau, 1e-6))


def _scale(m):
    return 0.01 + 0.12 * np.abs(m - 1.0)


def hetero(seed: int, alpha: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_cal, n_te = 800, 2500
    m_cal, tau_cal = rng.uniform(0.5, 1.5, n_cal), rng.uniform(0.08, 1.0, n_cal)
    m_te, tau_te = rng.uniform(0.5, 1.5, n_te), rng.uniform(0.08, 1.0, n_te)
    u_cal = _mean(m_cal, tau_cal) + rng.normal(0.0, 1.0, n_cal) * _scale(m_cal)
    u_te = _mean(m_te, tau_te) + rng.normal(0.0, 1.0, n_te) * _scale(m_te)
    pred = Oracle(residual=True)
    res = {
        "vanilla": split_conformal_interval(pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, kappa=0.0),
        "normalised": split_conformal_interval(pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, scale_kappa=True),
        "mondrian": mondrian_split_conformal_interval(pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, kappa=0.0),
    }
    rows = []
    far = np.abs(m_te - 1.0) > 0.25
    for name, r in res.items():
        d = dict(experiment="heteroscedastic", seed=seed, alpha=alpha, method=name, kappa=r.kappa, n_cal=n_cal, n_test=n_te)
        d.update(interval_metrics(r.lo, r.hi, u_te, m_te, tau_te))
        inside = (u_te >= r.lo) & (u_te <= r.hi)
        d["coverage_near_atm"] = float(inside[~far].mean())
        d["coverage_away_atm"] = float(inside[far].mean())
        d["width_near_atm"] = float(np.mean((r.hi - r.lo)[~far]))
        d["width_away_atm"] = float(np.mean((r.hi - r.lo)[far]))
        rows.append(d)
    return rows


def homo(seed: int, alpha: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    m_cal, tau_cal = rng.uniform(0.6, 1.4, 400), rng.uniform(0.1, 1.0, 400)
    m_te, tau_te = rng.uniform(0.6, 1.4, 1500), rng.uniform(0.1, 1.0, 1500)
    u_cal = _mean(m_cal, tau_cal) + rng.normal(0.0, 0.03, 400)
    u_te = _mean(m_te, tau_te) + rng.normal(0.0, 0.03, 1500)
    r = split_conformal_interval(Oracle(False), m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha)
    d = dict(experiment="homoscedastic", seed=seed, alpha=alpha, method="vanilla", kappa=0.0, n_cal=400, n_test=1500)
    d.update(interval_metrics(r.lo, r.hi, u_te, m_te, tau_te))
    return [d]


def main() -> None:
    t0 = time.perf_counter()
    rows = []
    for a in ALPHAS:
        for s in SEEDS:
            rows += hetero(s, a)
            rows += homo(s, a)
    raw = pd.DataFrame(rows)
    out = PAPER / "p3"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "p3_raw.csv", index=False)
    metrics = ["coverage", "mean_width", "worst_bucket_cov", "coverage_near_atm", "coverage_away_atm",
               "width_near_atm", "width_away_atm", "kappa"] + [f"cov_b{k}" for k in range(6)] + [f"width_b{k}" for k in range(6)]
    metrics = [c for c in metrics if c in raw.columns]
    summ = summarise(raw, ["experiment", "alpha", "method"], metrics)
    summ.to_csv(out / "p3_summary.csv", index=False)
    wall = time.perf_counter() - t0
    (out / "p3_meta.json").write_text(json.dumps({"seeds": SEEDS, "alphas": ALPHAS, "wall_seconds": wall}, indent=2))
    pd.set_option("display.width", 250)
    print(summ[["experiment", "alpha", "method", "n_seeds", "coverage_mean", "coverage_std", "mean_width_mean",
                "worst_bucket_cov_mean", "coverage_away_atm_mean", "coverage_near_atm_mean"]].to_string(index=False))
    print(f"wall {wall:.1f}s")


if __name__ == "__main__":
    main()
