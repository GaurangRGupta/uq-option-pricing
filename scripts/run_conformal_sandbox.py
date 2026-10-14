"""Synthetic conformal experiments for the Review 2 paper.

Writes CSVs and a figure under results/uq/. Does not need sklearn.

  1. Oracle mean + homoscedastic noise: coverage should sit near 1 - alpha.
  2. Heteroscedastic noise: vanilla CP vs residual-normalised CP.
  3. Mondrian buckets on the same heteroscedastic draw.
  4. A small European PINN vs closed-form Black-Scholes (slow; skip with --skip-pinn).

Run from the repo root:

    python scripts/run_conformal_sandbox.py
    python scripts/run_conformal_sandbox.py --skip-pinn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.pricing.black_scholes import price as bs_price
from ivuq.uq import (
    PINNAdapter,
    empirical_coverage,
    mean_width,
    mondrian_split_conformal_interval,
    split_conformal_interval,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uq"
OUT.mkdir(parents=True, exist_ok=True)


class OracleMean:
    name = "oracle"

    def __init__(self, mean_fn, residual_fn=None) -> None:
        self.mean_fn = mean_fn
        self.residual_fn = residual_fn

    def predict(self, m, tau):
        return self.mean_fn(np.asarray(m, dtype=np.float64), np.asarray(tau, dtype=np.float64))

    def constraint_residual(self, m, tau):
        if self.residual_fn is None:
            return None
        return self.residual_fn(np.asarray(m, dtype=np.float64), np.asarray(tau, dtype=np.float64))


def _mean(m, tau):
    return 0.08 + 0.25 * np.maximum(1.0 - m, 0.0) * np.sqrt(np.maximum(tau, 1e-6))


def homoscedastic_trials(n_trials: int, alpha: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pred = OracleMean(_mean)
    rows = []
    for t in range(n_trials):
        m_cal = rng.uniform(0.6, 1.4, 400)
        tau_cal = rng.uniform(0.1, 1.0, 400)
        m_te = rng.uniform(0.6, 1.4, 1500)
        tau_te = rng.uniform(0.1, 1.0, 1500)
        u_cal = _mean(m_cal, tau_cal) + rng.normal(0.0, 0.03, 400)
        u_te = _mean(m_te, tau_te) + rng.normal(0.0, 0.03, 1500)
        res = split_conformal_interval(pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha)
        rows.append(
            {
                "trial": t,
                "alpha": alpha,
                "coverage": empirical_coverage(res.lo, res.hi, u_te),
                "mean_width": mean_width(res.lo, res.hi),
                "q": res.q,
            }
        )
    return pd.DataFrame(rows)


def heteroscedastic_compare(alpha: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_cal, n_te = 800, 2500
    m_cal = rng.uniform(0.5, 1.5, n_cal)
    tau_cal = rng.uniform(0.08, 1.0, n_cal)
    m_te = rng.uniform(0.5, 1.5, n_te)
    tau_te = rng.uniform(0.08, 1.0, n_te)

    def scale(m):
        return 0.01 + 0.12 * np.abs(m - 1.0)

    def residual(m, tau):
        return np.abs(m - 1.0)

    pred = OracleMean(_mean, residual)
    u_cal = _mean(m_cal, tau_cal) + rng.normal(0.0, 1.0, n_cal) * scale(m_cal)
    u_te = _mean(m_te, tau_te) + rng.normal(0.0, 1.0, n_te) * scale(m_te)

    vanilla = split_conformal_interval(pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, kappa=0.0)
    normalised = split_conformal_interval(
        pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, scale_kappa=True
    )
    mondrian = mondrian_split_conformal_interval(
        pred, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, kappa=0.0
    )

    def pack(name, res):
        far = np.abs(m_te - 1.0) > 0.25
        near = ~far
        finite = np.isfinite(res.lo) & np.isfinite(res.hi)
        return {
            "method": name,
            "kappa": res.kappa,
            "coverage": empirical_coverage(res.lo[finite], res.hi[finite], u_te[finite]),
            "mean_width": mean_width(res.lo[finite], res.hi[finite]),
            "coverage_near_atm": empirical_coverage(res.lo[near], res.hi[near], u_te[near]),
            "coverage_away_atm": empirical_coverage(res.lo[far], res.hi[far], u_te[far]),
            "width_near_atm": mean_width(res.lo[near], res.hi[near]),
            "width_away_atm": mean_width(res.lo[far], res.hi[far]),
            "n_infinite": int((~finite).sum()),
        }

    return pd.DataFrame(
        [
            pack("split_vanilla", vanilla),
            pack("split_normalised", normalised),
            pack("mondrian", mondrian),
        ]
    )


def pinn_vs_black_scholes(alpha: float, seed: int) -> pd.DataFrame:
    from ivuq.pinn import EuropeanPINN, EuropeanPINNConfig

    cfg = EuropeanPINNConfig(
        r=0.03,
        q=0.0,
        sigma=0.2,
        option_type="put",
        m_max=3.0,
        tau_max=1.0,
        hidden_layers=3,
        hidden_width=24,
        epochs=1500,
        n_interior=1000,
        n_terminal=200,
        n_boundary=100,
        lr=2e-3,
        lr_decay_every=700,
        seed=seed,
        lambda_pde=1.0,
        lambda_terminal=3.0,
    )
    pinn = EuropeanPINN(cfg).fit()
    adapter = PINNAdapter(pinn, K=100.0, name="n1_put")

    rng = np.random.default_rng(seed + 7)
    n_cal, n_te = 400, 800
    m_cal = rng.uniform(0.7, 1.3, n_cal)
    tau_cal = rng.uniform(0.15, 0.85, n_cal)
    m_te = rng.uniform(0.7, 1.3, n_te)
    tau_te = rng.uniform(0.15, 0.85, n_te)

    def truth(m, tau):
        return np.array(
            [bs_price(float(mi) * 100.0, 100.0, float(ti), 0.03, 0.0, 0.2, "put") / 100.0 for mi, ti in zip(m, tau)]
        )

    u_cal = truth(m_cal, tau_cal)
    u_te = truth(m_te, tau_te)
    vanilla = split_conformal_interval(adapter, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, kappa=0.0)
    normalised = split_conformal_interval(
        adapter, m_cal, tau_cal, u_cal, m_te, tau_te, alpha=alpha, scale_kappa=True
    )
    mae = float(np.mean(np.abs(vanilla.u_hat - u_te)))
    rows = []
    for name, res in (("pinn_split_vanilla", vanilla), ("pinn_split_normalised", normalised)):
        rows.append(
            {
                "method": name,
                "alpha": alpha,
                "kappa": res.kappa,
                "coverage": empirical_coverage(res.lo, res.hi, u_te),
                "mean_width_u": mean_width(res.lo, res.hi),
                "mean_width_usd": 100.0 * mean_width(res.lo, res.hi),
                "point_mae_usd": 100.0 * mae,
                "q": res.q if np.isscalar(res.q) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def maybe_figure(homo: pd.DataFrame, hetero: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=140)
    axes[0].hist(homo["coverage"], bins=10, color="0.35", edgecolor="white")
    axes[0].axvline(1.0 - homo["alpha"].iloc[0], color="0.0", linestyle="--", linewidth=1)
    axes[0].set_xlabel("empirical coverage")
    axes[0].set_ylabel("trials")
    axes[0].set_title("Oracle, homoscedastic, alpha=0.1")
    names = hetero["method"].tolist()
    widths = hetero["mean_width"].to_numpy()
    axes[1].bar(range(len(names)), widths, color="0.45", edgecolor="black", linewidth=0.6)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(["vanilla", "normalised", "Mondrian"], rotation=0)
    axes[1].set_ylabel("mean interval width (u units)")
    axes[1].set_title("Heteroscedastic noise")
    fig.tight_layout()
    fig.savefig(OUT / "conformal_sandbox.png", facecolor="white")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-pinn", action="store_true")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    homo = homoscedastic_trials(args.trials, args.alpha, args.seed)
    homo.to_csv(OUT / "homoscedastic_trials.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "experiment": "homoscedastic_oracle",
                "alpha": args.alpha,
                "n_trials": args.trials,
                "mean_coverage": homo["coverage"].mean(),
                "std_coverage": homo["coverage"].std(ddof=1),
                "mean_width": homo["mean_width"].mean(),
            }
        ]
    )

    hetero = heteroscedastic_compare(args.alpha, args.seed)
    hetero.to_csv(OUT / "heteroscedastic_compare.csv", index=False)

    maybe_figure(homo, hetero)

    if not args.skip_pinn:
        pinn = pinn_vs_black_scholes(args.alpha, args.seed)
        pinn.to_csv(OUT / "pinn_vs_bs.csv", index=False)
        print(pinn.to_string(index=False))

    print("homoscedastic mean coverage", float(homo["coverage"].mean()), "+/-", float(homo["coverage"].std(ddof=1)))
    print(hetero.to_string(index=False))
    summary.to_csv(OUT / "summary.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
