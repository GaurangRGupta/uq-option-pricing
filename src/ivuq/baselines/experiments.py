"""The PINN vs ML/DL comparison harness.

Three regimes, each answering a different question about why one would choose
a PINN over a data-driven model:

  full         Labels cover the whole trading domain. Is the PINN even
               competitive on plain accuracy when the baselines have data?
  extrapolate  Labels cover only a narrow box around the money and short
               maturities (`EXTRAPOLATION_BOX`); the models are then scored on
               the full grid. Data-driven models have nothing pinning them down
               outside the box; the PINN's PDE holds everywhere.
  noisy        Labels carry additive noise (`NOISY_LABELS_SD` in strike units,
               i.e. $0.50 on a $100 strike, the size of a wide bid-ask
               spread). Noise hurts models that fit labels; the PINN uses none.

For each regime the baselines are trained on n labelled examples for several
n, and over several seeds. The PINNs are trained once, with no labels, so
their score is a horizontal line against n. The number of labels a baseline
needs to match the PINN is the sample-efficiency result.

Each PINN is passed in as a `predict_u` callable (see `pinn_predictor`).
"""

from __future__ import annotations

import time
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from ivuq.baselines.data import Box, MarketParams, add_noise, reference_u, sample_labelled
from ivuq.baselines.metrics import BOUNDARY_M, BOUNDARY_TAU, IV_M, IV_TAU, american_iv, evaluate, exercise_boundary
from ivuq.baselines.models import FAMILY, make_baselines

__all__ = [
    "FULL_BOX",
    "EXTRAPOLATION_BOX",
    "NOISY_LABELS_SD",
    "TEST_M",
    "TEST_TAU",
    "pinn_predictor",
    "run_regime",
]

# Same interior evaluation grid as the PINN test-suite (S in 60..180 on a $100
# strike, tau in 0.1..0.9), refined to a step of 0.05 in moneyness.
TEST_M = np.arange(0.6, 1.8001, 0.05)
TEST_TAU = np.linspace(0.1, 0.9, 9)

FULL_BOX = Box(m_lo=0.4, m_hi=2.4, tau_lo=0.02, tau_hi=1.0)
EXTRAPOLATION_BOX = Box(m_lo=0.85, m_hi=1.15, tau_lo=0.02, tau_hi=0.5)
NOISY_LABELS_SD = 0.005

Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]


def pinn_predictor(pinn, K: float = 100.0) -> Predictor:
    """Adapt a fitted EuropeanPINN / AmericanPINN, which price in real
    (S, K, tau) units, to the (m, tau) -> u callable the harness uses."""

    def predict_u(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return np.asarray(pinn.price(m * K, K, tau), dtype=np.float64) / K

    return predict_u


def run_regime(
    regime: str,
    style: str,
    option_type: str,
    params: MarketParams,
    *,
    n_list: list[int],
    seeds: list[int],
    external: Mapping[str, tuple[Predictor, float]] | None = None,
    fast: bool = False,
    tree_steps: int = 500,
    max_workers: int | None = None,
) -> pd.DataFrame:
    """One row per (model, n_train, seed). `external` maps a name such as
    "pinn_n1" to (predict_u, training_seconds); those rows carry n_train = 0."""
    if regime not in ("full", "extrapolate", "noisy"):
        raise ValueError(f"unknown regime {regime!r}")
    train_box = EXTRAPOLATION_BOX if regime == "extrapolate" else FULL_BOX
    noise = NOISY_LABELS_SD if regime == "noisy" else 0.0

    mm, tt = np.meshgrid(TEST_M, TEST_TAU)
    m_test, tau_test = mm.ravel(), tt.ravel()
    u_test = reference_u(style, option_type, m_test, tau_test, params, tree_steps, max_workers)

    boundary_ref = iv_ref = None
    if style == "american" and option_type == "put":
        bm, bt = np.meshgrid(BOUNDARY_M, BOUNDARY_TAU)
        u_b = reference_u(style, option_type, bm.ravel(), bt.ravel(), params, tree_steps, max_workers).reshape(bm.shape)
        boundary_ref = exercise_boundary(lambda m, tau: u_b.ravel()[: m.size])
        # Reference IV: invert the 500-step reference price; drop points inside the
        # exercise region (price at intrinsic value), where vol has no effect.
        im, it = np.meshgrid(IV_M, IV_TAU)
        u_iv = reference_u(style, option_type, im.ravel(), it.ravel(), params, tree_steps, max_workers)
        in_ex = u_iv <= (1.0 - im.ravel()) + 1e-3
        iv_ref = american_iv(lambda m, tau: u_iv, params, active=~in_ex)

    def score(predict_u: Predictor) -> dict[str, float]:
        return evaluate(
            predict_u, m_test, tau_test, u_test, style=style, option_type=option_type, params=params, train_box=train_box,
            boundary_ref=boundary_ref,
            iv_ref=iv_ref,
        )

    common = dict(regime=regime, style=style, option_type=option_type, label_noise=noise)
    rows: list[dict] = []

    for name, (predict_u, train_seconds) in (external or {}).items():
        rows.append(
            dict(common, model=name, family="pinn", n_train=0, seed=0, fit_seconds=train_seconds, **score(predict_u))
        )

    n_max = max(n_list)
    for seed in seeds:
        X_pool, u_pool = sample_labelled(style, option_type, n_max, train_box, params, seed, tree_steps, max_workers)
        u_pool = add_noise(u_pool, noise, seed)
        for n in n_list:
            X, u = X_pool[:n], u_pool[:n]
            for name, model in make_baselines(seed=seed, fast=fast, option_type=option_type, params=params).items():
                t0 = time.perf_counter()
                model.fit(X, u)
                fit_seconds = time.perf_counter() - t0

                def predict_u(m: np.ndarray, tau: np.ndarray, _model=model) -> np.ndarray:
                    return _model.predict(np.column_stack([m, tau]))

                rows.append(
                    dict(common, model=name, family=FAMILY[name], n_train=n, seed=seed, fit_seconds=fit_seconds, **score(predict_u))
                )
    return pd.DataFrame(rows)
