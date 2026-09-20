"""Correctness checks for the ML/DL baselines and the scoring harness.

Fast by design (small n, fast=True budgets). These verify the machinery, not
the research result: the actual PINN-vs-baseline numbers come from
scripts/run_baseline_comparison.py at full training budgets.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ivuq.baselines import (
    FULL_BOX,
    MarketParams,
    evaluate,
    make_baselines,
    reference_u,
    run_regime,
    sample_labelled,
)
from ivuq.baselines.torch_models import PICNN
from ivuq.pricing.black_scholes import price as bs_price
from ivuq.pricing.binomial import crr_price

_P = MarketParams()
_M = np.arange(0.6, 1.8001, 0.1)
_TAU = np.linspace(0.1, 0.9, 5)
_MM, _TT = np.meshgrid(_M, _TAU)
_M_FLAT, _TAU_FLAT = _MM.ravel(), _TT.ravel()


def test_european_labels_match_black_scholes() -> None:
    X, u = sample_labelled("european", "call", 20, FULL_BOX, _P, seed=1)
    for (m, tau), ui in zip(X, u):
        assert ui == pytest.approx(bs_price(m, 1.0, tau, _P.r, _P.q, _P.sigma, "call"))


def test_american_labels_match_tree_and_exceed_european() -> None:
    m, tau = np.array([0.8, 1.0]), np.array([0.5, 0.5])
    amer = reference_u("american", "put", m, tau, _P)
    euro = reference_u("european", "put", m, tau, _P)
    assert amer[0] == pytest.approx(crr_price(0.8, 1.0, 0.5, _P.r, _P.q, _P.sigma, "put", True))
    assert np.all(amer >= euro - 1e-12)


def test_metrics_are_zero_on_the_exact_black_scholes_price() -> None:
    """Validates the finite-difference Greeks, the PDE residual and the
    violation checks: the true price must score (numerically) perfect."""

    def exact(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return reference_u("european", "put", m, tau, _P)

    res = evaluate(
        exact, _M_FLAT, _TAU_FLAT, exact(_M_FLAT, _TAU_FLAT), style="european", option_type="put", params=_P
    )
    assert res["mae_usd"] < 1e-9
    assert res["delta_mae"] < 1e-3
    # Central-difference gamma has O(h^2) truncation error, largest where gamma
    # peaks sharply (short maturity), so this is looser than delta.
    assert res["gamma_mae"] < 2e-2
    assert res["pde_rms"] < 5e-3
    assert res["viol_convex"] == 0.0 and res["viol_delta"] == 0.0 and res["viol_bound"] == 0.0


def test_metrics_flag_a_broken_price() -> None:
    """A price surface that ignores maturity and is concave in m must be
    caught by the structure checks."""

    def broken(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return -0.2 * (m - 1.0) ** 2 + 0.05

    u_ref = reference_u("european", "put", _M_FLAT, _TAU_FLAT, _P)
    res = evaluate(broken, _M_FLAT, _TAU_FLAT, u_ref, style="european", option_type="put", params=_P)
    assert res["viol_convex"] > 0.9
    assert res["mae_usd"] > 1.0


def test_picnn_is_convex_in_moneyness_by_construction() -> None:
    torch.manual_seed(0)
    net = PICNN()
    m = torch.linspace(0.3, 2.5, 200)
    for tau in (0.05, 0.4, 0.95):
        x = torch.stack([m, torch.full_like(m, tau)], dim=1)
        with torch.no_grad():
            u = net(x).squeeze()
        second_diff = u[2:] - 2 * u[1:-1] + u[:-2]
        assert float(second_diff.min()) > -1e-5, "untrained PICNN must already be convex in m"


@pytest.mark.parametrize("name", ["poly_ridge", "kernel_ridge", "random_forest", "grad_boosting", "mlp_tanh", "resnet_silu", "picnn", "bs_iv_net"])
def test_baselines_learn_european_put(name: str) -> None:
    X, u = sample_labelled("european", "put", 300, FULL_BOX, _P, seed=0)
    model = make_baselines(seed=0, fast=True, option_type="put", params=_P)[name].fit(X, u)
    pred = model.predict(np.column_stack([_M_FLAT, _TAU_FLAT]))
    truth = reference_u("european", "put", _M_FLAT, _TAU_FLAT, _P)
    mae_usd = 100 * float(np.mean(np.abs(pred - truth)))
    # Loose: fast budgets. A model that learned nothing misses by ~$10 on a $100 strike.
    assert mae_usd < 4.0, f"{name} mean abs error ${mae_usd:.2f}"


def test_run_regime_end_to_end_with_an_external_predictor() -> None:
    def exact(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return reference_u("european", "put", m, tau, _P)

    df = run_regime(
        "extrapolate",
        "european",
        "put",
        _P,
        n_list=[60],
        seeds=[0],
        external={"exact": (exact, 0.0)},
        fast=True,
    )
    assert set(df["model"]) >= {"exact", "poly_ridge", "picnn", "bs_iv_net", "kernel_ridge"}
    assert (df[df.model == "exact"]["mae_usd"] < 1e-9).all()
    assert (df["mae_outside_usd"].notna()).all()
    # Outside the narrow training box a polynomial fit should do worse than inside it.
    poly = df[df.model == "poly_ridge"].iloc[0]
    assert poly["mae_outside_usd"] > poly["mae_inside_usd"]


def test_exercise_boundary_of_the_tree_is_sensible_and_scores_zero_against_itself() -> None:
    from ivuq.baselines.metrics import BOUNDARY_M, BOUNDARY_TAU, exercise_boundary

    mm, tt = np.meshgrid(BOUNDARY_M, BOUNDARY_TAU)
    u = reference_u("american", "put", mm.ravel(), tt.ravel(), _P, max_workers=1).reshape(mm.shape)
    ref = exercise_boundary(lambda m, tau: u.ravel()[: m.size])
    assert not np.isnan(ref).any()
    # The put's exercise boundary lies below the strike and falls as maturity grows.
    assert np.all(ref < 1.0) and np.all(np.diff(ref) <= 1e-9)

    res = evaluate(
        lambda m, tau: reference_u("american", "put", m, tau, _P, max_workers=1),
        _M_FLAT, _TAU_FLAT, reference_u("american", "put", _M_FLAT, _TAU_FLAT, _P, max_workers=1),
        style="american", option_type="put", params=_P, boundary_ref=ref,
    )
    assert res["boundary_mae_usd"] < 1e-9 and res["boundary_missing"] == 0.0


def test_american_iv_recovers_the_true_vol_from_the_reference_price() -> None:
    from ivuq.baselines.metrics import IV_M, IV_TAU, american_iv

    im, it = np.meshgrid(IV_M, IV_TAU)
    u = reference_u("american", "put", im.ravel(), it.ravel(), _P, max_workers=1)
    active = u > (1.0 - im.ravel()) + 1e-3
    iv = american_iv(lambda m, tau: u, _P, active=active)
    assert active.sum() >= 10
    # 200-step inversion of a 500-step price: agree to well under one vol point.
    assert np.nanmax(np.abs(iv[active] - _P.sigma)) < 0.01
