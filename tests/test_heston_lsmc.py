"""Sanity checks for the LSMC American-Heston reference pricer
(`ivuq.pricing.heston_lsmc.lsmc_american_heston_price`) -- the same role
`tests/test_binomial.py` plays for the CRR tree. Monte Carlo, so tolerances
are wider than the closed-form/tree tests elsewhere in this project; fixed
seeds keep them reproducible.
"""

from __future__ import annotations

import pytest

from ivuq.pricing.binomial import crr_price
from ivuq.pricing.heston import heston_price
from ivuq.pricing.heston_lsmc import lsmc_american_heston_price

_KW = dict(S0=100.0, K=100.0, T=1.0, r=0.03, q=0.0, n_steps=80, n_paths=15000, seed=0)


def test_zero_vol_of_vol_matches_crr_binomial_tree():
    """xi=0 with v0=theta=sigma^2 collapses Heston onto GBM (same limit
    `test_pinn_european_heston.py` checks on the PDE residual directly) --
    LSMC-Heston should land near the CRR American tree under that GBM,
    within Monte Carlo noise."""
    sigma = 0.2
    lsmc = lsmc_american_heston_price(
        **_KW, kappa=2.0, theta=sigma**2, xi=0.0, rho=-0.5, v0=sigma**2, option_type="put"
    )
    tree = crr_price(100.0, 100.0, 1.0, 0.03, 0.0, sigma, "put", is_american=True, steps=500)
    assert lsmc == pytest.approx(tree, abs=0.6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_at_least_european_heston(option_type):
    """American >= European under the same Heston dynamics -- the
    early-exercise right can never make an option less valuable."""
    kwargs = dict(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    american = lsmc_american_heston_price(**_KW, **kwargs, option_type=option_type)
    european = heston_price(100.0, 100.0, 1.0, 0.03, 0.0, kwargs["kappa"], kwargs["theta"], kwargs["xi"], kwargs["rho"], kwargs["v0"], option_type)
    # Loose slack for MC noise -- this must hold in expectation, not path-by-path.
    assert american >= european - 1.0


def test_deep_itm_american_put_exercises_early():
    kwargs = dict(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    american = lsmc_american_heston_price(
        S0=40.0, K=100.0, T=1.0, r=0.06, q=0.0, n_steps=80, n_paths=15000, seed=0, **kwargs, option_type="put"
    )
    european = heston_price(40.0, 100.0, 1.0, 0.06, 0.0, kwargs["kappa"], kwargs["theta"], kwargs["xi"], kwargs["rho"], kwargs["v0"], "put")
    assert american > european + 0.5


def test_zero_dividend_american_call_close_to_european():
    """Never optimal to exercise a call early with q=0, same fact
    `test_binomial.test_zero_dividend_american_call_equals_european_call`
    checks exactly for the tree -- LSMC only approximates it (regression
    noise, discretized exercise dates), so the tolerance is looser."""
    kwargs = dict(kappa=2.0, theta=0.04, xi=0.4, rho=-0.5, v0=0.04)
    american = lsmc_american_heston_price(**_KW, **kwargs, option_type="call")
    european = heston_price(100.0, 100.0, 1.0, 0.03, 0.0, kwargs["kappa"], kwargs["theta"], kwargs["xi"], kwargs["rho"], kwargs["v0"], "call")
    assert american == pytest.approx(european, abs=0.5)


def test_rejects_bad_option_type():
    with pytest.raises(ValueError):
        lsmc_american_heston_price(
            100, 100, 1, 0.03, 0.0, 2.0, 0.04, 0.4, -0.5, 0.04, "strangle", n_paths=1000
        )
