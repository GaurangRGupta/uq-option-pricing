"""Heston closed-form pricer: sanity checks against Black-Scholes (the
xi->0 limit) and basic no-arbitrage bounds. This is the reference price
N1-Heston will be validated against, the same role test_black_scholes.py
plays for N1.
"""

from __future__ import annotations

import numpy as np
import pytest

from ivuq.pricing.black_scholes import price as bs_price
from ivuq.pricing.heston import heston_price

_S, _K, _T, _R, _Q, _SIGMA = 100.0, 100.0, 1.0, 0.03, 0.0, 0.2
_KAPPA, _RHO = 2.0, -0.5


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_heston_recovers_black_scholes_as_vol_of_vol_vanishes(option_type: str) -> None:
    """As xi->0 with v0=theta=sigma^2, variance stops moving and Heston must
    collapse onto plain constant-volatility GBM -- i.e. the Black-Scholes
    price."""
    bs = bs_price(_S, _K, _T, _R, _Q, _SIGMA, option_type)
    heston = heston_price(
        _S, _K, _T, _R, _Q, kappa=_KAPPA, theta=_SIGMA**2, xi=1e-4, rho=_RHO, v0=_SIGMA**2,
        option_type=option_type,
    )
    assert heston == pytest.approx(bs, abs=0.01), f"Heston({option_type}) xi->0 should match BS within a cent"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_heston_price_within_no_arbitrage_bounds(option_type: str) -> None:
    price = heston_price(
        _S, _K, _T, _R, _Q, kappa=_KAPPA, theta=0.06, xi=0.6, rho=_RHO, v0=0.05, option_type=option_type,
    )
    intrinsic = max(_S - _K, 0.0) if option_type == "call" else max(_K - _S, 0.0)
    upper = _S if option_type == "call" else _K
    assert intrinsic - 1e-6 <= price <= upper, f"Heston {option_type} price {price:.4f} outside no-arbitrage bounds"


def test_heston_put_call_parity() -> None:
    call = heston_price(_S, _K, _T, _R, _Q, kappa=_KAPPA, theta=0.06, xi=0.6, rho=_RHO, v0=0.05, option_type="call")
    put = heston_price(_S, _K, _T, _R, _Q, kappa=_KAPPA, theta=0.06, xi=0.6, rho=_RHO, v0=0.05, option_type="put")
    assert call - put == pytest.approx(_S - _K * np.exp(-_R * _T), abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_higher_initial_variance_increases_price(option_type: str) -> None:
    """More uncertainty about the underlying's future path should never make
    an option cheaper (all else equal) -- a basic monotonicity sanity check,
    independent of the exact numerical value."""
    low = heston_price(_S, _K, _T, _R, _Q, kappa=_KAPPA, theta=0.04, xi=0.4, rho=_RHO, v0=0.02, option_type=option_type)
    high = heston_price(_S, _K, _T, _R, _Q, kappa=_KAPPA, theta=0.04, xi=0.4, rho=_RHO, v0=0.20, option_type=option_type)
    assert high > low, f"Heston {option_type} price should increase with initial variance: {low:.4f} vs {high:.4f}"
