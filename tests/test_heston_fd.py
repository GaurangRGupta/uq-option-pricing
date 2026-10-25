"""Heston finite-difference reference (`ivuq.pricing.heston_fd`).

Checks the solver against things that are independently known: the
semi-analytic European price, the Black-Scholes early-exercise premium in the
xi -> 0 limit, and basic no-arbitrage structure of the American surface.
Tolerances are in dollars per $100 strike.
"""

import numpy as np
import pytest

from ivuq.pricing.binomial import crr_price
from ivuq.pricing.heston import heston_price
from ivuq.pricing.heston_fd import heston_fd_surface

CASES = [
    # r, q, kappa, theta, xi, rho
    (0.03, 0.0, 2.0, 0.04, 0.4, -0.5),
    (0.04, 0.01, 1.0, 0.06, 0.6, -0.7),
    (0.04, 0.0, 3.0, 0.03, 0.3, -0.3),
]


@pytest.mark.parametrize("r,q,kappa,theta,xi,rho", CASES)
def test_european_matches_semi_analytic(r, q, kappa, theta, xi, rho):
    s = heston_fd_surface(r, q, kappa, theta, xi, rho, american=False)
    worst = 0.0
    for m in (0.85, 1.0, 1.15):
        for tau in (0.1, 0.25):
            for v0 in (0.02, theta, 0.1):
                fd = 100.0 * s.price_u(m, v0, tau)[0]
                ref = heston_price(100 * m, 100.0, tau, r, q, kappa, theta, xi, rho, v0, "put")
                worst = max(worst, abs(fd - ref))
    assert worst < 0.02, worst


def test_american_premium_matches_black_scholes_limit():
    # xi tiny: variance is frozen near theta, so the early-exercise premium
    # (American minus European) should match the CRR premium at sigma = sqrt(theta).
    r, q, kappa, theta, xi, rho = 0.04, 0.01, 1.0, 0.04, 0.02, -0.5
    am = heston_fd_surface(r, q, kappa, theta, xi, rho, american=True)
    eu = heston_fd_surface(r, q, kappa, theta, xi, rho, american=False)
    for m, tau in [(0.95, 0.25), (1.0, 0.25), (1.05, 0.25)]:
        prem_fd = 100 * (am.price_u(m, theta, tau)[0] - eu.price_u(m, theta, tau)[0])
        prem_crr = crr_price(100 * m, 100, tau, r, q, 0.2, "put", True, 2000) - crr_price(100 * m, 100, tau, r, q, 0.2, "put", False, 2000)
        assert abs(prem_fd - prem_crr) < 0.01, (m, prem_fd, prem_crr)


def test_american_structure():
    r, q, kappa, theta, xi, rho = 0.04, 0.0, 2.0, 0.05, 0.5, -0.6
    am = heston_fd_surface(r, q, kappa, theta, xi, rho, american=True)
    eu = heston_fd_surface(r, q, kappa, theta, xi, rho, american=False)
    m = np.linspace(0.7, 1.3, 25)
    for tau in (0.05, 0.2):
        ua = am.price_u(m, 0.05, np.full_like(m, tau))
        ue = eu.price_u(m, 0.05, np.full_like(m, tau))
        assert np.all(ua >= np.maximum(1 - m, 0) - 1e-4)        # above the obstacle (up to interpolation between nodes)
        assert np.all(ua >= ue - 1e-6)                           # American >= European
        assert np.all(np.diff(ua) <= 1e-6)                       # put decreasing in m
        assert np.all(np.diff(ua, 2) >= -1e-4)                   # convex in m
    # delta of a put is in [-1, 0]
    d = am.delta(100 * m, 100.0, 0.05, 0.2)
    assert np.all(d <= 1e-3) and np.all(d >= -1.01)
