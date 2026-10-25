"""Validate the Heston finite-difference reference (pricing/heston_fd.py).

(1) European put vs the semi-analytic price over a random parameter box.
(2) American put vs Longstaff-Schwartz Monte Carlo (a lower bound with
    sampling noise of a few cents) at a handful of points.
Writes results/heston_trading/fd_validation.csv. Run from the repo root.
"""
import numpy as np
import pandas as pd

from ivuq.pricing.heston import heston_price
from ivuq.pricing.heston_fd import heston_fd_surface
from ivuq.pricing.heston_lsmc import lsmc_american_heston_price

rng = np.random.default_rng(0)
rows = []
for _ in range(25):
    r, q = rng.uniform(0.02, 0.05), rng.uniform(0.0, 0.02)
    kappa, theta = rng.uniform(0.5, 5.0), rng.uniform(0.02, 0.2)
    xi, rho = rng.uniform(0.2, 1.0), rng.uniform(-0.9, -0.1)
    s = heston_fd_surface(r, q, kappa, theta, xi, rho, american=False, v_max=max(1.0, 5 * theta))
    for m, tau, v0 in [(0.9, 0.1, theta), (1.0, 0.2, 0.5 * theta), (1.1, 0.25, 1.5 * theta), (1.0, 0.05, theta)]:
        fd = 100 * s.price_u(m, v0, tau)[0]
        ref = heston_price(100 * m, 100.0, tau, r, q, kappa, theta, xi, rho, v0, "put")
        rows.append(dict(kind="european", err=fd - ref, ref=ref, feller=2 * kappa * theta / xi**2))
eu = pd.DataFrame(rows)
print("European FD vs semi-analytic ($ per 100 strike): mean abs %.4f  p95 %.4f  max %.4f" % (
    eu.err.abs().mean(), eu.err.abs().quantile(0.95), eu.err.abs().max()))

am = []
for (r, q, kappa, theta, xi, rho) in [(0.04, 0.01, 2.0, 0.04, 0.4, -0.5), (0.04, 0.0, 1.5, 0.08, 0.6, -0.7)]:
    s = heston_fd_surface(r, q, kappa, theta, xi, rho, american=True)
    for m, tau in [(0.9, 0.2), (1.0, 0.2), (1.0, 0.08), (1.1, 0.25)]:
        fd = 100 * s.price_u(m, theta, tau)[0]
        ls = [100 * lsmc_american_heston_price(100 * m, 100.0, tau, r, q, kappa, theta, xi, rho, theta, "put",
                                               n_steps=100, n_paths=100000, seed=k) / 100 for k in range(4)]
        am.append(dict(kind="american", m=m, tau=tau, fd=fd, lsmc=np.mean(ls), lsmc_se=np.std(ls, ddof=1) / 2))
am = pd.DataFrame(am)
am["fd_minus_lsmc"] = am.fd - am.lsmc
print(am.round(4).to_string())
pd.concat([eu, am]).to_csv("results/heston_trading/fd_validation.csv", index=False)
