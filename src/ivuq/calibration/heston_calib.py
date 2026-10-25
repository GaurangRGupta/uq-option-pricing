"""Fit Heston (kappa, theta, xi, rho, v0) to a day of American-put quotes.

Objective. For quotes i with mid P_i, half-spread s_i, and model price
P_i(p) = K_i u(m_i, v0_{d(i)}, tau_i; kappa, theta, xi, rho, r_i, q_i):

    min over p   sum_i  rho_c( (P_i(p) - mid_i) / c_i )  +  lam * || (p - p_prior) / w ||^2

    c_i = max( s_i , 0.003 * vega_i )            error scale in dollars
    rho_c = soft-L1 with f_scale = 2             robust loss

c_i is the half-spread, floored at 0.3 vol points of Black-Scholes vega, so a
residual is measured in units of "how far outside the market's own uncertainty".
The soft-L1 loss keeps a few mispriced quotes (the ones the strategy later
trades) from dragging the fit. p_prior (previous day's fit) and the box bounds
tame the ill-posed directions of the Heston calibration (kappa vs xi, theta vs
v0). `day_idx` lets one call fit a shared (kappa, theta, xi, rho) with one v0
per day, which the walk-forward uses to estimate the slowly varying parameters
from a window of past days.

Pricers: `PINNPricer` (the conditioned PINN, fast, batched) and `FDPricer`
(classical finite differences, one solve per distinct (r, q), the benchmark).
Both expose `price_u(m, v, tau, P)` with P of shape (n, 6) =
(kappa, theta, xi, rho, r, q).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ivuq.pinn.heston_cond import HestonCondPINN, ParamBox
from ivuq.pricing import black_scholes

__all__ = ["CalibResult", "PINNPricer", "FDPricer", "calibrate", "error_scale"]

GLOBAL = ("kappa", "theta", "xi", "rho")


class PINNPricer:
    name = "pinn"

    def __init__(self, pinn: HestonCondPINN) -> None:
        self.pinn = pinn
        self.box: ParamBox = pinn.config.box

    def price_u(self, m, v, tau, P) -> np.ndarray:
        return self.pinn.price_u(m, v, tau, P)

    def delta_u(self, m, v, tau, P) -> np.ndarray:
        """du/dm = dV/dS, by autograd on the network."""
        return self.pinn.delta(m, np.ones_like(np.asarray(m, float)), v, tau, P)


class FDPricer:
    """Classical finite-difference pricer (`pricing/heston_fd.py`), the accurate reference.

    One surface solve covers every strike, expiry and variance level for a given parameter set, so a
    day's whole chain costs one solve per distinct (r, q) pair. r and q are rounded to 5 basis points
    for the solve (an error of a few hundredths of a cent per $100 strike). Surfaces are cached by
    parameters, so the delta bump and the final repricing reuse the same solve.
    grid "mid": about 0.3 cent mean error per $100 strike, for calibration iterations;
    grid "fine": about 0.1 cent, for the final theo."""

    name = "fd"
    GRIDS = {"mid": dict(nx_neg=70, nx_pos=70, nv=40, n_steps=80),
             "fine": dict(nx_neg=110, nx_pos=110, nv=60, n_steps=150)}

    def __init__(self, box: ParamBox, grid: str = "mid", m_lo: float = 0.6, m_hi: float = 1.6, tau_max: float = 0.2) -> None:
        self.box, self.grid, self.m_lo, self.m_hi, self.tau_max = box, grid, m_lo, m_hi, tau_max
        self._cache: dict = {}

    def _surface(self, kappa, theta, xi, rho, r, q):
        from ivuq.pricing.heston_fd import heston_fd_surface

        key = (round(kappa, 6), round(theta, 6), round(xi, 6), round(rho, 6), r, q)
        if key not in self._cache:
            if len(self._cache) > 24:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = heston_fd_surface(
                r, q, kappa, theta, xi, rho, tau_max=self.tau_max, american=True, x_min=float(np.log(self.m_lo)),
                x_max=float(np.log(self.m_hi)), v_max=self.box.v_max, **self.GRIDS[self.grid])
        return self._cache[key]

    def price_u(self, m, v, tau, P) -> np.ndarray:
        m, v, tau, P = np.asarray(m, float), np.asarray(v, float), np.asarray(tau, float), np.asarray(P, float)
        out = np.empty(len(m))
        rk = np.round(P[:, 4] / 5e-4).astype(int)
        qk = np.round(P[:, 5] / 5e-4).astype(int)
        keys = rk * 1000 + qk
        for k in np.unique(keys):
            sel = keys == k
            kappa, theta, xi, rho = P[sel][0, :4]
            surf = self._surface(kappa, theta, xi, rho, rk[sel][0] * 5e-4, qk[sel][0] * 5e-4)
            out[sel] = surf.price_u(np.clip(m[sel], self.m_lo * 1.001, self.m_hi * 0.999), v[sel], tau[sel])
        return out


def _fd_delta_u(self, m, v, tau, P, eps: float = 1e-2) -> np.ndarray:
    m = np.asarray(m, float)
    return (self.price_u(m * (1 + eps), v, tau, P) - self.price_u(m * (1 - eps), v, tau, P)) / (2 * eps * m)


FDPricer.delta_u = _fd_delta_u


@dataclass
class CalibResult:
    params: dict          # kappa, theta, xi, rho (shared)
    v0: np.ndarray        # one per day in the call
    cost: float
    rmse_scaled: float    # RMS of (model - mid) / c_i
    rmse_price: float     # dollars
    n: int
    success: bool
    n_eval: int


def error_scale(df) -> np.ndarray:
    """c_i in dollars: half-spread floored at 0.3 vol points of BS vega."""
    vega = np.array([
        black_scholes.vega(s, k, t, r, q, iv)
        for s, k, t, r, q, iv in zip(df["spot"], df["strike"], df["tau"], df["r"], df["q"], df["vendor_iv"])
    ])
    return np.maximum(0.5 * df["spread"].to_numpy(), 0.003 * vega)


def calibrate(
    pricer,
    df,
    p0: dict,
    v0_0,
    day_idx: np.ndarray | None = None,
    free: tuple = GLOBAL + ("v0",),
    prior: dict | None = None,
    prior_weight: float = 0.0,
    f_scale: float = 2.0,
    max_nfev: int = 60,
    scale: np.ndarray | None = None,
) -> CalibResult:
    """Fit the `free` subset of (kappa, theta, xi, rho, v0). Parameters not in `free`
    stay at their p0 / v0_0 values. `df` needs spot, strike, tau, m, mid, spread, r, q, vendor_iv."""
    box: ParamBox = pricer.box
    n = len(df)
    day_idx = np.zeros(n, dtype=int) if day_idx is None else np.asarray(day_idx, dtype=int)
    n_days = int(day_idx.max()) + 1
    v0_0 = np.broadcast_to(np.asarray(v0_0, float), (n_days,)).copy()

    lo_g = np.array([getattr(box, k)[0] for k in GLOBAL]); hi_g = np.array([getattr(box, k)[1] for k in GLOBAL])
    v_lo, v_hi = 0.002, 0.9 * box.v_max
    free_g = [k for k in GLOBAL if k in free]
    fit_v0 = "v0" in free

    def unpack(z):
        g = {k: p0[k] for k in GLOBAL}
        for j, k in enumerate(free_g):
            i = GLOBAL.index(k)
            g[k] = lo_g[i] + z[j] * (hi_g[i] - lo_g[i])
        v0 = v0_0.copy()
        if fit_v0:
            v0 = v_lo + z[len(free_g):] * (v_hi - v_lo)
        return g, v0

    z0 = [(p0[k] - lo_g[GLOBAL.index(k)]) / (hi_g[GLOBAL.index(k)] - lo_g[GLOBAL.index(k)]) for k in free_g]
    if fit_v0:
        z0 += list((v0_0 - v_lo) / (v_hi - v_lo))
    z0 = np.clip(np.array(z0), 1e-4, 1 - 1e-4)

    m, tau, mid, K = (df[c].to_numpy(float) for c in ("m", "tau", "mid", "strike"))
    rq = df[["r", "q"]].to_numpy(float)
    c = error_scale(df) if scale is None else scale
    if prior is not None:
        z_prior = np.array([(prior[k] - lo_g[GLOBAL.index(k)]) / (hi_g[GLOBAL.index(k)] - lo_g[GLOBAL.index(k)]) for k in free_g])

    counter = {"n": 0}

    def resid(z):
        counter["n"] += 1
        g, v0 = unpack(z)
        P = np.column_stack([np.full(n, g["kappa"]), np.full(n, g["theta"]), np.full(n, g["xi"]), np.full(n, g["rho"]), rq])
        model = K * pricer.price_u(m, v0[day_idx], tau, P)
        res = (model - mid) / c
        if prior is not None and prior_weight > 0 and len(free_g):
            res = np.concatenate([res, prior_weight * (z[: len(free_g)] - z_prior)])
        return res

    sol = least_squares(resid, z0, bounds=(0.0, 1.0), loss="soft_l1", f_scale=f_scale,
                        max_nfev=max_nfev, diff_step=1e-3, x_scale=1.0)
    g, v0 = unpack(sol.x)
    P = np.column_stack([np.full(n, g["kappa"]), np.full(n, g["theta"]), np.full(n, g["xi"]), np.full(n, g["rho"]), rq])
    model = K * pricer.price_u(m, v0[day_idx], tau, P)
    return CalibResult(params=g, v0=v0, cost=float(sol.cost),
                       rmse_scaled=float(np.sqrt(np.mean(((model - mid) / c) ** 2))),
                       rmse_price=float(np.sqrt(np.mean((model - mid) ** 2))),
                       n=n, success=bool(sol.success), n_eval=counter["n"])
