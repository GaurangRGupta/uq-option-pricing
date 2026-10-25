"""Finite-difference reference for the Heston put, European or American.

Why it exists. The PINN needs a trustworthy reference with no Monte Carlo
noise (`heston_lsmc.py` is only good to a few cents) and the backtest needs a
classical numerical solver to compare the PINN's theo against. One solve
returns the whole surface u(m, v, tau) for every tau up to tau_max, so a
day's whole option chain is priced from a single run.

Equation. With x = ln m, m = S/K, u = V/K, and tau = T - t, the Heston PDE
(`pinn/heston_pde.py`, eq. (4)) becomes

    u_tau = 0.5 v (u_xx - u_x) + (r - q) u_x - r u                       [A1]
          + 0.5 xi^2 v u_vv + kappa (theta - v) u_v                      [A2]
          + rho xi v u_xv                                                [A0]

using m u_m = u_x, m^2 u_mm = u_xx - u_x, m u_mv = u_xv. The reaction term -r u
is split evenly between A1 and A2. The American put is the linear
complementarity problem

    min( u_tau - L[u],  u - h ) = 0,    h(x) = max(1 - e^x, 0)

solved here by operator splitting: take an unconstrained step of the PDE,
then project, u <- max(u, h). This is first order in the time step for the
early-exercise region, so the step is kept small; convergence is checked in
`tests/test_heston_fd.py` against the semi-analytic European price, the
Black-Scholes limit (xi -> 0) and a refined grid.

Scheme. Craig-Sneyd ADI (theta = 1/2) with the mixed term A0 explicit
(In 't Hout and Welfert, 2007). The first `n_rannacher` steps use theta = 1
(fully implicit in each direction), which removes the oscillation the payoff
kink at x = 0 would otherwise send through the second-order scheme. All
implicit matrices are time independent, so each is LU-factorised once.

Grid. x uniform with a node at x = 0 (the kink), v stretched near 0 where the
diffusion degenerates (v = v_max sinh(c s) / sinh(c)).

Boundaries.
  x_min: u = h (American) or e^{-r tau} - m e^{-q tau} (European)
  x_max: u = 0
  v = 0: the diffusion terms vanish, the PDE itself holds with the one-sided
         second-order derivative for u_v (no extra condition)
  v_max: u_vv = 0 (the same artificial condition the PINN uses)
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import splu

__all__ = ["HestonFDSurface", "heston_fd_surface"]


def _d1_d2_weights(z: np.ndarray):
    """Central 3-point first and second derivative weights on a non-uniform grid.
    Returns arrays (lo, mid, hi) for each, defined for interior nodes 1..n-2."""
    h_m = z[1:-1] - z[:-2]
    h_p = z[2:] - z[1:-1]
    d1 = (-h_p / (h_m * (h_m + h_p)), (h_p - h_m) / (h_m * h_p), h_m / (h_p * (h_m + h_p)))
    d2 = (2 / (h_m * (h_m + h_p)), -2 / (h_m * h_p), 2 / (h_p * (h_m + h_p)))
    return d1, d2


def _forward_d1(z: np.ndarray):
    """Second-order one-sided first derivative weights at node 0."""
    h1, h2 = z[1] - z[0], z[2] - z[1]
    return (-(2 * h1 + h2) / (h1 * (h1 + h2)), (h1 + h2) / (h1 * h2), -h1 / (h2 * (h1 + h2)))


class HestonFDSurface:
    """Solved surface. `price_u(m, v, tau)` interpolates; `price(S, K, v, tau)`
    is in dollars. tau is linearly interpolated between stored time steps."""

    def __init__(self, x, v, taus, U, K_unit: float = 1.0):
        self.x, self.v, self.taus, self.U = x, v, taus, U
        self._interp = [RegularGridInterpolator((x, v), U[k], method="linear") for k in range(len(taus))]

    def price_u(self, m, v, tau):
        m, v, tau = np.broadcast_arrays(np.atleast_1d(np.asarray(m, float)),
                                        np.atleast_1d(np.asarray(v, float)),
                                        np.atleast_1d(np.asarray(tau, float)))
        x = np.clip(np.log(np.clip(m, 1e-12, None)), self.x[0], self.x[-1])
        vv = np.clip(v, self.v[0], self.v[-1])
        out = np.empty(m.shape)
        k = np.clip(np.searchsorted(self.taus, tau) , 1, len(self.taus) - 1)
        for kk in np.unique(k):
            sel = k == kk
            t0, t1 = self.taus[kk - 1], self.taus[kk]
            w = np.clip((tau[sel] - t0) / (t1 - t0), 0.0, 1.0)
            pts = np.stack([x[sel], vv[sel]], axis=1)
            out[sel] = (1 - w) * self._interp[kk - 1](pts) + w * self._interp[kk](pts)
        return out

    def price(self, S, K, v, tau):
        S, K = np.asarray(S, float), np.asarray(K, float)
        return K * self.price_u(S / K, v, tau)

    def delta(self, S, K, v, tau, eps: float = 1e-2):
        """dV/dS by central difference of the interpolated surface. eps is a relative
        spot bump; it is kept near the grid spacing in x so the difference is not
        taken inside one linear-interpolation cell."""
        S, K = np.asarray(S, float), np.asarray(K, float)
        up = self.price(S * (1 + eps), K, v, tau)
        dn = self.price(S * (1 - eps), K, v, tau)
        return (up - dn) / (2 * eps * S)

    def vega_v0(self, S, K, v, tau, eps: float = 1e-3):
        """dV/dv0 (variance sensitivity) by central difference."""
        up = self.price(S, K, v + eps, tau)
        dn = self.price(S, K, np.maximum(v - eps, 0.0), tau)
        return (up - dn) / (eps + np.minimum(eps, v))


def heston_fd_surface(
    r: float, q: float, kappa: float, theta: float, xi: float, rho: float,
    tau_max: float = 0.30,
    american: bool = True,
    nx_neg: int = 90, nx_pos: int = 90, x_min: float = float(np.log(0.4)), x_max: float = float(np.log(2.0)),
    nv: int = 60, v_max: float = 1.0, v_stretch: float = 3.0,
    n_steps: int = 150, n_rannacher: int = 3,
) -> HestonFDSurface:
    """Solve the Heston put surface on [x_min, x_max] x [0, v_max] x [0, tau_max]."""
    # ---- grids
    x = np.concatenate([np.linspace(x_min, 0.0, nx_neg + 1)[:-1], np.linspace(0.0, x_max, nx_pos + 1)])
    s = np.linspace(0.0, 1.0, nv)
    v = v_max * np.sinh(v_stretch * s) / np.sinh(v_stretch)
    nx = len(x)
    X, V = np.meshgrid(x, v, indexing="ij")
    H = np.maximum(1.0 - np.exp(X), 0.0)

    # ---- 1D operators on each grid
    (dx1, dx2), (dv1, dv2) = _d1_d2_weights(x)[0:2], _d1_d2_weights(v)[0:2]
    xs, vs = x[1:-1], v[1:-1]

    # x-direction operator A1 acts on the first axis, one block per v node
    def build_A1():
        rows, cols, vals = [], [], []
        for j in range(nv):
            vj = v[j]
            diff = 0.5 * vj
            lo = diff * dx2[0] + (-diff + (r - q)) * dx1[0]
            mid = diff * dx2[1] + (-diff + (r - q)) * dx1[1] - 0.5 * r
            hi = diff * dx2[2] + (-diff + (r - q)) * dx1[2]
            i = np.arange(1, nx - 1)
            idx = i * nv + j
            rows += [idx, idx, idx]
            cols += [(i - 1) * nv + j, idx, (i + 1) * nv + j]
            vals += [lo, mid, hi]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * nv, nx * nv))

    def build_A2():
        rows, cols, vals = [], [], []
        f0 = np.array(_forward_d1(v))
        j = np.arange(1, nv - 1)
        diff = 0.5 * xi**2 * v[1:-1]
        drift = kappa * (theta - v[1:-1])
        for i in range(nx):
            base = i * nv
            # j = 0: no diffusion, drift kappa*theta with the one-sided u_v, and the -r/2 u reaction
            rows += [np.full(4, base)]
            cols += [np.array([base, base + 1, base + 2, base])]
            vals += [np.array([kappa * theta * f0[0] - 0.5 * r, kappa * theta * f0[1], kappa * theta * f0[2], 0.0])]
            idx = base + j
            rows += [idx, idx, idx]
            cols += [idx - 1, idx, idx + 1]
            vals += [diff * dv2[0] + drift * dv1[0], diff * dv2[1] + drift * dv1[1] - 0.5 * r, diff * dv2[2] + drift * dv1[2]]
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * nv, nx * nv))

    def build_A0():
        # rho xi v u_xv with central differences on both axes
        Dx = sp.lil_matrix((nx, nx)); Dv = sp.lil_matrix((nv, nv))
        for k, i in enumerate(range(1, nx - 1)):
            Dx[i, i - 1], Dx[i, i], Dx[i, i + 1] = dx1[0][k], dx1[1][k], dx1[2][k]
        for k, j in enumerate(range(1, nv - 1)):
            Dv[j, j - 1], Dv[j, j], Dv[j, j + 1] = dv1[0][k], dv1[1][k], dv1[2][k]
        f0 = _forward_d1(v)
        Dv[0, 0], Dv[0, 1], Dv[0, 2] = f0
        coef = sp.diags((rho * xi * np.tile(v, nx)))
        return coef @ sp.kron(sp.csr_matrix(Dx), sp.csr_matrix(Dv), format="csr")

    A1, A2, A0 = build_A1(), build_A2(), build_A0()
    # zero rows at x boundaries for A0 (Dirichlet nodes)
    Ntot = nx * nv
    I = sp.identity(Ntot, format="csr")

    # constraint row at v_max: u_{N-1} - 2 u_{N-2} + u_{N-3} = 0  (u_vv = 0 on non-uniform grid)
    hm, hp = v[-1] - v[-2], v[-2] - v[-3]
    # linear extrapolation: u_N = u_{N-1} + (u_{N-1}-u_{N-2}) * hm/hp
    # u_j - [(1+hm/hp) u_{j-1} - (hm/hp) u_{j-2}] = 0  -> coeffs on (j-2, j-1, j): (hm/hp, -(1+hm/hp), 1)
    c_last = np.array([hm / hp, -(1.0 + hm / hp), 1.0])
    last_nodes = np.arange(nx) * nv + (nv - 1)

    def with_constraint(M: sp.csr_matrix) -> sp.csr_matrix:
        M = M.tolil()
        for ln in last_nodes:
            M.rows[ln] = []; M.data[ln] = []
            M[ln, ln - 2], M[ln, ln - 1], M[ln, ln] = c_last
        return M.tocsr()

    # ---- implicit matrices for theta = 1/2 and theta = 1 (Rannacher)
    dt = tau_max / n_steps
    mats = {}
    for th in (0.5, 1.0):
        M1 = (I - th * dt * A1).tocsc()
        M2 = with_constraint((I - th * dt * A2).tocsr()).tocsc()
        mats[th] = (splu(M1), splu(M2))

    low_nodes = np.arange(nv)
    high_nodes = (nx - 1) * nv + np.arange(nv)

    def apply_bc(u: np.ndarray, tau: float) -> np.ndarray:
        u = u.copy()
        if american:
            u[low_nodes] = H.ravel()[low_nodes]
        else:
            m0 = np.exp(x[0])
            u[low_nodes] = np.exp(-r * tau) - m0 * np.exp(-q * tau)
        u[high_nodes] = 0.0
        return u

    def project(u: np.ndarray) -> np.ndarray:
        return np.maximum(u, H.ravel()) if american else u

    def step(U: np.ndarray, tau_new: float, th: float) -> np.ndarray:
        lu1, lu2 = mats[th]
        Y0 = U + dt * (A0 @ U + A1 @ U + A2 @ U)
        Y0 = apply_bc(Y0, tau_new)
        rhs1 = Y0 - th * dt * (A1 @ U)
        Y1 = lu1.solve(rhs1)
        Y1 = apply_bc(Y1, tau_new)
        rhs2 = Y1 - th * dt * (A2 @ U)
        rhs2[last_nodes] = 0.0
        Y2 = apply_bc(lu2.solve(rhs2), tau_new)
        if th == 0.5:
            Yh0 = Y0 + 0.5 * dt * (A0 @ Y2 - A0 @ U)
            Yh0 = apply_bc(Yh0, tau_new)
            rhs1 = Yh0 - th * dt * (A1 @ U)
            Yh1 = apply_bc(lu1.solve(rhs1), tau_new)
            rhs2 = Yh1 - th * dt * (A2 @ U)
            rhs2[last_nodes] = 0.0
            Y2 = lu2.solve(rhs2)
        return project(apply_bc(Y2, tau_new))

    U = H.ravel().copy()
    snaps = [U.reshape(nx, nv).copy()]
    taus = [0.0]
    for n in range(1, n_steps + 1):
        th = 1.0 if n <= n_rannacher else 0.5
        U = step(U, n * dt, th)
        snaps.append(U.reshape(nx, nv).copy())
        taus.append(n * dt)
    return HestonFDSurface(x, v, np.array(taus), np.stack(snaps))
