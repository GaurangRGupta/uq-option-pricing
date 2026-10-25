"""FD labels and held-out evaluation for the parameter-conditioned Heston PINN.

`fd_labels` draws parameter sets from the box, solves the American put with
`pricing/heston_fd.py` (one solve per set, all points read off the surface) and
returns (inputs, u) tensors for the optional anchor term. `evaluate` compares a
trained PINN with fresh FD solves on parameter sets it never saw, restricted
to the region the market data occupies (m in [0.8, 1.25], tau in [0.02, 0.2],
v around theta). Errors are in dollars per $100 strike.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from ivuq.pricing.heston_fd import heston_fd_surface

from .heston_cond import HestonCondPINN, ParamBox

__all__ = ["fd_surface_for", "fd_labels", "fd_labels_coupled", "evaluate"]


def fd_surface_for(box: ParamBox, p):
    kappa, theta, xi, rho, r, q = (float(a) for a in p)
    return heston_fd_surface(r, q, kappa, theta, xi, rho, tau_max=box.tau_max, american=True,
                             x_min=float(np.log(box.m_lo)), x_max=float(np.log(box.m_hi)),
                             nx_neg=110, nx_pos=110, nv=60, v_max=box.v_max, n_steps=150)


def _label_job(args):
    box, p, n_pts, seed = args
    rng = np.random.default_rng(seed)
    s = fd_surface_for(box, p)
    theta = p[1]
    n_band = int(0.7 * n_pts)
    m = np.concatenate([rng.uniform(0.7, 1.4, n_band), rng.uniform(box.m_lo, box.m_hi, n_pts - n_band)])
    ratio = np.exp(rng.uniform(np.log(0.1), np.log(3.0), n_pts))
    v = np.where(rng.random(n_pts) < 0.7, np.minimum(theta * ratio, box.v_max), rng.uniform(0, box.v_max, n_pts))
    tau = box.tau_max * rng.random(n_pts) ** 1.3
    u = s.price_u(m, v, tau)
    X = np.column_stack([m, v, tau, np.tile(p, (n_pts, 1))]).astype(np.float32)
    return X, u.astype(np.float32)


def fd_labels(box: ParamBox, n_params: int, pts_per_param: int, seed: int = 0, workers: int = 8):
    rng = np.random.default_rng(seed)
    lo, hi = box.lo_hi()
    P = lo + (hi - lo) * rng.random((n_params, len(lo)))
    jobs = [(box, P[i], pts_per_param, seed * 100003 + i) for i in range(n_params)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_label_job, jobs, chunksize=4))
    X = torch.tensor(np.concatenate([r[0] for r in res]))
    y = torch.tensor(np.concatenate([r[1] for r in res])).reshape(-1, 1)
    return X, y


def _eval_job(args):
    box, p, n_pts, seed = args
    rng = np.random.default_rng(seed)
    s = fd_surface_for(box, p)
    theta = p[1]
    m = rng.uniform(0.8, 1.25, n_pts)
    v = np.minimum(theta * np.exp(rng.uniform(np.log(0.3), np.log(2.5), n_pts)), box.v_max)
    tau = rng.uniform(0.02, 0.2, n_pts)
    return p, m, v, tau, s.price_u(m, v, tau)


def evaluate(pinn: HestonCondPINN, n_params: int = 40, n_pts: int = 200, seed: int = 12345, workers: int = 8) -> dict:
    box = pinn.config.box
    rng = np.random.default_rng(seed)
    lo, hi = box.lo_hi()
    P = lo + (hi - lo) * rng.random((n_params, len(lo)))
    jobs = [(box, P[i], n_pts, seed + i) for i in range(n_params)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_eval_job, jobs, chunksize=2))
    err, atm, otm = [], [], []
    for p, m, v, tau, u_fd in res:
        u = pinn.price_u(m, v, tau, p)
        e = 100.0 * (u - u_fd)
        err.append(e)
        atm.append(np.abs(m - 1.0) < 0.05)
    err, atm = np.concatenate(err), np.concatenate(atm)
    a = np.abs(err)
    return {"mae": float(a.mean()), "p95": float(np.quantile(a, 0.95)), "max": float(a.max()),
            "bias": float(err.mean()), "mae_atm": float(a[atm].mean()), "mae_wings": float(a[~atm].mean()),
            "n": int(len(err))}


# ---------------------------------------------------------------------------
# labels for the coupled (solution + boundary) model
# ---------------------------------------------------------------------------

def _coupled_job(args):
    """Continuation-region price labels and exercise-boundary labels from one FD surface."""
    box, p, n_u, n_b, seed = args
    rng = np.random.default_rng(seed)
    s = fd_surface_for(box, p)
    theta = p[1]

    def draw_v(n):
        ratio = np.exp(rng.uniform(np.log(0.1), np.log(3.0), n))
        return np.where(rng.random(n) < 0.7, np.minimum(theta * ratio, box.v_max), rng.uniform(0, box.v_max, n))

    n_band = int(0.7 * n_u)
    m = np.concatenate([rng.uniform(0.7, 1.4, n_band), rng.uniform(box.m_lo, box.m_hi, n_u - n_band)])
    v, tau = draw_v(n_u), box.tau_max * rng.random(n_u) ** 1.3
    u = s.price_u(m, v, tau)
    keep = u - np.maximum(1.0 - m, 0.0) > 1e-5                    # continuation region only
    Xu = np.column_stack([m, v, tau, np.tile(p, (n_u, 1))])[keep].astype(np.float32)

    vb, taub = draw_v(n_b), box.tau_max * rng.random(n_b) ** 1.3
    grid = np.linspace(box.m_lo, 1.0, 241)
    bfd = np.empty(n_b)
    for i in range(n_b):
        ug = s.price_u(grid, np.full_like(grid, vb[i]), np.full_like(grid, taub[i]))
        ex = ug - np.maximum(1.0 - grid, 0.0) <= 1e-5
        bfd[i] = grid[ex].max() if ex.any() else grid[0]          # largest exercised moneyness
    Xb = np.column_stack([vb, taub, np.tile(p, (n_b, 1))]).astype(np.float32)
    return Xu, u[keep].astype(np.float32), Xb, bfd.astype(np.float32)


def fd_labels_coupled(box: ParamBox, n_params: int, n_u: int, n_b: int, seed: int = 0, workers: int = 8):
    rng = np.random.default_rng(seed)
    lo, hi = box.lo_hi()
    P = lo + (hi - lo) * rng.random((n_params, len(lo)))
    jobs = [(box, P[i], n_u, n_b, seed * 100003 + i) for i in range(n_params)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_coupled_job, jobs, chunksize=4))
    Xu = torch.tensor(np.concatenate([r[0] for r in res]))
    yu = torch.tensor(np.concatenate([r[1] for r in res])).reshape(-1, 1)
    Xb = torch.tensor(np.concatenate([r[2] for r in res]))
    yb = torch.tensor(np.concatenate([r[3] for r in res])).reshape(-1, 1)
    return Xu, yu, Xb, yb
