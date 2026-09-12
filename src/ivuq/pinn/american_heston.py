"""User-facing wrapper for N2-Heston (`heston_free_boundary.py`,
`train_american_heston.py`): fits the coupled solution+boundary network and
exposes price/boundary/PDE-residual queries in real (S, K, v, tau) units.

Unlike every other wrapper in this package, `.price()` is a piecewise
function of the two trained networks rather than a single forward pass:
below the boundary it returns intrinsic value directly (the solution
network was never trained to represent that region -- see the module
docstring in `heston_free_boundary.py`); above it, the solution network's
own prediction. Put only, same scope as the rest of N2-Heston."""

from __future__ import annotations

import numpy as np
import torch

from .config import AmericanHestonPINNConfig
from .heston_free_boundary import BoundaryNet
from .heston_pde import pde_residual as _pde_residual_fn
from .network import PINN
from .train_american_heston import train_american_heston_pinn

__all__ = ["AmericanHestonPINN"]


class AmericanHestonPINN:
    """Trained on one (r, q, kappa, theta, xi, rho) Heston-American-put
    setup; prices any (S, K, v, tau) within the trained domain."""

    def __init__(self, config: AmericanHestonPINNConfig) -> None:
        self.config = config
        self.sol_net: PINN | None = None
        self.bnd_net: BoundaryNet | None = None
        self.history: dict[str, list[float]] | None = None

    def fit(self) -> "AmericanHestonPINN":
        self.sol_net, self.bnd_net, self.history = train_american_heston_pinn(self.config)
        self.sol_net.eval()
        self.bnd_net.eval()
        return self

    def _check_fitted(self) -> tuple[PINN, BoundaryNet]:
        if self.sol_net is None or self.bnd_net is None:
            raise RuntimeError("call .fit() before .price(), .boundary(), or .pde_residual()")
        return self.sol_net, self.bnd_net

    def boundary(self, v: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray | float:
        """S*(v, tau), the exercise boundary in real units: K * b_phi(v, tau)
        for the strike passed in separately (moneyness only, so callers
        multiply by their own K -- see `price` for the analogous pattern)."""
        _, bnd_net = self._check_fitted()
        v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        v_t = torch.tensor(v_arr, dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            b = bnd_net(v_t, tau_t).numpy().reshape(-1)
        return float(b[0]) if b.shape[0] == 1 and np.isscalar(v) else b

    def price(
        self,
        S: np.ndarray | float,
        K: np.ndarray | float,
        v: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> np.ndarray | float:
        """V(S, v, tau) = K * u(S/K, v, tau) in the continuation region,
        K * max(1-S/K, 0) in the exercise region. Returns a scalar if the
        inputs were scalars."""
        sol_net, bnd_net = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m_arr = S_arr / K_arr
        m = torch.tensor(m_arr, dtype=torch.float32).reshape(-1, 1)
        v_t = torch.tensor(np.broadcast_to(v_arr, m.shape[:1]).copy(), dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)

        with torch.no_grad():
            b_arr = bnd_net(v_t, tau_t).numpy().reshape(-1)
            u_continuation = sol_net(torch.cat([m, v_t, tau_t], dim=1)).numpy().reshape(-1)
        intrinsic = np.maximum(1.0 - m_arr, 0.0)
        u = np.where(m_arr >= b_arr, u_continuation, intrinsic)
        price = u * K_arr

        return float(price[0]) if price.shape[0] == 1 and np.isscalar(S) else price

    def pde_residual(
        self,
        S: np.ndarray | float,
        K: np.ndarray | float,
        v: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> np.ndarray:
        """L[u] at (S, v, tau) via the solution network alone -- only
        meaningful where m >= boundary(v, tau); the solution net was never
        trained to satisfy the PDE below its own boundary estimate."""
        sol_net, _ = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m = torch.tensor(S_arr / K_arr, dtype=torch.float32).reshape(-1, 1)
        v_t = torch.tensor(np.broadcast_to(v_arr, m.shape[:1]).copy(), dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        residual = _pde_residual_fn(
            sol_net, m, v_t, tau_t, self.config.r, self.config.q,
            self.config.kappa, self.config.theta, self.config.xi, self.config.rho,
        )
        return residual.detach().numpy().reshape(-1)
