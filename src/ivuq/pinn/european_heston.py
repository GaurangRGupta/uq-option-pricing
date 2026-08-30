"""User-facing wrapper: fit a European-Heston PINN, then price and inspect
its PDE residual in real (S, K, v, tau) units instead of normalized
(m, v, tau). Same interface as `EuropeanPINN`, plus a `v` argument since
variance is now a state variable rather than a fixed parameter."""

from __future__ import annotations

import numpy as np
import torch

from .config import HestonEuropeanPINNConfig
from .heston_pde import pde_residual as _pde_residual_fn
from .network import PINN
from .train_european_heston import train_european_heston_pinn

__all__ = ["EuropeanHestonPINN"]


class EuropeanHestonPINN:
    """Trained on one (option_type, r, q, kappa, theta, xi, rho) Heston
    setup; prices any (S, K, v, tau) within the trained moneyness/variance/
    maturity domain by rescaling m=S/K."""

    def __init__(self, config: HestonEuropeanPINNConfig) -> None:
        self.config = config
        self.model: PINN | None = None
        self.history: dict[str, list[float]] | None = None

    def fit(self) -> "EuropeanHestonPINN":
        self.model, self.history = train_european_heston_pinn(self.config)
        self.model.eval()
        return self

    def _check_fitted(self) -> PINN:
        if self.model is None:
            raise RuntimeError("call .fit() before .price() or .pde_residual()")
        return self.model

    def price(
        self,
        S: np.ndarray | float,
        K: np.ndarray | float,
        v: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> np.ndarray | float:
        """V(S, v, tau) = K * u(S/K, v, tau). Returns a scalar if the inputs
        were scalars. `v` defaults to the config's v0 if not given."""
        model = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m = torch.tensor(S_arr / K_arr, dtype=torch.float32).reshape(-1, 1)
        v_t = torch.tensor(np.broadcast_to(v_arr, m.shape[:1]).copy(), dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            u = model(torch.cat([m, v_t, tau_t], dim=1)).numpy().reshape(-1)
        price = u * K_arr

        return float(price[0]) if price.shape[0] == 1 and np.isscalar(S) else price

    def pde_residual(
        self,
        S: np.ndarray | float,
        K: np.ndarray | float,
        v: np.ndarray | float,
        tau: np.ndarray | float,
    ) -> np.ndarray:
        """The Heston PDE residual at (S, v, tau) -- near zero where the
        model has learned the dynamics well."""
        model = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        v_arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m = torch.tensor(S_arr / K_arr, dtype=torch.float32).reshape(-1, 1)
        v_t = torch.tensor(np.broadcast_to(v_arr, m.shape[:1]).copy(), dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        residual = _pde_residual_fn(
            model, m, v_t, tau_t, self.config.r, self.config.q,
            self.config.kappa, self.config.theta, self.config.xi, self.config.rho,
        )
        return residual.detach().numpy().reshape(-1)
