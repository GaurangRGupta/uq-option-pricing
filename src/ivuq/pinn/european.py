"""User-facing wrapper: fit a European PINN, then price and inspect its
PDE residual in real (S, K, tau) units instead of normalized (m, tau).
"""

from __future__ import annotations

import numpy as np
import torch

from .black_scholes_pde import pde_residual as _pde_residual_fn
from .config import EuropeanPINNConfig
from .network import PINN
from .train import train_european_pinn

__all__ = ["EuropeanPINN"]


class EuropeanPINN:
    """Trained on one (option_type, r, q, sigma) setup; prices any (S, K, tau)
    within the trained moneyness/maturity domain by rescaling m=S/K."""

    def __init__(self, config: EuropeanPINNConfig) -> None:
        self.config = config
        self.model: PINN | None = None
        self.history: dict[str, list[float]] | None = None

    def fit(self) -> "EuropeanPINN":
        self.model, self.history = train_european_pinn(self.config)
        self.model.eval()
        return self

    def _check_fitted(self) -> PINN:
        if self.model is None:
            raise RuntimeError("call .fit() before .price() or .pde_residual()")
        return self.model

    def price(self, S: np.ndarray | float, K: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray | float:
        """V(S, tau) = K * u(S/K, tau). Returns a scalar if the inputs were scalars."""
        model = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m = torch.tensor(S_arr / K_arr, dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            u = model(torch.cat([m, tau_t], dim=1)).numpy().reshape(-1)
        v = u * K_arr

        return float(v[0]) if v.shape[0] == 1 and np.isscalar(S) else v

    def pde_residual(self, S: np.ndarray | float, K: np.ndarray | float, tau: np.ndarray | float) -> np.ndarray:
        """The PDE residual at (S, tau) -- near zero where the model has learned
        the dynamics well, large where it hasn't. This is the raw signal the
        physics-informed nonconformity score (Phase 4) is built on."""
        model = self._check_fitted()
        S_arr = np.atleast_1d(np.asarray(S, dtype=np.float64))
        K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))

        m = torch.tensor(S_arr / K_arr, dtype=torch.float32).reshape(-1, 1)
        tau_t = torch.tensor(tau_arr, dtype=torch.float32).reshape(-1, 1)
        residual = _pde_residual_fn(model, m, tau_t, self.config.r, self.config.q, self.config.sigma)
        return residual.detach().numpy().reshape(-1)
