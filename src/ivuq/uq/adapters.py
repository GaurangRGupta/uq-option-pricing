"""Put existing PINNs and labelled baselines behind `PointPredictor`."""

from __future__ import annotations

import numpy as np

__all__ = ["PINNAdapter", "BaselineAdapter"]


class PINNAdapter:
    """Wraps EuropeanPINN or AmericanPINN (GBM, two inputs).

    Heston wrappers take an extra variance argument; do not use this adapter
    for those. Write a thin Heston adapter when conformal work starts on
    (m, v, tau).
    """

    def __init__(self, pinn, K: float = 100.0, name: str | None = None) -> None:
        self.pinn = pinn
        self.K = float(K)
        self.name = name or type(pinn).__name__

    def predict(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        price = np.asarray(self.pinn.price(m * self.K, self.K, tau), dtype=np.float64)
        return price.reshape(-1) / self.K

    def constraint_residual(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        return np.asarray(
            self.pinn.pde_residual(m * self.K, self.K, tau), dtype=np.float64
        ).reshape(-1)


class BaselineAdapter:
    """Wraps a fitted labelled model with `.name` and `.predict(X)`.

    X is the (n, 2) array of (m, tau). No PDE residual. sklearn is not
    imported here, so this module can load without the baselines extra.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.name = model.name

    def predict(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        X = np.column_stack([m, tau])
        return np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1)

    def constraint_residual(self, m: np.ndarray, tau: np.ndarray) -> None:
        return None
