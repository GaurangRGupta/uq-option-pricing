"""Shared contract for anything that will later sit under conformal prediction.

A conformal wrapper only needs a point prediction and, optionally, a
non-negative difficulty score. For PINNs the difficulty score is the absolute
PDE residual. For labelled baselines it can be omitted (vanilla split
conformal) or replaced later by a residual model.

Inputs are the same non-dimensional coordinates the rest of the library uses:
m = S/K and tau = T - t. Predictions are u = V/K.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PointPredictor(Protocol):
    name: str

    def predict(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Return u(m, tau) with the same length as m."""

    def constraint_residual(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray | None:
        """Signed PDE (or other physics) residual, or None if the model has none.

        Conformal code should use abs(residual) as a difficulty weight. None
        means fall back to unweighted split conformal.
        """
