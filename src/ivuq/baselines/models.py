"""The data-driven baselines: classical machine learning and deep learning.

Every model has the same two-method interface, `fit(X, u)` and `predict(X)`,
with X = (m, tau) and u = V/K, so the comparison harness treats them
identically. None of them is told anything about option pricing beyond the
labelled examples.

Classical machine learning ("ml"):
  - poly_ridge    : ridge regression on degree-6 polynomial features of (m, tau).
  - kernel_ridge  : RBF kernel ridge regression, the smooth low-dimensional
                    interpolator (the posterior mean of a Gaussian process);
                    bandwidth and penalty are chosen by cross-validation.
  - random_forest : bagged regression trees.
  - grad_boosting : histogram gradient boosted trees.

Deep learning ("dl"), defined in `torch_models.py`: mlp_tanh (architecture
matched to the PINN), mlp_relu, resnet_silu, fourier_mlp, picnn (convex in
moneyness by construction) and bs_iv_net (learned implied-volatility surface
fed through the Black-Scholes formula). See that module for the reasoning.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ivuq.baselines.data import MarketParams
from ivuq.baselines.torch_models import (
    PICNN,
    BSIVNet,
    FourierMLP,
    MLP,
    ResNetMLP,
    TorchRegressor,
)

__all__ = ["Baseline", "make_baselines", "FAMILY"]

FAMILY = {
    "poly_ridge": "ml",
    "kernel_ridge": "ml",
    "random_forest": "ml",
    "grad_boosting": "ml",
    "mlp_tanh": "dl",
    "mlp_relu": "dl",
    "resnet_silu": "dl",
    "fourier_mlp": "dl",
    "picnn": "dl",
    "bs_iv_net": "dl",
}


class Baseline:
    name = "baseline"

    def fit(self, X: np.ndarray, u: np.ndarray) -> "Baseline":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _Sklearn(Baseline):
    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, X: np.ndarray, u: np.ndarray) -> "_Sklearn":
        self.estimator.fit(X, u)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X), dtype=np.float64).ravel()


class _GradBoosting(_Sklearn):
    """min_samples_leaf scales with n: the library default of 20 leaves nothing
    to split on when only 50 labels are available."""

    def __init__(self, seed: int, max_iter: int) -> None:
        super().__init__("grad_boosting", None)
        self.seed, self.max_iter = seed, max_iter

    def fit(self, X: np.ndarray, u: np.ndarray) -> "_GradBoosting":
        leaf = int(max(2, min(20, len(u) // 10)))
        self.estimator = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=0.05,
            min_samples_leaf=leaf,
            early_stopping=False,
            random_state=self.seed,
        )
        return super().fit(X, u)


class _KernelRidgeCV(Baseline):
    name = "kernel_ridge"

    def __init__(self) -> None:
        self.search: GridSearchCV | None = None

    def fit(self, X: np.ndarray, u: np.ndarray) -> "_KernelRidgeCV":
        grid = {"alpha": [1e-9, 1e-7, 1e-5, 1e-3], "gamma": [0.3, 1.0, 3.0, 10.0]}
        cv = int(min(4, max(2, len(u) // 5)))
        self.search = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=cv, scoring="neg_mean_squared_error")
        self.search.fit(X, u)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.search.predict(X), dtype=np.float64).ravel()


def make_baselines(
    seed: int = 0,
    fast: bool = False,
    option_type: str = "put",
    params: MarketParams | None = None,
) -> dict[str, Baseline]:
    """Fresh, unfitted instances of every baseline. `fast` shrinks training
    budgets for the test suite; reported numbers use fast=False. `option_type`
    and `params` are needed only by bs_iv_net, which evaluates the
    Black-Scholes formula and so must know r, q and the payoff."""
    params = params or MarketParams()
    trees = 60 if fast else 300
    boost = 150 if fast else 600
    epochs = 600 if fast else 3000
    decay = max(1, epochs // 3)

    def torch_model(name: str, build) -> TorchRegressor:
        return TorchRegressor(name, build, epochs=epochs, lr_decay_every=decay, seed=seed)

    return {
        "poly_ridge": _Sklearn("poly_ridge", make_pipeline(PolynomialFeatures(6), Ridge(alpha=1e-8))),
        "kernel_ridge": _KernelRidgeCV(),
        "random_forest": _Sklearn(
            "random_forest", RandomForestRegressor(n_estimators=trees, random_state=seed, n_jobs=-1)
        ),
        "grad_boosting": _GradBoosting(seed, boost),
        "mlp_tanh": torch_model("mlp_tanh", lambda d: MLP(d, 4, 64, "tanh")),
        "mlp_relu": torch_model("mlp_relu", lambda d: MLP(d, 4, 64, "relu")),
        "resnet_silu": torch_model("resnet_silu", lambda d: ResNetMLP(d, 3, 64)),
        "fourier_mlp": torch_model("fourier_mlp", lambda d: FourierMLP(d)),
        "picnn": torch_model("picnn", lambda d: PICNN()),
        "bs_iv_net": torch_model("bs_iv_net", lambda d: BSIVNet(option_type, params)),
    }
