"""Uncertainty quantification on top of a PointPredictor.

Split conformal, residual-normalised conformal (PINN PDE residual as
difficulty), and Mondrian (per-bucket) conformal live in `conformal.py`.
Adapters put existing PINNs and labelled baselines behind the protocol.
"""

from .adapters import BaselineAdapter, PINNAdapter
from .conformal import (
    IntervalResult,
    auto_kappa,
    conformal_quantile,
    empirical_coverage,
    mean_width,
    mondrian_split_conformal_interval,
    moneyness_tenor_buckets,
    split_conformal_interval,
)
from .protocol import PointPredictor

__all__ = [
    "PointPredictor",
    "PINNAdapter",
    "BaselineAdapter",
    "IntervalResult",
    "conformal_quantile",
    "auto_kappa",
    "split_conformal_interval",
    "mondrian_split_conformal_interval",
    "moneyness_tenor_buckets",
    "empirical_coverage",
    "mean_width",
]
