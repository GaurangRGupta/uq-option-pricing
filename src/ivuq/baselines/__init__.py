"""Machine-learning and deep-learning baselines, and the harness that
compares them with the PINNs (why use a PINN at all?).

See `experiments.py` for the three comparison regimes and `metrics.py` for
what is scored beyond price error.
"""

from .data import Box, MarketParams, reference_u, sample_labelled
from .experiments import EXTRAPOLATION_BOX, FULL_BOX, pinn_predictor, run_regime
from .metrics import evaluate
from .models import FAMILY, Baseline, make_baselines

__all__ = [
    "Box",
    "MarketParams",
    "reference_u",
    "sample_labelled",
    "FULL_BOX",
    "EXTRAPOLATION_BOX",
    "pinn_predictor",
    "run_regime",
    "evaluate",
    "FAMILY",
    "Baseline",
    "make_baselines",
]
