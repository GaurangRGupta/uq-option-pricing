"""Physics-informed neural network pricers.

Start with the European option under GBM (this module). American
free-boundary and Heston-dynamics versions build on the same PDE-residual
machinery in later phases (see planning/RESEARCH_GAP_AND_ROADMAP.md
Section 8).
"""

from .config import EuropeanPINNConfig
from .european import EuropeanPINN

__all__ = ["EuropeanPINNConfig", "EuropeanPINN"]
