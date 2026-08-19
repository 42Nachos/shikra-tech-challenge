"""The BESS model family: what a storage model is, and what it costs.

`sim/assets/bess.py` is the PORT. Each module here is one storage model,
complete: parameters, state, recursion, and the cost its own quantities
determine. Degradation is priced from throughput, which only the model counts.

ONE MODEL TODAY: `tank`. A power-vs-SOC taper, capacity fade or a temperature
term is a second module beside it, not a flag on this one.

NO REGISTRY: config names a model, the port's params class has one typed field
per model, and each spec builds its own. This module only re-exports.
"""

from __future__ import annotations

from sim.assets.bess_models.base import BessModel
from sim.assets.bess_models.tank import TankCost, TankModel, TankSpec, TankState

__all__ = [
    "BessModel",
    "TankCost",
    "TankModel",
    "TankSpec",
    "TankState",
]
