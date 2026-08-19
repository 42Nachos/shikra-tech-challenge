"""The grid model family: what a utility connection is, and what it charges.

`sim/assets/grid.py` is the PORT. Each module here is one kind of connection,
complete: the equipment's physics and the tariff structure it is billed under.

ONE MODEL TODAY: `tnerc_ht`. An 11-month demand ratchet, or a block rate whose
marginal price falls with consumption, is a different STRUCTURE and belongs in a
second module.

NO REGISTRY: config names a model, the port's params class has one typed field
per model, and each spec builds its own. This module only re-exports.
"""

from __future__ import annotations

from sim.assets.grid_models.tnerc_ht import (
    DemandChargeParams,
    ExportParams,
    GridState,
    PfPenaltyBand,
    PfPenaltyParams,
    TnercHtCost,
    TnercHtModel,
    TnercHtSpec,
)

__all__ = [
    "DemandChargeParams",
    "ExportParams",
    "GridState",
    "PfPenaltyBand",
    "PfPenaltyParams",
    "TnercHtCost",
    "TnercHtModel",
    "TnercHtSpec",
]
