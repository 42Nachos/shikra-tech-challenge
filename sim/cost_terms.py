"""Which cost terms exist, and the params model each validates with.

ADR-0004 D7, amended by ADR-0016 D3. THIS MODULE PRICES NOTHING: every term
lives on the model that generates the quantity it prices.

    energy, export, demand_charge, pf_penalty  sim/assets/grid_models/tnerc_ht.py
    fuel                                       sim/assets/genset_models/fuel_curve.py
    degradation                                sim/assets/bess_models/tank.py
    unserved                                   sim/load.py

DERIVED, NOT LISTED: every entry comes from a cost class's own `KIND_PARAMS`, so
there is no second list of kinds to edit only one of. The tables are ORDERED
MAPPINGS, and that order is the order each member's lines appear in.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from sim.assets.bess_models import TankCost
from sim.assets.genset_models import FuelCurveCost
from sim.assets.grid_models.tnerc_ht import TnercHtCost
from sim.load import UnservedCost

# Every cost class, named directly, in the order site_a declares its members.
# This is the one place that needs to see all of them at once.
_COST_CLASSES: tuple[Mapping[str, type[BaseModel]], ...] = (
    TankCost.KIND_PARAMS,
    TnercHtCost.KIND_PARAMS,
    FuelCurveCost.KIND_PARAMS,
    UnservedCost.KIND_PARAMS,
)

# Kind -> the params model it validates against. `metrics/oracle_objective.py`
# reads it, so that it parses declared terms with the SAME model the
# implementation validates with (ADR-0008's sharing licence).
COST_TERM_PARAMS: Mapping[str, type[BaseModel]] = {
    kind: params_cls for table in _COST_CLASSES for kind, params_cls in table.items()
}

# The cost terms this testbed can actually price; `sim/fleet.py` refuses any
# kind not in here (rejection 4).
#
# TWO HONEST ABSENCES, each a real gap:
#
#   start_cost   `GensetState.on` and `GensetSpec.start_cost_inr` exist; nothing
#                prices the transition.
#   curtailment  `AssetDelivered.curtailed_kw` carries the quantity; nothing
#                prices it (ADR-0004 D4).
#
# Each creates real money on the stress case, so each lands with its own bless.
IMPLEMENTED_COST_TERMS: frozenset[str] = frozenset(COST_TERM_PARAMS)
