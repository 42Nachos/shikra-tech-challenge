"""The PV model family: one array model today, and yours beside it.

`sim/assets/pv.py` is the PORT. Each module here carries its own parameters,
state, thermal model, DC equation, derate chain, envelope, step and cost. None
imports another and none inherits from a shared base, so a second model cannot
change what `pvwatts` computes.

THE DUPLICATION IS THE POINT: two MODELS are competing hypotheses about the same
equipment and are supposed to differ, unlike two implementations of one thing. If
you reuse the PVWatts chain, COPY it into your module rather than importing it.

NO REGISTRY: config names a model, the port's params class has one typed field
per model, and each spec builds its own. This module only re-exports.
"""

from __future__ import annotations

from sim.assets.pv_models.pvwatts import PvWattsCost, PvWattsModel, PvWattsSpec, PvWattsState

__all__ = [
    "PvWattsCost",
    "PvWattsModel",
    "PvWattsSpec",
    "PvWattsState",
]
