"""The genset model family: what a generating set is, and what it burns.

`sim/assets/genset.py` is the PORT. Each module here is one set model, complete.
Fuel is litres per hour at a load and only the curve knows how many, which is
why the cost lives beside the physics.

ONE MODEL TODAY: `fuel_curve`. Minimum run and down times, or a thermal derate
with ambient, is a second module beside it (ADR-0006).

NO REGISTRY: config names a model, the port's params class has one typed field
per model, and each spec builds its own. This module only re-exports.
"""

from __future__ import annotations

from sim.assets.genset_models.fuel_curve import (
    FuelCurve,
    FuelCurveCost,
    FuelCurveModel,
    FuelCurveSpec,
    GensetState,
    deliver,
    fuel_lph,
)

__all__ = [
    "FuelCurve",
    "FuelCurveCost",
    "FuelCurveModel",
    "FuelCurveSpec",
    "GensetState",
    "deliver",
    "fuel_lph",
]
