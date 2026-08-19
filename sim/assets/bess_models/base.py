"""The template a storage model's PHYSICS must satisfy.

A Protocol rather than a base class, so a model is checked structurally and
inherits nothing it did not ask for.

THERE IS NO COST PROTOCOL HERE, deliberately.
"""

from __future__ import annotations

from typing import Protocol

from sim.asset import AssetModel, AssetState


class BessModel(AssetModel, Protocol):
    """`AssetModel` plus what only storage publishes. ADR-0016 D1.

    Extends rather than restates: every shared member -- `envelope`, `step`,
    `build_cost`, `check_delivery` and the ratings -- is inherited, so a
    signature can move in one place only.
    """

    @property
    def energy_capacity_kwh(self) -> float:
        """Rated energy, for `AssetRatings`. Never the band, never the SOC."""
        ...

    def observed_soc_kwh(self, state: AssetState) -> float:
        """The TRUE stored energy. The port degrades it through the noise block."""
        ...
