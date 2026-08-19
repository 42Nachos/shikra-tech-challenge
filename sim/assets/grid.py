"""The grid PORT: fixed inputs and outputs, no physics and no tariff.

ADR-0004 D5/D7, "the grid is just an asset". Which connection model runs is a
config line -- `model:` under the grid spec, selecting from
`sim/assets/grid_models/`. No equations and no rates here.

THE PORT OWNS the ADR-0004 asset shape, the absent-command refusal, reactive
power and the D3 kVA circle, and the observation. Everything else, availability
included, belongs to the model.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from contracts import (
    AssetCommand,
    AssetCostData,
    AssetMeasurement,
    AssetObservation,
    AssetRatings,
)
from sim.asset import (
    Asset,
    AssetCost,
    AssetDelivered,
    AssetEnvelope,
    AssetModel,
    AssetState,
    ResolvedAsset,
)
from sim.assets.grid_models.tnerc_ht import TnercHtSpec
from sim.scenario import ScenarioRow, availability_field


class GridModelParams(BaseModel):
    """The model-keyed parameter blocks, one per implemented connection.

    TYPED, never a raw mapping: `fitting/space.py` discovers parameters by
    walking pydantic fields. `extra="forbid"` so a typo'd model name is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tnerc_ht: TnercHtSpec | None = None


class GridSpec(BaseModel):
    """Which connection model, and its parameters. Nothing else."""

    model_config = ConfigDict(frozen=True)

    # NO DEFAULT. A plant that runs without saying which physics it used is what
    # ADR-0009 exists to prevent.
    model: str
    params: GridModelParams

    @model_validator(mode="after")
    def _check_model_is_known_and_parameterised(self) -> Self:
        known = tuple(GridModelParams.model_fields)
        if self.model not in known:
            raise ValueError(
                f"unknown grid model {self.model!r}. Known models: {', '.join(known)}. "
                f"There is no default and no fallback."
            )
        if getattr(self.params, self.model) is None:
            raise ValueError(
                f"model is {self.model!r} but params.{self.model} is absent. The selected "
                f"model's parameters are required in full and are never defaulted."
            )
        return self

    def build_model(self, member_id: str, dt_minutes: int) -> AssetModel:
        """The selected model, built by its own params block.

        No table lookup: the block IS the selected model's spec. The fields of
        the params class above are the one list of models.
        """
        block = getattr(self.params, self.model)
        assert block is not None, f"GridSpec validated params.{self.model} present"
        model: AssetModel = block.build(member_id, dt_minutes)
        return model


@dataclass(frozen=True, kw_only=True)
class GridAsset(Asset):
    """The utility connection as an ADR-0004 asset.

    Both signs of the D2 convention: > 0 import, < 0 export, bounded by the SAME
    rating because every kW crosses the same windings.

    THE SETPOINT IS ADVISORY (D5): while the grid holds the bus, the PLANT
    computes the balance and hands the balancing value down, and records the
    commanded-versus-delivered delta. Nothing here fabricates it.
    """

    spec: GridSpec
    model: AssetModel

    @classmethod
    def from_config(cls, resolved: ResolvedAsset, dt_minutes: int) -> GridAsset:
        spec = resolved.spec
        if not isinstance(spec, GridSpec):
            raise TypeError(
                f"asset {resolved.id!r}: GridAsset needs a GridSpec, got "
                f"{type(spec).__name__}. The registry's spec_cls and asset_cls "
                f"for 'grid' no longer agree."
            )
        model = spec.build_model(resolved.id, dt_minutes)
        p_min_kw, p_max_kw = model.p_limits_kw
        return cls(
            id=resolved.id,
            # The transformer rates apparent power; with PCC reactive not yet
            # modelled (site PF assumed 1.0), the P bounds equal it.
            ratings=AssetRatings(p_min_kw=p_min_kw, p_max_kw=p_max_kw, s_max_kva=model.s_max_kva),
            cost_data=AssetCostData(terms=resolved.cost_terms),
            p_controllable=resolved.p_controllable,
            q_controllable=resolved.q_controllable,
            on_off_controllable=resolved.on_off_controllable,
            kva_priority=resolved.kva_priority,
            slack_bearing_capable=resolved.slack_bearing_capable,
            resource_limited=resolved.resource_limited,
            spec=spec,
            model=model,
            noise=resolved.noise,
        )

    def build_cost(self) -> AssetCost:
        """The connection's tariff, built from this member's declared terms."""
        return self.model.build_cost(self.cost_data.terms)

    def initial_state(self, carry_in: Mapping[str, float]) -> AssetState:
        return self.model.initial_state(carry_in)

    def required_scenario_fields(self) -> frozenset[str]:
        # The model names the data columns; the port adds availability, which
        # is keyed by member id and is the same fact for every asset.
        return self.model.required_scenario_fields() | {availability_field(self.id)}

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        return self.model.envelope(state, row)

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[AssetState, AssetDelivered]:
        if cmd.p_setpoint_kw is None:
            raise ValueError(f"asset {self.id!r}: p_setpoint_kw is absent, not zero (D4)")
        next_state, delivered = self.model.step(state, cmd, row)
        self.model.check_delivery(state, next_state, delivered, row)
        # No reactive clipping, unlike PV and the BESS: there is no PCC reactive
        # model yet, so the model's 0.0 stands unaltered.
        return next_state, delivered

    def observe(
        self,
        state: AssetState,
        row: ScenarioRow,
        measured_prev: AssetMeasurement | None,
        rng: random.Random,
    ) -> AssetObservation:
        """Availability, nameplate, last measurement. NO PRICES.

        The tariff is DECLARED, not measured: a dispatcher already holds it a
        priori on `AssetView.cost_terms` and prices against it in its own 5.2
        model. Publishing it here too was a second channel for one fact.

        A DYNAMIC price -- a spot market, a demand-response signal -- is
        genuinely exogenous and would be a scenario column. On the day a site has
        one, `GridObservation` returns carrying it.
        """
        return AssetObservation(
            available=row.available[self.id],
            ratings=self.ratings,
            measured_prev=self.metered(measured_prev, rng),
        )
