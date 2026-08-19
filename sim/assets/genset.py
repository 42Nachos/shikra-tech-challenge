"""The genset PORT: fixed inputs and outputs, no physics and no fuel price.

Which generating-set model runs is a config line -- `model:` under the genset
spec, selecting from `sim/assets/genset_models/`. No fuel curve, no minimum
stable load and no on/off semantics here.

THE PORT OWNS the ADR-0004 asset shape, the absent-command refusal, reactive
power and the observation. Everything else, availability included, belongs to
the model.
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

from sim.assets.genset_models.fuel_curve import FuelCurveSpec
from sim.scenario import ScenarioRow, availability_field


class GensetModelParams(BaseModel):
    """The model-keyed parameter blocks, one per implemented set.

    Typed, never a raw mapping: `fitting/space.py` discovers parameters by
    walking pydantic fields. `extra="forbid"` so a typo'd MODEL name is refused
    rather than silently ignored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fuel_curve: FuelCurveSpec | None = None


class GensetSpec(BaseModel):
    """Which generating-set model, and its parameters. Nothing else."""

    model_config = ConfigDict(frozen=True)

    # NO DEFAULT. A plant that runs without saying which physics it used is what
    # ADR-0009 exists to prevent.
    model: str
    params: GensetModelParams

    @model_validator(mode="after")
    def _check_model_is_known_and_parameterised(self) -> Self:
        known = tuple(GensetModelParams.model_fields)
        if self.model not in known:
            raise ValueError(
                f"unknown genset model {self.model!r}. Known models: {', '.join(known)}. "
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
        assert block is not None, f"GensetSpec validated params.{self.model} present"
        model: AssetModel = block.build(member_id, dt_minutes)
        return model


@dataclass(frozen=True, kw_only=True)
class GensetAsset(Asset):
    """The diesel genset as an ADR-0004 asset, over whichever model config chose.

    An injector with a separate on/off channel. The record is a
    `GensetDelivered`, since `p_net_kw` alone cannot distinguish an idling set
    from a stopped one.
    """

    spec: GensetSpec
    model: AssetModel

    @classmethod
    def from_config(cls, resolved: ResolvedAsset, dt_minutes: int) -> GensetAsset:
        spec = resolved.spec
        if not isinstance(spec, GensetSpec):
            raise TypeError(
                f"asset {resolved.id!r}: GensetAsset needs a GensetSpec, got "
                f"{type(spec).__name__}. The registry's spec_cls and asset_cls "
                f"for 'genset' no longer agree."
            )
        model = spec.build_model(resolved.id, dt_minutes)
        p_min_kw, p_max_kw = model.p_limits_kw
        return cls(
            id=resolved.id,
            # Nameplate, not this-interval truth (that is envelope()). No
            # reactive channel: excitation is not modelled, so the Q extent is
            # zero and s_max equals the real rating.
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
        """What the set burns, from its own curve and the declared fuel price."""
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
        # No reactive clipping: this asset declares no Q channel, so there is no
        # kVA circle to resolve against.
        return next_state, delivered

    def observe(
        self,
        state: AssetState,
        row: ScenarioRow,
        measured_prev: AssetMeasurement | None,
        rng: random.Random,
    ) -> AssetObservation:
        """Availability, nameplate, last measurement. `running` is truth-side and
        does not cross."""
        return AssetObservation(
            available=row.available[self.id],
            ratings=self.ratings,
            measured_prev=self.metered(measured_prev, rng),
        )
