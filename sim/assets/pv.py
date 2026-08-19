"""The PV PORT: fixed inputs and outputs, no physics.

Which array model runs is a config line -- `model:` under the pv spec, selecting
from `sim/assets/pv_models/`. No equations here.

THE PORT OWNS the ADR-0004 asset shape, the absent-command refusal, reactive
power and the D3 kVA circle, and the observation. Everything else, availability
included, belongs to the model.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
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
    resolve_kva_circle,
)

from sim.assets.pv_models.pvwatts import PvWattsSpec
from sim.scenario import ScenarioRow, availability_field


class PvModelParams(BaseModel):
    """The model-keyed parameter blocks, one per implemented array model.

    Blocks for models you are NOT using are tolerated and inert (ADR-0011 D3);
    `build_model` reads exactly one field.

    TYPED, never a raw mapping: `fitting/space.py` discovers parameters by
    walking pydantic fields. `extra="forbid"` so a typo'd model name is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pvwatts: PvWattsSpec | None = None


class PvSpec(BaseModel):
    """Which array model, and its parameters. Nothing else."""

    model_config = ConfigDict(frozen=True)

    # NO DEFAULT. A plant that runs without saying which physics it used is what
    # ADR-0009 exists to prevent.
    model: str
    params: PvModelParams

    # ADR-0011's refusals are VALIDATORS, never checks inside a build step:
    # `fitting/harness.py` turns a ValidationError into one infeasible candidate,
    # but a ValueError from `from_config` crashes the search (ADR-0010 D5).

    @model_validator(mode="after")
    def _check_model_is_known_and_parameterised(self) -> Self:
        known = tuple(PvModelParams.model_fields)
        if self.model not in known:
            raise ValueError(
                f"unknown pv model {self.model!r}. Known models: {', '.join(known)} "
                f"(ADR-0011 D1). There is no default and no fallback."
            )
        if getattr(self.params, self.model) is None:
            raise ValueError(
                f"model is {self.model!r} but params.{self.model} is absent. The selected "
                f"model's parameters are required in full and are never defaulted "
                f"(ADR-0011 D3). Blocks for models you are NOT using are tolerated and "
                f"inert, so keeping them costs nothing."
            )
        return self

    def build_model(self, member_id: str, dt_minutes: int) -> AssetModel:
        """The selected model, built by its own params block.

        No table lookup: the block IS the selected model's spec, and the spec
        knows which model it parameterises. The fields of the params class above
        are the one list of models.
        """
        block = getattr(self.params, self.model)
        assert block is not None, f"PvSpec validated params.{self.model} present"
        model: AssetModel = block.build(member_id, dt_minutes)
        return model


@dataclass(frozen=True, kw_only=True)
class PvAsset(Asset):
    """The PV array as an ADR-0004 asset, over whichever model config selected.

    An injector: p_net_kw >= 0 always, asserted by the model (D2). Commanding
    below available IS the curtailment (D4: there is no curtail channel).
    """

    spec: PvSpec
    model: AssetModel

    @classmethod
    def from_config(cls, resolved: ResolvedAsset, dt_minutes: int) -> PvAsset:
        spec = resolved.spec
        if not isinstance(spec, PvSpec):
            raise TypeError(
                f"asset {resolved.id!r}: PvAsset needs a PvSpec, got "
                f"{type(spec).__name__}. The registry's spec_cls and asset_cls "
                f"for 'pv' no longer agree."
            )
        model = spec.build_model(resolved.id, dt_minutes)
        p_min_kw, p_max_kw = model.p_limits_kw
        return cls(
            id=resolved.id,
            # Nameplate, not this-interval truth (that is envelope()). The Q
            # extent is the inverter's full rating, reached only at P = 0; the
            # kVA circle is the joint constraint (D3).
            ratings=AssetRatings(
                p_min_kw=p_min_kw,
                p_max_kw=p_max_kw,
                q_min_kvar=-model.s_max_kva,
                q_max_kvar=model.s_max_kva,
                s_max_kva=model.s_max_kva,
            ),
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
            # validate_commands() refuses this before the plant calls step();
            # repeated here because a defaulted zero would silently mean
            # "curtail everything" and still balance (D4).
            raise ValueError(f"asset {self.id!r}: p_setpoint_kw is absent, not zero (D4)")
        next_state, delivered = self.model.step(state, cmd, row)
        self.model.check_delivery(state, next_state, delivered, row)

        # OFFLINE IS THE WHOLE INVERTER, so no reactive either (D11). Read off
        # the model's envelope, which keeps availability inside the model.
        # A CLOSED envelope carries s_max_kva=None, not 0.0: an offline asset has
        # no circle rather than a zero-radius one.
        s_max_kva = self.model.envelope(state, row).s_max_kva
        if s_max_kva is None:
            return next_state, replace(delivered, q_net_kvar=0.0)
        return resolve_kva_circle(
            model=self.model,
            state=state,
            cmd=cmd,
            row=row,
            next_state=next_state,
            delivered=delivered,
            s_max_kva=s_max_kva,
            priority=self.kva_priority,
            label="pv",
        )

    def observe(
        self,
        state: AssetState,
        row: ScenarioRow,
        measured_prev: AssetMeasurement | None,
        rng: random.Random,
    ) -> AssetObservation:
        """Availability, nameplate, last measurement. NO FORECAST: forecasting is
        a dispatcher-side model, which is what makes forecast error a measured
        output rather than a quantity defined away.

        Noise applies to the one channel this asset publishes -- not to
        `available`, a contactor state, nor to `ratings`, a nameplate.
        """
        return AssetObservation(
            available=row.available[self.id],
            ratings=self.ratings,
            measured_prev=self.metered(measured_prev, rng),
        )
