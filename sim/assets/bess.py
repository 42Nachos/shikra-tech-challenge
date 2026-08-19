"""The BESS port: fixed inputs and outputs, no physics.

Which storage model runs is a config line -- `model:` under the bess spec,
selecting from `sim/assets/bess_models/`. No SOC recursion, no efficiency and no
headroom expression here.

THE PORT OWNS:

    the ADR-0004 asset shape   id, ratings, capability flags, cost data
    the absent-command refusal a missing setpoint is a bug, never a zero
    the INVERTER               reactive power and the D3 kVA circle
    the OBSERVATION            including degrading the true SOC through the
                               noise block

Everything else belongs to the model. Adding a storage model must not need an
edit here.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import (
    AssetCommand,
    AssetCostData,
    AssetMeasurement,
    AssetRatings,
    BessObservation,
)
from sim.asset import (
    Asset,
    AssetCost,
    AssetDelivered,
    AssetEnvelope,
    AssetNoise,
    AssetState,
    ResolvedAsset,
    resolve_kva_circle,
)
from sim.assets.bess_models.base import BessModel
from sim.assets.bess_models.tank import TankSpec
from sim.noise import add_noise
from sim.scenario import ScenarioRow, availability_field


class BessModelParams(BaseModel):
    """The model-keyed parameter blocks, one per implemented storage model.

    TYPED, never a raw mapping: `fitting/space.py` discovers parameters by
    walking pydantic fields (ADR-0011 D3). `extra="forbid"` so a typo'd model
    name is refused.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tank: TankSpec | None = None


class BessSpec(BaseModel):
    """Which storage model, and its parameters. Nothing else.

    The sizing, efficiencies and SOC band belong to whichever model is selected;
    there is no common superset worth writing down.
    """

    model_config = ConfigDict(frozen=True)

    # NO DEFAULT: a plant that runs without saying which physics it used is what
    # ADR-0009 exists to prevent.
    model: str
    params: BessModelParams

    @model_validator(mode="after")
    def _check_model_is_known_and_parameterised(self) -> Self:
        known = tuple(BessModelParams.model_fields)
        if self.model not in known:
            raise ValueError(
                f"unknown bess model {self.model!r}. Known models: {', '.join(known)}. "
                f"There is no default and no fallback."
            )
        if getattr(self.params, self.model) is None:
            raise ValueError(
                f"model is {self.model!r} but params.{self.model} is absent. The selected "
                f"model's parameters are required in full and are never defaulted."
            )
        return self

    def build_model(self, member_id: str, dt_minutes: int) -> BessModel:
        """The selected model, built by its own params block.

        No table lookup: the block IS the selected model's spec. The fields of
        the params class above are the one list of models.
        """
        block = getattr(self.params, self.model)
        assert block is not None, f"BessSpec validated params.{self.model} present"
        model: BessModel = block.build(member_id, dt_minutes)
        return model


class BessNoise(AssetNoise):
    """A battery's meter, plus its state-of-charge estimator. ADR-0012 D2.

    `soc_sigma_kwh` has no counterpart elsewhere in the fleet: SOC is not
    measured but ESTIMATED by the BMS, and it drifts.

    ON THE PORT, not on a model: every storage model is metered the same way.
    """

    soc_sigma_kwh: float = Field(ge=0.0)


@dataclass(frozen=True, kw_only=True)
class BessAsset(Asset):
    """The battery as an ADR-0004 asset, over whichever model config selected.

    Uses BOTH signs of the D2 convention (+ discharge, - charge), so there is no
    one-sided sign assert; `check_delivery` asserts the model's own conservation
    law instead.

    The envelope narrows with state, which is why `AssetEnvelope` stays out of
    contracts.py (D9): its `p_max_kw` inverts to the exact SOC.
    """

    spec: BessSpec
    model: BessModel

    @classmethod
    def from_config(cls, resolved: ResolvedAsset, dt_minutes: int) -> BessAsset:
        spec = resolved.spec
        if not isinstance(spec, BessSpec):
            raise TypeError(
                f"asset {resolved.id!r}: BessAsset needs a BessSpec, got "
                f"{type(spec).__name__}. The registry's spec_cls and asset_cls "
                f"for 'bess' no longer agree."
            )
        model = spec.build_model(resolved.id, dt_minutes)
        p_min_kw, p_max_kw = model.p_limits_kw
        # RATED energy, never the band and never the current SOC, so a policy can
        # express a threshold as a fraction (ADR-0006). On `BessModel` rather than
        # the shared template: only storage has one.
        energy_capacity_kwh = model.energy_capacity_kwh
        return cls(
            id=resolved.id,
            ratings=AssetRatings(
                # Nameplate, unconditioned by state. The Q extent is the
                # inverter's rating, jointly bound by the kVA circle (D3).
                p_min_kw=p_min_kw,
                p_max_kw=p_max_kw,
                q_min_kvar=-model.s_max_kva,
                q_max_kvar=model.s_max_kva,
                s_max_kva=model.s_max_kva,
                energy_capacity_kwh=energy_capacity_kwh,
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

    def initial_state(self, carry_in: Mapping[str, float]) -> AssetState:
        return self.model.initial_state(carry_in)

    def build_cost(self) -> AssetCost:
        """The selected model's cost, built from this member's declared terms."""
        return self.model.build_cost(self.cost_data.terms)

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

        # A CLOSED envelope carries s_max_kva=None, not 0.0 -- an offline asset
        # does not have a zero-radius circle, it has no circle.
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
            label="bess",
        )

    def observe(
        self,
        state: AssetState,
        row: ScenarioRow,
        measured_prev: AssetMeasurement | None,
        rng: random.Random,
    ) -> BessObservation:
        """`soc_observed_kwh` is the DEGRADED view of the true SOC.

        Not a meter reading: a BMS ESTIMATES stored energy, which is why
        `soc_sigma_kwh` is separate from and generally larger than the two meter
        sigmas.

        DELIBERATELY UNCLAMPED (ADR-0012 D4): an estimate may read below zero or
        above capacity, which is when a policy is most exposed.
        """
        return BessObservation(
            available=row.available[self.id],
            ratings=self.ratings,
            measured_prev=self.metered(measured_prev, rng),
            soc_observed_kwh=add_noise(
                self.model.observed_soc_kwh(state), self._noise().soc_sigma_kwh, rng
            ),
        )

    def _noise(self) -> BessNoise:
        """Each asset narrows what the registry handed it."""
        if not isinstance(self.noise, BessNoise):
            raise TypeError(f"asset {self.id!r} expects BessNoise, got {type(self.noise).__name__}")
        return self.noise
