"""PVWatts: linear in irradiance, linear in temperature. ADR-0003 D5's incumbent.

A STANDALONE MODEL: parameters, state, thermal model, DC equation, derate chain,
envelope, step and cost all live in this file. Nothing is inherited from a
neighbouring model and nothing is shared with one, so a model wanting a
wind-driven thermal term or a different clip can have it without changing its
neighbours (ADR-0016 D2). What IS shared is the contract -- `AssetModel` in
sim/asset.py -- not the behaviour.

    P_dc = pdc0 * (G_eff / 1000) * (1 + gamma_pdc * (T_cell - 25))

Its two parameters are printed on a datasheet. Its nameplate is AGGREGATE, so
module-to-module mismatch is lumped inside `system_derate`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Self

from pvlib.pvsystem import pvwatts_dc
from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import AssetCommand, CostTerm
from sim.asset import AssetDelivered, AssetEnvelope, AssetState
from sim.scenario import Interval, ScenarioRow

# The NOCT reference conditions, from the DEFINITION of nominal operating cell
# temperature: 800 W/m2, 20 degC ambient, 1 m/s wind, open rack.
_NOCT_REFERENCE_IRRADIANCE_W_M2 = 800.0
_NOCT_REFERENCE_AMBIENT_C = 20.0

# PVWatts' own reference condition. Definitional, not a belief about the module.
_TEMP_REF_C = 25.0

# Floor on the low-light factor: a promised bound that crashes is not a bound.
_ETA_FLOOR = 1e-6


class PvWattsSpec(BaseModel):
    """Every parameter this model needs. `extra="forbid"` so a parameter that
    belongs to a DIFFERENT model is refused by name rather than silently
    dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_dc0_kw: float = Field(gt=0)
    gamma_pdc: float  # per degC, negative
    # ADR-0003 D5a's logarithmic low-light modifier. ONLY THIS MODEL HAS IT,
    # since PVWatts is linear all the way down. Required, not optional.
    low_light_k: float
    low_light_g_ref_w_m2: float = Field(gt=0)

    # Nominal operating cell temperature: a parameter an estimator is expected
    # to move, not a rating a datasheet reliably prints.
    noct_c: float = Field(gt=0)
    bifacial_gain_pct: float = Field(ge=0)
    soiling: float = Field(gt=0, le=1)
    system_derate: float = Field(gt=0, le=1)
    inverter_eff: float = Field(gt=0, le=1)
    p_ac_max_kw: float = Field(gt=0)
    # Inverter apparent-power rating, for the ADR-0004 D3 kVA circle.
    s_max_kva: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_apparent_power_covers_real(self) -> Self:
        if self.s_max_kva < self.p_ac_max_kw:
            raise ValueError(
                f"s_max_kva={self.s_max_kva} < p_ac_max_kw={self.p_ac_max_kw}: real power "
                f"alone cannot exceed apparent power"
            )
        return self

    def build(self, member_id: str, dt_minutes: int) -> PvWattsModel:
        """This spec's model, bound to its member and the plant step.

        ON THE SPEC, so there is no second table mapping names to classes.
        """
        return PvWattsModel(member_id=member_id, spec=self, dt_minutes=dt_minutes)


@dataclass(frozen=True, kw_only=True)
class PvWattsState(AssetState):
    """Stateless. Declared anyway, so the first thing THIS model must remember
    lands here without moving a signature."""


@dataclass(kw_only=True)
class PvWattsCost:
    """What this array costs. ZERO, for now.

    `AssetDelivered.curtailed_kw` carries the quantity a curtailment charge would
    price (ADR-0004 D4); what is missing is what curtailed energy is WORTH.

    REFUSES a declared kind rather than billing it as zero forever.
    """

    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {}
    KINDS: ClassVar[frozenset[str]] = frozenset()

    _total_inr: float = 0.0

    @classmethod
    def build(cls, *, member_id: str, declared: tuple[CostTerm, ...]) -> PvWattsCost:
        if declared:
            raise ValueError(
                f"asset {member_id!r} declares cost terms "
                f"{sorted({t.kind for t in declared})}, but this PV model prices none of "
                f"them. Refused rather than billed as zero."
            )
        return cls()

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        """Zero per interval; the one line that changes when curtailment gets a
        price."""
        self._total_inr += 0.0

    def close_period(self) -> None:
        """Nothing settles at a billing boundary."""

    def lines(self) -> Mapping[str, float]:
        """Empty while the fleet declares no PV term."""
        return {}


@dataclass(frozen=True, kw_only=True)
class PvWattsModel:
    """The array, bound to its parameters, its member and the plant step."""

    member_id: str
    spec: PvWattsSpec
    dt_minutes: int

    @property
    def s_max_kva(self) -> float:
        return self.spec.s_max_kva

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        """An injector: it cannot draw."""
        return 0.0, self.spec.p_ac_max_kw

    # ---- the physics ------------------------------------------------------

    def cell_temperature_c(self, g_poa_w_m2: float, temp_air_c: float) -> float:
        """T_cell = T_amb + (NOCT - 20) * (G_poa / 800). ADR-0003 D5.

        Linear in irradiance, no wind term, no thermal mass; `noct_c` absorbs the
        average error.
        """
        return temp_air_c + (self.spec.noct_c - _NOCT_REFERENCE_AMBIENT_C) * (
            g_poa_w_m2 / _NOCT_REFERENCE_IRRADIANCE_W_M2
        )

    def effective_irradiance(self, g_poa_w_m2: float) -> float:
        """ADR-0003 D5a's correction, folded into EFFECTIVE IRRADIANCE rather
        than applied to power -- pvlib's idiom, so later corrections compose.

        Clamped to (0, 1]: the ratio underflows to 0.0 for denormal irradiance
        and log(0) raises, so the bound needs the guard below.
        """
        if g_poa_w_m2 <= 0.0:
            return 0.0
        ratio = g_poa_w_m2 / self.spec.low_light_g_ref_w_m2
        if ratio <= 0.0:
            return g_poa_w_m2 * _ETA_FLOOR
        eta = 1.0 + self.spec.low_light_k * math.log(ratio)
        return g_poa_w_m2 * min(1.0, max(_ETA_FLOOR, eta))

    def dc_power_kw(self, g_eff_w_m2: float, temp_cell_c: float) -> float:
        return float(
            pvwatts_dc(
                effective_irradiance=g_eff_w_m2,
                temp_cell=temp_cell_c,
                pdc0=self.spec.p_dc0_kw,
                gamma_pdc=self.spec.gamma_pdc,
                temp_ref=_TEMP_REF_C,
            )
        )

    def available_ac_kw(self, g_poa_w_m2: float, temp_air_c: float) -> float:
        """The whole chain, before any dispatcher curtailment."""
        if g_poa_w_m2 <= 0.0:
            return 0.0
        temp_cell = self.cell_temperature_c(g_poa_w_m2, temp_air_c)
        p_dc_kw = self.dc_power_kw(self.effective_irradiance(g_poa_w_m2), temp_cell)
        p_dc_kw *= (1.0 + self.spec.bifacial_gain_pct) * self.spec.soiling * self.spec.system_derate
        p_ac_kw = p_dc_kw * self.spec.inverter_eff
        return max(0.0, min(p_ac_kw, self.spec.p_ac_max_kw))

    # ---- the AssetModel template -----------------------------------------

    def required_scenario_fields(self) -> frozenset[str]:
        """The irradiance and ambient the chain reads."""
        return frozenset({"poa_irradiance_w_m2", "temp_air_c"})

    def initial_state(self, carry_in: Mapping[str, float]) -> PvWattsState:
        if carry_in:
            raise ValueError(
                f"asset {self.member_id!r}: this pv model carries no state, but carry_in "
                f"declares {sorted(carry_in)}. Refused rather than ignored."
            )
        return PvWattsState()

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        self.narrow(state)
        if not row.available[self.member_id]:
            # Offline is the whole inverter, so no reactive either (D11).
            return AssetEnvelope(p_min_kw=0.0, p_max_kw=0.0)
        return AssetEnvelope(
            p_min_kw=0.0,
            p_max_kw=self.available_ac_kw(row.poa_irradiance_w_m2, row.temp_air_c),
            q_min_kvar=-self.spec.s_max_kva,
            q_max_kvar=self.spec.s_max_kva,
            s_max_kva=self.spec.s_max_kva,
        )

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[PvWattsState, AssetDelivered]:
        pv_state = self.narrow(state)
        assert cmd.p_setpoint_kw is not None, "the port refuses an absent setpoint"
        setpoint_kw = cmd.p_setpoint_kw
        violations: list[str] = []

        if not row.available[self.member_id]:
            # Forced off (D11). Nothing was available, so nothing was
            # curtailed; a positive setpoint to a dead array is recorded.
            if setpoint_kw > 0.0:
                violations.append(f"pv_unavailable:{setpoint_kw:.1f}")
            return pv_state, AssetDelivered(
                p_net_kw=0.0, q_net_kvar=0.0, curtailed_kw=0.0, violations=tuple(violations)
            )

        available_kw = self.available_ac_kw(row.poa_irradiance_w_m2, row.temp_air_c)
        if setpoint_kw < 0.0:
            # An injector asked to draw. Clip to zero and say so (D2).
            violations.append(f"pv_negative_setpoint:{setpoint_kw:.1f}")

        # THE SETPOINT IS A CEILING, not a target: above available is not an
        # error and is not recorded, below available is curtailment (D4).
        p_kw = min(max(0.0, setpoint_kw), available_kw)
        return pv_state, AssetDelivered(
            p_net_kw=p_kw,
            q_net_kvar=0.0,  # the port clips reactive against the kVA circle
            curtailed_kw=available_kw - p_kw,
            violations=tuple(violations),
        )

    def build_cost(self, declared: tuple[CostTerm, ...]) -> PvWattsCost:
        return PvWattsCost.build(member_id=self.member_id, declared=declared)

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        assert (
            delivered.p_net_kw >= 0.0
        ), f"pv {self.member_id}: injector sign rule broken, p_net_kw={delivered.p_net_kw}"

    def narrow(self, state: AssetState) -> PvWattsState:
        if not isinstance(state, PvWattsState):
            raise TypeError(f"this pv model expects PvWattsState, got {type(state).__name__}")
        return state
