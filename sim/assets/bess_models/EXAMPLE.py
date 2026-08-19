"""A lossless tank. The dumbest storage model that is not `tank`.

REFERENCE SUBMISSION, held on the truth side and never shipped in the pack. Its
only job is to be a SECOND model in a family, so the pipeline has something to
register, assemble, run and score. It is deliberately not good: no efficiency, no
taper, no temperature. Anything a candidate submits should beat it on physics.

What it does prove is that a second storage model can exist at all -- which was
false until the port stopped narrowing to `TankModel` by name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import AssetCommand, CostTerm, hours_in
from sim.asset import AssetDelivered, AssetEnvelope, AssetState
from sim.scenario import Interval, ScenarioRow

_EPS = 1e-9


class FlatSpec(BaseModel):
    """Sizing and the band. No efficiency: that is the point.

    EVERY FIELD CARRIES ITS OWN VALUE, so config selects this model with an empty
    params block. That is the track 2 arrangement and a deliberate exception to
    the repository rule that physical values live in config: the exercise is
    whether a model satisfies the port, not whether a candidate can fill in a
    YAML. The numbers are the shipped pack's, so a run against this model is
    comparable to a run against the tank.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    energy_nom_kwh: float = Field(default=2000.0, gt=0)
    p_max_charge_kw: float = Field(default=500.0, gt=0)
    p_max_discharge_kw: float = Field(default=500.0, gt=0)
    soc_min_kwh: float = Field(default=200.0, ge=0)
    soc_max_kwh: float = Field(default=2000.0, gt=0)
    s_max_kva: float = Field(default=550.0, gt=0)

    @model_validator(mode="after")
    def _check_band(self) -> Self:
        if self.soc_min_kwh >= self.soc_max_kwh:
            raise ValueError(f"soc_min_kwh={self.soc_min_kwh} is not below {self.soc_max_kwh}")
        if self.soc_max_kwh > self.energy_nom_kwh:
            raise ValueError(f"soc_max_kwh={self.soc_max_kwh} exceeds {self.energy_nom_kwh}")
        return self

    def build(self, member_id: str, dt_minutes: int) -> FlatModel:
        return FlatModel(member_id=member_id, spec=self, dt_minutes=dt_minutes)


@dataclass(frozen=True, kw_only=True)
class FlatState(AssetState):
    soc_kwh: float
    cumulative_throughput_kwh: float = 0.0


@dataclass(frozen=True)
class FlatModel:
    member_id: str
    spec: FlatSpec
    dt_minutes: int

    @property
    def s_max_kva(self) -> float:
        return self.spec.s_max_kva

    @property
    def energy_capacity_kwh(self) -> float:
        return self.spec.energy_nom_kwh

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        return -self.spec.p_max_charge_kw, self.spec.p_max_discharge_kw

    def required_scenario_fields(self) -> frozenset[str]:
        """Nothing exogenous."""
        return frozenset()

    def initial_state(self, carry_in: Mapping[str, float]) -> FlatState:
        if "soc_kwh" not in carry_in:
            raise ValueError(f"asset {self.member_id!r}: carry_in must declare soc_kwh")
        soc_kwh = carry_in["soc_kwh"]
        if not self.spec.soc_min_kwh <= soc_kwh <= self.spec.soc_max_kwh:
            raise ValueError(
                f"asset {self.member_id!r}: carry_in soc_kwh={soc_kwh} outside "
                f"[{self.spec.soc_min_kwh}, {self.spec.soc_max_kwh}]"
            )
        return FlatState(soc_kwh=soc_kwh)

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        pack = self.narrow(state)
        if not row.available[self.member_id]:
            return AssetEnvelope(p_min_kw=0.0, p_max_kw=0.0)
        spec, hours = self.spec, hours_in(self.dt_minutes)
        return AssetEnvelope(
            p_min_kw=-min(spec.p_max_charge_kw, (spec.soc_max_kwh - pack.soc_kwh) / hours),
            p_max_kw=min(spec.p_max_discharge_kw, (pack.soc_kwh - spec.soc_min_kwh) / hours),
            q_min_kvar=-spec.s_max_kva,
            q_max_kvar=spec.s_max_kva,
            s_max_kva=spec.s_max_kva,
        )

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[FlatState, AssetDelivered]:
        pack = self.narrow(state)
        assert cmd.p_setpoint_kw is not None, "the port refuses an absent setpoint"
        commanded_kw = cmd.p_setpoint_kw
        violations: list[str] = []

        if not row.available[self.member_id]:
            if commanded_kw != 0.0:
                violations.append(f"bess_unavailable:{commanded_kw:.1f}")
            return pack, AssetDelivered(p_net_kw=0.0, q_net_kvar=0.0, violations=tuple(violations))

        limits = self.envelope(pack, row)
        delivered_kw = min(limits.p_max_kw, max(limits.p_min_kw, commanded_kw))
        if abs(delivered_kw - commanded_kw) > _EPS:
            violations.append(f"bess_flat_clipped:{commanded_kw:.1f}->{delivered_kw:.1f}")

        hours = hours_in(self.dt_minutes)
        return FlatState(
            soc_kwh=pack.soc_kwh - delivered_kw * hours,
            cumulative_throughput_kwh=pack.cumulative_throughput_kwh + abs(delivered_kw) * hours,
        ), AssetDelivered(p_net_kw=delivered_kw, q_net_kvar=0.0, violations=tuple(violations))

    def build_cost(self, declared: tuple[CostTerm, ...]) -> FlatCost:
        return FlatCost.build(member_id=self.member_id, declared=declared)

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        """Lossless: the energy that left equals the energy delivered."""
        before, after = self.narrow(state_before), self.narrow(state_after)
        moved_kwh = before.soc_kwh - after.soc_kwh
        expected_kwh = delivered.p_net_kw * hours_in(self.dt_minutes)
        if abs(moved_kwh - expected_kwh) > 1e-6:
            raise AssertionError(
                f"asset {self.member_id!r}: soc moved {moved_kwh} against {expected_kwh} delivered"
            )

    def observed_soc_kwh(self, state: AssetState) -> float:
        return self.narrow(state).soc_kwh

    def narrow(self, state: AssetState) -> FlatState:
        if not isinstance(state, FlatState):
            raise TypeError(
                f"asset {self.member_id!r} expects FlatState, got {type(state).__name__}"
            )
        return state


class FlatDegradationParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inr_per_kwh_throughput: float = Field(ge=0)


@dataclass
class FlatCost:
    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {"degradation": FlatDegradationParams}
    KINDS: ClassVar[frozenset[str]] = frozenset({"degradation"})

    member_id: str
    params: FlatDegradationParams
    _total_inr: float = 0.0

    @classmethod
    def build(cls, *, member_id: str, declared: tuple[CostTerm, ...]) -> FlatCost:
        unknown = {t.kind for t in declared} - cls.KINDS
        if unknown:
            raise ValueError(f"asset {member_id!r} declares {sorted(unknown)}, not priced here")
        term = next((t for t in declared if t.kind == "degradation"), None)
        if term is None:
            raise ValueError(f"asset {member_id!r} declares no `degradation` term")
        return cls(
            member_id=member_id,
            params=FlatDegradationParams.model_validate(dict(term.params)),
        )

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        self._total_inr += (
            abs(delivered.p_net_kw) * hours_in(iv.dt_minutes) * self.params.inr_per_kwh_throughput
        )

    def close_period(self) -> None:
        """Nothing path-dependent."""

    def lines(self) -> Mapping[str, float]:
        return {"degradation": round(self._total_inr, 2)}
