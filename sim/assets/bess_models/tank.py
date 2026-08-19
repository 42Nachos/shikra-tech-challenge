"""The energy-tank storage model. ADR-0003 D8.

    soc[k+1] = soc[k] + (p_charge * eta_c - p_discharge / eta_d) * dt

No electrochemistry: one number for stored energy, two one-way efficiencies, a
band it may move inside, and a rate limit at the terminals.

TWO EFFICIENCIES, NOT ONE ROUND TRIP (ADR-0003 D8), applied at the TERMINALS
both ways: delivering P kW draws `P / eta_discharge` from the tank, and drawing
P kW puts `P * eta_charge` into it.

WHAT THIS MODEL DOES NOT HAVE: no power-vs-SOC taper beyond the energy limit,
which ADR-0003 calls the plant's single largest known optimism; no temperature
term; no capacity fade. Each is a reason to write a SECOND model beside this one.
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

# The lowest round trip this model will accept. Below anything real (flow
# batteries reach the mid 0.60s), so its job is catching 0.09 typed for 0.90.
_ROUND_TRIP_FLOOR = 0.50


class TankSpec(BaseModel):
    """The tank's parameters.

    Sizing figures are placeholders for site_a: no BESS datasheet is in hand.
    The validator accepts a RANGE of round trips rather than a pin -- site_a
    assumes 0.95, the IIT Madras pack measures 0.883.
    """

    model_config = ConfigDict(frozen=True)

    energy_nom_kwh: float = Field(gt=0)
    p_max_charge_kw: float = Field(gt=0)
    p_max_discharge_kw: float = Field(gt=0)
    eta_charge: float = Field(gt=0, le=1)
    eta_discharge: float = Field(gt=0, le=1)
    soc_min_kwh: float = Field(ge=0)
    soc_max_kwh: float = Field(gt=0)
    # The inverter this store is wired behind, for the ADR-0004 D3 kVA circle.
    # Every storage model declares one.
    s_max_kva: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if not (0 <= self.soc_min_kwh < self.soc_max_kwh <= self.energy_nom_kwh):
            raise ValueError("require 0 <= soc_min_kwh < soc_max_kwh <= energy_nom_kwh")
        p_max_kw = max(self.p_max_charge_kw, self.p_max_discharge_kw)
        if self.s_max_kva < p_max_kw:
            raise ValueError(
                f"s_max_kva={self.s_max_kva} < the larger power limit {p_max_kw}: real power "
                f"alone cannot exceed apparent power"
            )
        # A RANGE, not a pin, and what a per-leg bound cannot see: above 1.0 the
        # pair generates energy, below the floor it is a misplaced decimal.
        round_trip = self.eta_charge * self.eta_discharge
        if not _ROUND_TRIP_FLOOR <= round_trip <= 1.0:
            raise ValueError(
                f"eta_charge * eta_discharge = {round_trip:.4f}, which must lie in "
                f"[{_ROUND_TRIP_FLOOR}, 1.0]. Above 1.0 the pair generates energy; below the "
                f"floor it is almost certainly a misplaced decimal. Declare the two legs "
                f"separately (ADR-0003 D8) -- a real round trip need not be any particular value."
            )
        return self

    def build(self, member_id: str, dt_minutes: int) -> TankModel:
        """This spec's model, bound to its member and the plant step.

        ON THE SPEC, so there is no second table mapping names to classes.
        """
        return TankModel(member_id=member_id, spec=self, dt_minutes=dt_minutes)


@dataclass(frozen=True, kw_only=True)
class TankState(AssetState):
    """What the tank must remember between intervals.

    `cumulative_throughput_kwh` counts energy MOVED at the terminals, either
    direction -- not energy stored. The degradation charge below is priced from
    it.

    `soh` is CARRIED AND NOT YET WRITTEN: nothing here fades capacity, so
    `step()` copies it forward untouched. RESERVED for capacity fade, not dead.
    """

    soc_kwh: float
    soh: float = 1.0
    cumulative_throughput_kwh: float = 0.0


@dataclass(frozen=True, kw_only=True)
class TankModel:
    """The tank, bound to its parameters and the plant step. Frozen and holding
    only numbers."""

    member_id: str
    spec: TankSpec
    dt_minutes: int

    @property
    def s_max_kva(self) -> float:
        """The inverter rating the port clips reactive power against."""
        return self.spec.s_max_kva

    @property
    def energy_capacity_kwh(self) -> float:
        """RATED energy, never the band and never the current SOC (ADR-0006)."""
        return self.spec.energy_nom_kwh

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        """Nameplate rate limits, unconditioned by SOC: (p_min, p_max)."""
        return -self.spec.p_max_charge_kw, self.spec.p_max_discharge_kw

    def required_scenario_fields(self) -> frozenset[str]:
        """Nothing exogenous: the tank runs off state and command alone."""
        return frozenset()

    def initial_state(self, carry_in: Mapping[str, float]) -> TankState:
        """Requires exactly soc_kwh, inside the band. No default."""
        if set(carry_in) != {"soc_kwh"}:
            raise ValueError(
                f"asset {self.member_id!r}: the tank model's carry_in must declare exactly "
                f"soc_kwh, got {sorted(carry_in) or 'nothing'}. A starting SOC has no default."
            )
        soc_kwh = carry_in["soc_kwh"]
        if not (self.spec.soc_min_kwh <= soc_kwh <= self.spec.soc_max_kwh):
            raise ValueError(
                f"asset {self.member_id!r}: carry_in soc_kwh={soc_kwh} outside the SOC band "
                f"[{self.spec.soc_min_kwh}, {self.spec.soc_max_kwh}]"
            )
        return TankState(soc_kwh=soc_kwh)

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        """Rate limits narrowed by SOC. The two headroom expressions are the same
        ones `step()` clips against."""
        tank = self.narrow(state)
        if not row.available[self.member_id]:
            return AssetEnvelope(p_min_kw=0.0, p_max_kw=0.0)
        spec = self.spec
        hours = hours_in(self.dt_minutes)
        discharge_soc_limit_kw = max(
            0.0, (tank.soc_kwh - spec.soc_min_kwh) * spec.eta_discharge / hours
        )
        charge_soc_limit_kw = max(
            0.0, (spec.soc_max_kwh - tank.soc_kwh) / (spec.eta_charge * hours)
        )
        return AssetEnvelope(
            p_min_kw=-min(spec.p_max_charge_kw, charge_soc_limit_kw),
            p_max_kw=min(spec.p_max_discharge_kw, discharge_soc_limit_kw),
            q_min_kvar=-spec.s_max_kva,
            q_max_kvar=spec.s_max_kva,
            s_max_kva=spec.s_max_kva,
        )

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[TankState, AssetDelivered]:
        """Positive commanded_kw = discharge, negative = charge (contracts.py).

        Clipped twice in each direction -- rate limit, then SOC headroom -- and
        each clip reported. Every `p_soc_limit` is the exact inverse of the
        recursion beside it, so discharging at the limit lands exactly on
        `soc_min_kwh`.
        """
        tank = self.narrow(state)
        spec = self.spec
        soc_kwh = tank.soc_kwh
        hours = hours_in(self.dt_minutes)
        assert cmd.p_setpoint_kw is not None, "the port refuses an absent setpoint"
        commanded_kw = cmd.p_setpoint_kw
        violations: list[str] = []

        if not row.available[self.member_id]:
            # Forced idle: no flow either direction, state held (D11). Tests
            # != 0, since a commanded CHARGE to a dead battery is equally an
            # error.
            if commanded_kw != 0.0:
                violations.append(f"bess_unavailable:{commanded_kw:.1f}")
            return tank, AssetDelivered(p_net_kw=0.0, q_net_kvar=0.0, violations=tuple(violations))

        if commanded_kw >= 0.0:
            p_rate_clipped = min(commanded_kw, spec.p_max_discharge_kw)
            if commanded_kw > spec.p_max_discharge_kw:
                violations.append(
                    f"bess_discharge_rate_clipped:{commanded_kw:.1f}->{p_rate_clipped:.1f}"
                )

            p_soc_limit = max(0.0, (soc_kwh - spec.soc_min_kwh) * spec.eta_discharge / hours)
            delivered_kw = min(p_rate_clipped, p_soc_limit)
            if p_rate_clipped > p_soc_limit + _EPS:
                violations.append(
                    f"bess_discharge_soc_clipped:{p_rate_clipped:.1f}->{delivered_kw:.1f}"
                )

            new_soc_kwh = soc_kwh - (delivered_kw / spec.eta_discharge) * hours
        else:
            p_requested = -commanded_kw
            p_rate_clipped = min(p_requested, spec.p_max_charge_kw)
            if p_requested > spec.p_max_charge_kw:
                violations.append(
                    f"bess_charge_rate_clipped:{p_requested:.1f}->{p_rate_clipped:.1f}"
                )

            p_soc_limit = max(0.0, (spec.soc_max_kwh - soc_kwh) / (spec.eta_charge * hours))
            p_charge = min(p_rate_clipped, p_soc_limit)
            if p_rate_clipped > p_soc_limit + _EPS:
                violations.append(f"bess_charge_soc_clipped:{p_rate_clipped:.1f}->{p_charge:.1f}")

            new_soc_kwh = soc_kwh + p_charge * spec.eta_charge * hours
            delivered_kw = -p_charge

        next_state = TankState(
            soc_kwh=new_soc_kwh,
            # Carried, not recomputed: this model does not fade capacity. See
            # TankState on why the field is here ahead of its consumer.
            soh=tank.soh,
            cumulative_throughput_kwh=(tank.cumulative_throughput_kwh + abs(delivered_kw) * hours),
        )
        return next_state, AssetDelivered(
            p_net_kw=delivered_kw,
            q_net_kvar=0.0,  # the port clips reactive against the kVA circle
            violations=tuple(violations),
        )

    def build_cost(self, declared: tuple[CostTerm, ...]) -> TankCost:
        return TankCost.build(member_id=self.member_id, declared=declared)

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        """The tank's conservation tripwire, in place of an injector sign rule.

        BOTH states, not interchangeable: the envelope is the one this interval
        OPENED with, so it comes from `state_before`; the SOC band check is on
        `state_after`, the value that has to be legal.
        """
        delivered_kw = delivered.p_net_kw
        envelope = self.envelope(state_before, row)
        assert envelope.p_min_kw - _EPS <= delivered_kw <= envelope.p_max_kw + _EPS, (
            f"bess {self.member_id}: delivered {delivered_kw} kW outside envelope "
            f"[{envelope.p_min_kw}, {envelope.p_max_kw}]"
        )
        soc_kwh = self.narrow(state_after).soc_kwh
        assert self.spec.soc_min_kwh - _EPS <= soc_kwh <= self.spec.soc_max_kwh + _EPS, (
            f"bess {self.member_id}: SOC {soc_kwh} kWh left the band "
            f"[{self.spec.soc_min_kwh}, {self.spec.soc_max_kwh}]"
        )

    def observed_soc_kwh(self, state: AssetState) -> float:
        """The true SOC. The port degrades it through the noise block."""
        return self.narrow(state).soc_kwh

    def narrow(self, state: AssetState) -> TankState:
        """Each model narrows the state it was handed; the port cannot."""
        if not isinstance(state, TankState):
            raise TypeError(f"tank model expects TankState, got {type(state).__name__}")
        return state


class DegradationParams(BaseModel):
    """Flat per-kWh-throughput charge. The `degradation` term's declared params."""

    model_config = ConfigDict(frozen=True)

    inr_per_kwh_throughput: float = Field(ge=0)


@dataclass(kw_only=True)
class TankCost:
    """What the tank costs, priced from quantities only the tank counts.

    ONE term today: `degradation`, a flat charge per kWh of terminal throughput,
    accumulated per interval and not path-dependent. Rainflow over the full SOC
    path is a function of the state TRAJECTORY and is why this lives beside the
    model rather than in the ledger.

    NOT frozen, unlike the model beside it: this accumulates.
    """

    # The kinds this model prices, and the params model each validates against.
    # Two readers: `KINDS` for the refusal below, and the params class for
    # anything parsing a declared term without a ledger (the ADR-0008 oracle).
    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {"degradation": DegradationParams}
    KINDS: ClassVar[frozenset[str]] = frozenset({"degradation"})

    params: DegradationParams
    _total_inr: float = 0.0

    @classmethod
    def build(cls, *, member_id: str, declared: tuple[CostTerm, ...]) -> TankCost:
        """Validate this member's declared terms into the tank's cost model.

        Validated at construction, which is startup, so a mistyped tariff refuses
        to start rather than mispricing a run.
        """
        kinds = [term.kind for term in declared]
        unknown = [k for k in kinds if k not in cls.KINDS]
        if unknown:
            raise ValueError(
                f"asset {member_id!r}: the tank model prices {sorted(cls.KINDS)}, but the "
                f"fleet declares {sorted(set(unknown))}. A term nobody implements must be "
                f"refused, not billed as zero."
            )
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"asset {member_id!r}: duplicate cost term kinds {sorted(kinds)}")

        degradation = next((t for t in declared if t.kind == "degradation"), None)
        params = DegradationParams.model_validate(
            dict(degradation.params) if degradation is not None else {"inr_per_kwh_throughput": 0.0}
        )
        return cls(params=params, declares_degradation=degradation is not None)

    # Whether the fleet declared the term. A member that declares none emits no
    # line: `CostBreakdown` is one line per DECLARED (member, term).
    declares_degradation: bool = True

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        """Throughput counts BOTH legs, so this is |p_net|."""
        self._total_inr += (
            abs(delivered.p_net_kw) * hours_in(iv.dt_minutes) * self.params.inr_per_kwh_throughput
        )

    def close_period(self) -> None:
        """Nothing settles at a billing boundary: the charge is per interval."""

    def lines(self) -> Mapping[str, float]:
        """One entry per DECLARED term, keyed by kind; the ledger prefixes the
        member id."""
        return {"degradation": self._total_inr} if self.declares_degradation else {}
