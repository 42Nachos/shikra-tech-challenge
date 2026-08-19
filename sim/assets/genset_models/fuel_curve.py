"""The datasheet genset: a quadratic fuel curve and a minimum stable load.

ONE MODULE FOR BOTH HALVES: the physics, and the cost the physics determines.
Fuel is litres per hour at a load and only this curve knows how many.

WHAT THIS MODEL DOES NOT HAVE: no minimum run or down time, so a price-driven
policy may cycle the set as fast as the plant steps (ADR-0006, still open); no
thermal derate; no start transient. Each is a reason for a SECOND model beside
this one.

THE CURVE RAISES OUTSIDE ITS FIT DOMAIN rather than clamping: a concave quadratic
extrapolated far enough returns negative fuel.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import AssetCommand, CostTerm, hours_in
from sim.asset import AssetDelivered, AssetEnvelope, AssetState, GensetDelivered
from sim.scenario import Interval, ScenarioRow


@dataclass(frozen=True, kw_only=True)
class GensetState(AssetState):
    """`on` survives in state so a start cost can be edge-triggered by comparing
    it against the previous interval. No such term exists yet: `start_cost` is an
    honest absence in IMPLEMENTED_COST_TERMS."""

    on: bool = False
    run_hours: float = 0.0


class FuelCurve(BaseModel):
    """L/h = a + b*P_kw + c*P_kw^2. See ADR-0003 D3."""

    model_config = ConfigDict(frozen=True)

    a: float
    b: float
    c: float


class FuelCurveSpec(BaseModel):
    """Cummins C200D2R (QSB7), datasheet D-6570, PRIME rating. ADR-0003 D2-D4."""

    model_config = ConfigDict(frozen=True)

    p_rated_kw: float = Field(gt=0)
    p_min_stable_kw: float = Field(gt=0)
    fuel_curve: FuelCurve
    fuel_curve_domain_kw: tuple[float, float]
    # No-load fuel burn: what the set consumes with its breaker closed and no
    # output. An INDEPENDENT figure, never the fuel curve evaluated at zero --
    # the curve is fitted to 25/50/75/100% load points and carries no
    # information about no-load. NO DEFAULT: a genset declared without this
    # idles for free.
    idle_fuel_lph: float = Field(ge=0)
    max_avg_load_factor_24h: float = Field(gt=0, le=1)
    # The last PRICE in a spec block, and it stays only because the `start_cost`
    # term is not implemented, so declaring it as cost data would be refused. It
    # moves onto a cost term the moment the edge-trigger lands.
    start_cost_inr: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_ranges(self) -> Self:
        if not (0 < self.p_min_stable_kw < self.p_rated_kw):
            raise ValueError("p_min_stable_kw must lie strictly within (0, p_rated_kw)")
        lo, hi = self.fuel_curve_domain_kw
        if not (0 < lo < hi <= self.p_rated_kw):
            raise ValueError("fuel_curve_domain_kw must be increasing and within rating")
        return self

    def build(self, member_id: str, dt_minutes: int) -> FuelCurveModel:
        """This spec's model, bound to its member and the plant step.

        ON THE SPEC, so there is no second table mapping names to classes.
        """
        return FuelCurveModel(member_id=member_id, spec=self, dt_minutes=dt_minutes)


def deliver(commanded_kw: float, spec: FuelCurveSpec) -> tuple[float, tuple[str, ...]]:
    """Clip-and-report, with the asymmetry ADR-0003 D4 requires.

    An explicit zero is "off": no violation. Below the minimum stable load and
    above zero, delivered output is zero rather than clipped UP to the floor.
    Above rated clips down.

    Whether the unit is then idling or shut down is `step()`'s business: this
    sees a setpoint, not the `on` channel.
    """
    if commanded_kw <= 0.0:
        return 0.0, ()

    if commanded_kw < spec.p_min_stable_kw:
        violation = f"genset_under_min_stable_idled:{commanded_kw:.1f}<{spec.p_min_stable_kw:.1f}"
        return 0.0, (violation,)

    if commanded_kw > spec.p_rated_kw:
        violation = f"genset_over_rated_clipped:{commanded_kw:.1f}->{spec.p_rated_kw:.1f}"
        return spec.p_rated_kw, (violation,)

    return commanded_kw, ()


def fuel_lph(p_kw: float, spec: FuelCurveSpec) -> float:
    """L/h = a + b*P + c*P^2. Raises outside the fit domain -- never
    extrapolate a concave quadratic; it eventually returns negative fuel.
    """
    if p_kw <= 0.0:
        return 0.0

    lo, hi = spec.fuel_curve_domain_kw
    if not (lo <= p_kw <= hi):
        raise ValueError(f"fuel_lph: {p_kw:.1f} kW is outside the fit domain [{lo}, {hi}] kW")

    c = spec.fuel_curve
    return c.a + c.b * p_kw + c.c * p_kw**2


@dataclass(frozen=True, kw_only=True)
class FuelCurveModel:
    """The set's physics, bound to its datasheet and the plant step."""

    member_id: str
    spec: FuelCurveSpec
    dt_minutes: int

    @property
    def s_max_kva(self) -> float:
        """No reactive model: excitation is not modelled, so the apparent-power
        rating equals the real one. A placeholder, not a claim about the set."""
        return self.spec.p_rated_kw

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        """An injector: it cannot draw. The floor is zero rather than the minimum
        stable load, because OFF is a legal operating point."""
        return 0.0, self.spec.p_rated_kw

    def required_scenario_fields(self) -> frozenset[str]:
        """Nothing exogenous: the curve runs off the command alone."""
        return frozenset()

    def initial_state(self, carry_in: Mapping[str, float]) -> GensetState:
        if carry_in:
            raise ValueError(
                f"asset {self.member_id!r}: this genset model starts stopped with zero run "
                f"hours, but carry_in declares {sorted(carry_in)}. Refused rather than ignored."
            )
        return GensetState()

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        self.narrow(state)
        if not row.available[self.member_id]:
            return AssetEnvelope(p_min_kw=0.0, p_max_kw=0.0)
        return AssetEnvelope(p_min_kw=0.0, p_max_kw=self.spec.p_rated_kw)

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[GensetState, GensetDelivered]:
        """`on` and the P setpoint are separate intentions and BOTH are honoured
        literally (ADR-0003 D4 amendment):

            on=False, p=0                       off. The only quiet way to be off.
            on=False, p>0                       a contradiction. Delivers nothing, recorded.
            on=True,  p in [min_stable, rated]  runs at the setpoint.
            on=True,  p = 0                     SPINNING IDLE: breaker closed, output
                                                zero, burning `idle_fuel_lph`. No
                                                violation -- a legitimate operating point.
            on=True,  0 < p < min_stable        idles too, and keeps a violation: the
                                                machine cannot be loaded below its floor.
            on=True,  p > rated                 clips to rated, recorded.

        IDLING IS NOT FREE and is not the same as being off; both deliver 0.0 kW,
        which is why the record carries `running`.
        """
        genset_state = self.narrow(state)
        assert cmd.p_setpoint_kw is not None, "the port refuses an absent setpoint"
        setpoint_kw = cmd.p_setpoint_kw
        commanded_on = cmd.on if cmd.on is not None else setpoint_kw > 0.0
        violations: list[str] = []

        if not row.available[self.member_id]:
            if commanded_on:
                violations.append(f"genset_unavailable:{setpoint_kw:.1f}")
            delivered_kw = 0.0
            running = False
        elif not commanded_on:
            if setpoint_kw > 0.0:
                violations.append(f"genset_off_but_commanded:{setpoint_kw:.1f}")
            delivered_kw = 0.0
            running = False
        elif setpoint_kw < 0.0:
            violations.append(f"genset_negative_setpoint:{setpoint_kw:.1f}")
            delivered_kw = 0.0
            running = True  # breaker closed, and a sign error does not open it
        else:
            delivered_kw, deliver_violations = deliver(setpoint_kw, self.spec)
            violations.extend(deliver_violations)
            running = True

        next_state = GensetState(
            on=running,
            run_hours=genset_state.run_hours + (hours_in(self.dt_minutes) if running else 0.0),
        )
        return next_state, GensetDelivered(
            p_net_kw=delivered_kw,
            q_net_kvar=0.0,  # measurement; no reactive model yet
            running=running,
            violations=tuple(violations),
        )

    def build_cost(self, declared: tuple[CostTerm, ...]) -> FuelCurveCost:
        """Passes its own spec: the fuel bill is the curve evaluated at a load."""
        return FuelCurveCost.build(member_id=self.member_id, spec=self.spec, declared=declared)

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        """An injector: it cannot draw (ADR-0004 D2)."""
        assert delivered.p_net_kw >= 0.0, (
            f"genset {self.member_id}: injector sign rule broken, " f"p_net_kw={delivered.p_net_kw}"
        )

    def narrow(self, state: AssetState) -> GensetState:
        if not isinstance(state, GensetState):
            raise TypeError(f"the fuel_curve model expects GensetState, got {type(state).__name__}")
        return state


class FuelParams(BaseModel):
    """The PRICE of fuel, never the curve.

    The L/h curve is a model parameter and lives in the spec (ADR-0003 D3);
    declaring the coefficients here too would be a second fuel curve.
    """

    model_config = ConfigDict(frozen=True)

    price_inr_per_litre: float = Field(gt=0)


@dataclass(kw_only=True)
class FuelCurveCost:
    """What the set burns, priced. ADR-0004 D7, amended by ADR-0016 D3.

    Holds the SPEC, not a copy of its curve, so the testbed keeps exactly one
    fuel curve.

    THE IDLE BRANCH NEVER CALLS THE CURVE: `fuel_lph()` raises below its fit
    domain, and no-load is far below it, so a turning set that is not producing
    burns the separately declared `idle_fuel_lph`. Reads
    `GensetDelivered.running`, since a running and a stopped set both deliver
    0.0 kW.
    """

    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {"fuel": FuelParams}
    KINDS: ClassVar[frozenset[str]] = frozenset(KIND_PARAMS)

    spec: FuelCurveSpec
    declared: tuple[str, ...]
    params: FuelParams
    _total_inr: float = 0.0

    @classmethod
    def build(
        cls, *, member_id: str, spec: FuelCurveSpec, declared: tuple[CostTerm, ...]
    ) -> FuelCurveCost:
        kinds = [term.kind for term in declared]
        unknown = [k for k in kinds if k not in cls.KINDS]
        if unknown:
            raise ValueError(
                f"asset {member_id!r}: the genset prices {sorted(cls.KINDS)}, but the fleet "
                f"declares {sorted(set(unknown))}. A term nobody implements must be refused, "
                f"not billed as zero."
            )
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"asset {member_id!r}: duplicate cost term kinds {sorted(kinds)}")

        fuel = next((t for t in declared if t.kind == "fuel"), None)
        params = FuelParams.model_validate(
            dict(fuel.params) if fuel is not None else {"price_inr_per_litre": 1.0}
        )
        return cls(
            spec=spec,
            declared=tuple(k for k in cls.KIND_PARAMS if k in kinds),
            params=params,
        )

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        if delivered.p_net_kw > 0.0:
            lph = fuel_lph(delivered.p_net_kw, self.spec)
        elif isinstance(delivered, GensetDelivered) and delivered.running:
            lph = self.spec.idle_fuel_lph
        else:
            return
        self._total_inr += lph * hours_in(iv.dt_minutes) * self.params.price_inr_per_litre

    def close_period(self) -> None:
        """Nothing settles at a billing boundary: fuel is burnt per interval."""

    def lines(self) -> Mapping[str, float]:
        return {"fuel": self._total_inr} if self.declared else {}
