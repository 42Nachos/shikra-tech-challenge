"""The Asset port: what every asset is, plus the config record it is built from.

ADR-0004 D1. An asset declares its identity, its ratings, which channels it
accepts, what it costs and what exogenous data it needs, and owns the methods
that produce its per-interval records. An asset is a PORT over a MODEL
(ADR-0016 D1); the physics lives in sim/assets/<asset>_models/.

Everything here is truth, or reveals it, so it is unreachable from dispatch/.
The dispatcher-visible counterparts:

    AssetEnvelope   (here)  ->  AssetRatings      (contracts.py)
    AssetDelivered  (here)  ->  AssetMeasurement  (contracts.py)
    AssetState      (here)  ->  nothing.

ResolvedAsset, at the bottom, is the exception: validated config, not truth.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from contracts import (
    AssetCapabilities,
    AssetCommand,
    AssetCostData,
    AssetMeasurement,
    AssetObservation,
    AssetRatings,
    CostTerm,
)
from sim.noise import add_noise
from sim.scenario import Interval, ScenarioRow


class AssetNoise(BaseModel):
    """The meter on any asset: the two channels of `AssetMeasurement`. ADR-0012 D2.

    Both are declared even where q is never commanded -- a capability says what
    may be COMMANDED, not what is MEASURED.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_meter_sigma_kw: float = Field(ge=0.0)
    q_meter_sigma_kvar: float = Field(ge=0.0)


class KvaPriority(StrEnum):
    """How an inverter resolves a command outside its kVA circle. ADR-0004 D3."""

    P = "p"  # hold P, shrink Q to fit the circle
    Q = "q"  # hold Q, shrink P
    SCALE = "scale"  # shrink both, preserving the power factor


_EPS = 1e-9  # numerical tolerance, same figure sim/plant.py uses; not physics


def _q_at_real_power_priority(
    *, p_net_kw: float, q_kvar: float, s_max_kva: float, label: str, violations: list[str]
) -> float:
    """Q shrunk to whatever reactive headroom `p_net_kw` leaves. ADR-0004 D3.

    `p` is held and only `q` moves, so no re-step is needed. Also the fallback
    after a `q`/`scale` re-step, where the model may not have reached the target
    P and the circle must hold against what it actually delivered.
    """
    if math.hypot(p_net_kw, q_kvar) <= s_max_kva + _EPS:
        return q_kvar
    headroom_kvar = math.sqrt(max(0.0, s_max_kva**2 - min(abs(p_net_kw), s_max_kva) ** 2))
    violations.append(f"{label}_kva_clip:{abs(q_kvar) - headroom_kvar:.1f}")
    return math.copysign(headroom_kvar, q_kvar)


def resolve_kva_circle(
    *,
    model: AssetModel,
    state: AssetState,
    cmd: AssetCommand,
    row: ScenarioRow,
    next_state: AssetState,
    delivered: AssetDelivered,
    s_max_kva: float,
    priority: KvaPriority,
    label: str,
) -> tuple[AssetState, AssetDelivered]:
    """Reactive delivery under the D3 kVA circle, shared by every inverter port.

    An uncommanded Q channel (None) delivers zero reactive: a non-declared
    channel has no command to be absent. P enters signed and only its magnitude
    matters -- the circle is symmetric in all four quadrants.

    Outside the circle, `kva_priority` decides which of P and Q gives way:

        p      hold P, shrink Q to the headroom. No re-step.
        q      hold Q, shrink |P| to sqrt(s_max^2 - q^2), then RE-STEP the model.
        scale  shrink both by s_max / hypot(p, q), preserving the power factor,
               then RE-STEP the model.

    `q` and `scale` reduce real power, and P was already delivered by
    `model.step()` along with a next state -- so the model is re-entered at the
    reduced setpoint and its first result discarded, the same manoeuvre
    `Plant.step()`'s island backoff performs.

    Q is then re-clipped at real-power priority against what the model ACTUALLY
    delivered. Defensive today: no shipped model can refuse a reduced |P|, since
    reducing it is always feasible when the larger value was. A model with a
    minimum discharge or a power-vs-SOC taper could, and the circle must hold
    when it does.

    VIOLATIONS: the first step's strings are kept and the re-step's discarded,
    because the first describe clipping against what the dispatcher commanded
    while the re-step's P was the port's own choice.
    """
    q_setpoint_kvar = cmd.q_setpoint_kvar
    if q_setpoint_kvar is None:
        return next_state, dataclasses.replace(delivered, q_net_kvar=0.0)

    p_net_kw = delivered.p_net_kw
    q_kvar = min(max(q_setpoint_kvar, -s_max_kva), s_max_kva)
    violations = list(delivered.violations)

    if math.hypot(p_net_kw, q_kvar) <= s_max_kva + _EPS or priority is KvaPriority.P:
        q_final = _q_at_real_power_priority(
            p_net_kw=p_net_kw,
            q_kvar=q_kvar,
            s_max_kva=s_max_kva,
            label=label,
            violations=violations,
        )
        return next_state, dataclasses.replace(
            delivered, q_net_kvar=q_final, violations=tuple(violations)
        )

    if priority is KvaPriority.Q:
        p_target_kw = math.copysign(math.sqrt(max(0.0, s_max_kva**2 - q_kvar**2)), p_net_kw)
    else:  # KvaPriority.SCALE, at constant power factor
        shrink = s_max_kva / math.hypot(p_net_kw, q_kvar)
        p_target_kw = p_net_kw * shrink
        q_kvar *= shrink

    re_state, re_delivered = model.step(
        state, dataclasses.replace(cmd, p_setpoint_kw=p_target_kw), row
    )
    model.check_delivery(state, re_state, re_delivered, row)
    violations.append(f"{label}_kva_p_backed_off:{p_net_kw - re_delivered.p_net_kw:.1f}")
    q_final = _q_at_real_power_priority(
        p_net_kw=re_delivered.p_net_kw,
        q_kvar=q_kvar,
        s_max_kva=s_max_kva,
        label=label,
        violations=violations,
    )
    return re_state, dataclasses.replace(
        re_delivered, q_net_kvar=q_final, violations=tuple(violations)
    )


class AssetCost(Protocol):
    """What an asset's own model charges. ADR-0004 D7, amended by ADR-0016 D3.

    The MODEL owns the arithmetic; the ledger owns only the loop and the
    calendar. MUTABLE, because it accumulates, which is why it is separate from
    the frozen asset.
    """

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        """Price one interval and add it to this member's running totals."""
        ...

    def close_period(self) -> None:
        """Settle anything path-dependent. Called by the ledger on the calendar."""
        ...

    def lines(self) -> Mapping[str, float]:
        """Cost-term kind -> rupees, one per kind the fleet declared, in a
        deterministic order. The ledger prefixes the member id."""
        ...


class AssetModel(Protocol):
    """The template every asset model satisfies. ADR-0016 D1.

    Every member is read-only, which is what lets a frozen dataclass satisfy it.
    A family may EXTEND this (storage adds `energy_capacity_kwh` and
    `observed_soc_kwh`) but may not narrow it.

    THE MODEL HANDLES AVAILABILITY, not the port: "off" means something
    different for each kind of equipment.
    """

    @property
    def member_id(self) -> str:
        """The fleet member this model was built for."""
        ...

    @property
    def dt_minutes(self) -> int:
        """The plant step this model was bound to."""
        ...

    @property
    def s_max_kva(self) -> float:
        """Apparent-power rating, for `AssetRatings` and the D3 kVA circle."""
        ...

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        """Nameplate (p_min, p_max), unconditioned by state or by this row."""
        ...

    def required_scenario_fields(self) -> frozenset[str]:
        """The exogenous columns this model reads. Availability is the port's,
        since it is keyed by member id and every asset has one."""
        ...

    def initial_state(self, carry_in: Mapping[str, float]) -> AssetState:
        """State at k=0, validated against what config declared. No defaults."""
        ...

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        """What the asset can ACTUALLY do this interval, given that state and
        this row. Plant-side only: it leaks state quantitatively."""
        ...

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[AssetState, AssetDelivered]:
        """(next state, what it delivered). Clip and REPORT, never silently fix.

        Returns the full delivered record, since what it must carry is per-model.
        `q_net_kvar` is left at 0.0 -- the PORT clips reactive and replaces it.
        """
        ...

    def build_cost(self, declared: tuple[CostTerm, ...]) -> AssetCost:
        """This model's cost, from the member's declared `cost_terms:`."""
        ...

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        """Assert whatever conservation law this model has.

        Both states, because the envelope is a function of the state the interval
        OPENED with while the legality check is on the state it closed with.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class AssetState:
    """One asset's TRUE mutable state. ADR-0004 D1. dispatch/ must never see it.

    Empty by design: only what CHANGES lives here, and that is per-MODEL, so each
    concrete state is declared beside its model (`PvWattsState` in
    sim/assets/pv_models/pvwatts.py, `TankState` in sim/assets/bess_models/tank.py).
    """


@dataclass(frozen=True, kw_only=True)
class AssetEnvelope:
    """What one asset can ACTUALLY do this interval: AssetRatings narrowed by
    true state and this row. ADR-0004 D3.

    PLANT-SIDE ONLY: it leaks state quantitatively, since a BESS's `p_max_kw`
    inverts to the exact SOC. The dispatcher gets AssetRatings instead.
    """

    p_min_kw: float  # most negative injection possible this interval
    p_max_kw: float  # largest injection possible this interval
    q_min_kvar: float = 0.0
    q_max_kvar: float = 0.0
    s_max_kva: float | None = None

    def __post_init__(self) -> None:
        if self.p_min_kw > self.p_max_kw:
            raise ValueError(
                f"envelope inverted: p_min_kw={self.p_min_kw} > p_max_kw={self.p_max_kw}"
            )
        if self.q_min_kvar > self.q_max_kvar:
            raise ValueError(
                f"envelope inverted: q_min_kvar={self.q_min_kvar} > q_max_kvar={self.q_max_kvar}"
            )
        if self.s_max_kva is not None and self.s_max_kva <= 0.0:
            raise ValueError(f"s_max_kva must be positive when set, got {self.s_max_kva}")


@dataclass(frozen=True, kw_only=True)
class AssetDelivered:
    """The FULL truth of what one asset did this interval. ADR-0004 D2.

        p_net_kw   > 0 injecting into the bus,  < 0 drawing from it
        q_net_kvar > 0 supplying reactive,      < 0 absorbing it

    One convention for every asset, so the plant closes the bus balance as a
    signed sum. The sign RESTRICTIONS are per-type and enforced by each asset's
    own step(), not here.

    `curtailed_kw` is recorded, never commanded (D4), and is NOT a bus flow.
    `measurement()` is the projection that may cross to dispatch/.
    """

    p_net_kw: float
    q_net_kvar: float = 0.0
    curtailed_kw: float = 0.0
    violations: tuple[str, ...] = ()

    def measurement(self) -> AssetMeasurement:
        """The dispatcher-visible projection. The ONLY sanctioned way to get one."""
        return AssetMeasurement(p_net_kw=self.p_net_kw, q_net_kvar=self.q_net_kvar)


@dataclass(frozen=True, kw_only=True)
class GensetDelivered(AssetDelivered):
    """A rotating set's delivered record, carrying whether it was TURNING.

    The one thing `p_net_kw` cannot say: an idling set and a stopped set both
    deliver 0.0 kW. `dataclasses.replace` preserves the subclass.
    """

    running: bool = False


@dataclass(frozen=True, kw_only=True)
class Asset:
    """One fleet element: PV, BESS, genset, grid, or anything added later.

    Subclass it, add the asset's own spec, and override the methods below. The
    static properties default to the conservative end, so a subclass states only
    what differs.
    """

    id: str  # unique within the fleet; validated at config load
    ratings: AssetRatings  # nameplate; the interval envelope comes from envelope()
    cost_data: AssetCostData  # declared terms, NOT a cost function (D7)
    # What this asset's instruments do to the truth (ADR-0012). Read in
    # observe() only; envelope() and step() must never consult it.
    noise: AssetNoise

    # Which channels of AssetCommand this asset accepts (D3). Validated in both
    # directions at config load.
    p_controllable: bool = True
    q_controllable: bool = False
    on_off_controllable: bool = False

    # How a command outside the kVA circle is resolved (D3).
    kva_priority: KvaPriority = KvaPriority.P

    # D5. Marking an incapable asset is a startup error.
    slack_bearing_capable: bool = False

    # See AssetCapabilities.resource_limited. Makes a P setpoint a CEILING.
    resource_limited: bool = False

    def initial_state(self, carry_in: Mapping[str, float]) -> AssetState:
        """This asset's TRUE state at k=0, validated from its fleet entry's
        carry_in block. A stateless asset refuses a non-empty block; a stateful
        one refuses a missing or alien key. No defaults."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, resolved: ResolvedAsset, dt_minutes: int) -> Self:
        """Build this asset from its validated fleet entry. The ONE construction
        signature, so registry-driven assembly can call it on any `asset_cls`.

        `dt_minutes` is the plant step, passed as the integer rather than the
        whole `SiteConfig` (ADR-0016 D9)."""
        raise NotImplementedError

    def required_scenario_fields(self) -> frozenset[str]:
        """The exogenous columns this asset needs. D1.

        Checked against what the scenario supplies by
        AssembledFleet.validate_scenario_row; a scenario may be a SUPERSET and
        extra columns are ignored. Deterministic, because it feeds a refusal
        message a test asserts on.
        """
        raise NotImplementedError

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        """`ratings` narrowed by true state and this row. Plant-side only."""
        raise NotImplementedError

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[AssetState, AssetDelivered]:
        """Clip the command to the envelope, report the difference, return the
        next state. Never silently do something other than what was commanded
        without recording the delta.

        Also where an asset enforces its own D2 sign rule.
        """
        raise NotImplementedError

    def build_cost(self) -> AssetCost:
        """This asset's own cost model, built from its declared `cost_terms:`.

        An asset that declares no terms returns one whose `lines()` is empty.
        """
        raise NotImplementedError

    def observe(
        self,
        state: AssetState,
        row: ScenarioRow,
        measured_prev: AssetMeasurement | None,
        rng: random.Random,
    ) -> AssetObservation:
        """Degrade the truth into what the dispatcher may see. The ONLY thing
        that crosses to dispatch/, and the one noise point (ADR-0012).

        `measured_prev` is this asset's own outcome for k-1, already projected
        through AssetDelivered.measurement(). None only at k=0.

        `rng` is this asset's OWN stream (ADR-0012 D3), not the fleet's. Use
        `self.metered()` for the two common channels and degrade any additional
        channel of a subclass's observation beside it.
        """
        raise NotImplementedError

    def metered(
        self, measured_prev: AssetMeasurement | None, rng: random.Random
    ) -> AssetMeasurement | None:
        """This asset's last delivered record as its METER reported it. ADR-0012.

        None passes straight through: k=0 is the absence of a previous interval,
        not a reading of zero.
        """
        if measured_prev is None:
            return None
        return AssetMeasurement(
            p_net_kw=add_noise(measured_prev.p_net_kw, self.noise.p_meter_sigma_kw, rng),
            q_net_kvar=add_noise(measured_prev.q_net_kvar, self.noise.q_meter_sigma_kvar, rng),
        )


@dataclass(frozen=True, kw_only=True)
class ResolvedAsset:
    """One fleet asset after its config choices have been checked against its
    type's capability ceiling. The flags here are the SETTLED answer.

    Validated CONFIG, not truth -- the one type in this file that is not.

    Here rather than in sim/fleet.py because every module under sim/assets/ must
    name it in `from_config`, and defining it there closes a cycle:

        fleet -> registry -> assets/pv -> fleet

    sim/fleet.py still owns every rule that BUILDS one. `ResolvedLoad` sits in
    sim/load.py for the same reason.
    """

    id: str
    capabilities: AssetCapabilities  # the type's ceiling
    p_controllable: bool  # this site's choices, already checked against it
    q_controllable: bool
    on_off_controllable: bool
    # How this site resolves a command outside the kVA circle (D3), already
    # checked against `q_controllable`. Defaulted so hand-built records stay
    # expressible; `_resolve_kva_priority` is what a fleet entry goes through.
    kva_priority: KvaPriority = KvaPriority.P
    # The declared terms IN FULL, params included, so an asset class can carry
    # them into its cost_data unchanged. Kinds are checked against the
    # implemented set at build (rejection 4).
    cost_terms: tuple[CostTerm, ...]
    # Declared state at k=0, raw off the fleet entry; validated in
    # initial_state(). Defaulted so hand-built records stay expressible.
    carry_in: Mapping[str, float] = field(default_factory=dict)
    # Validated model parameters, typed by the registry's `spec_cls`. None only
    # for a type whose model takes no parameters (today: grid). Typed as the
    # pydantic base; the asset class narrows it on entry.
    spec: BaseModel | None
    # Validated sensor noise, typed by the registry's `noise_cls` (ADR-0012 D2).
    # Annotated with the concrete base, since every registered noise class
    # derives from AssetNoise. No default.
    noise: AssetNoise

    @property
    def slack_bearing_capable(self) -> bool:
        """Never a per-site choice; read straight from the type's capabilities."""
        return self.capabilities.slack_bearing_capable

    @property
    def resource_limited(self) -> bool:
        """Never a per-site choice either; a site cannot declare its sun
        schedulable."""
        return self.capabilities.resource_limited
