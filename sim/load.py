"""The Load class, and everything about a load that is TRUTH.

ADR-0004 D6. Loads are first-class and symmetric to assets: they declare what
scenario fields they need, produce a demand, and are stepped against what they
were actually served.

A load's demand is the quantity a policy must FORECAST for itself, so none of it
crosses. The dispatcher-visible counterparts:

    LoadState     (here)  ->  nothing.
    LoadDemand    (here)  ->  nothing. Forecasting it is dispatch/'s job.
    LoadDelivered (here)  ->  LoadMeasurement, via measurement()

FixedLoad, at the bottom, is the only load type: its profile() reads
`load_p_kw` straight off the ScenarioRow.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field

from contracts import (
    CostTerm,
    hours_in,
    LoadCommand,
    LoadCostData,
    LoadMeasurement,
    LoadObservation,
)
from sim.noise import add_noise
from sim.scenario import Interval, ScenarioRow

_EPS = 1e-9  # numerical tolerance, same figure sim/plant.py uses; not physics


class LoadNoise(BaseModel):
    """The meter on a load feeder: the three channels of `LoadMeasurement`.
    ADR-0012 D2.

    `unserved_p_sigma_kw` is its own field: unserved power is a DIFFERENCE and
    inherits both errors, so it is generally the worse number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    served_p_sigma_kw: float = Field(ge=0.0)
    served_q_sigma_kvar: float = Field(ge=0.0)
    unserved_p_sigma_kw: float = Field(ge=0.0)


class LoadCost(Protocol):
    """What a load's own model charges. Symmetric to `AssetCost`."""

    def update(self, delivered: LoadDelivered, iv: Interval) -> None:
        """Price one interval and add it to this member's running totals."""
        ...

    def close_period(self) -> None:
        """Settle anything path-dependent. Called by the ledger on the calendar."""
        ...

    def lines(self) -> Mapping[str, float]:
        """Cost-term kind -> rupees, deterministically ordered."""
        ...


@dataclass(frozen=True, kw_only=True)
class LoadState:
    """One load's TRUE mutable state. ADR-0004 D6.

    Empty: no load modelled today carries state. The type exists because Load is
    symmetric to Asset.
    """


@dataclass(frozen=True, kw_only=True)
class LoadDemand:
    """What a load WANTS this interval. ADR-0004 D6.

    Positive = drawing, the OPPOSITE polarity to AssetDelivered.p_net_kw, because
    the D2 balance puts loads on the right:

        sum(P_net over assets) + unserved == sum(load served)

    `q_kvar` is pinned to zero until reactive demand is modelled.
    """

    p_kw: float
    q_kvar: float = 0.0

    def __post_init__(self) -> None:
        if self.p_kw < 0.0:
            raise ValueError(
                f"load demand p_kw={self.p_kw} is negative. A load draws; something that "
                f"injects is an asset. Check the sign convention (D6: + = drawing)."
            )


@dataclass(frozen=True, kw_only=True)
class LoadDelivered:
    """The FULL truth of what one load got. ADR-0004 D6a.

    Two separate unserved fields, priced identically but meaning opposite things:

      `shed_p_kw`     VOLUNTARY -- the dispatcher commanded below demand.
      `unserved_p_kw` INVOLUNTARY -- generation could not meet load and the plant
                      shed by priority order (D5 secondary slack).

    Involuntary shedding stands in for a bus trip and is therefore OPTIMISTIC:
    it prices the missed energy, not the outage.

    `measurement()` is the projection that may cross to dispatch/.
    """

    served_p_kw: float
    served_q_kvar: float = 0.0
    shed_p_kw: float = 0.0
    unserved_p_kw: float = 0.0
    violations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("served_p_kw", "shed_p_kw", "unserved_p_kw"):
            value: float = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name}={value} is negative; load quantities are magnitudes")

    def measurement(self) -> LoadMeasurement:
        """The dispatcher-visible projection. The ONLY sanctioned way to get one."""
        return LoadMeasurement(
            served_p_kw=self.served_p_kw,
            served_q_kvar=self.served_q_kvar,
            unserved_p_kw=self.unserved_p_kw,
        )


@dataclass(frozen=True, kw_only=True)
class Load:
    """One load. Symmetric to Asset, with two parameters that must not be
    collapsed into one (D6):

      `priority`   an ORDERING -- LOWER number = more critical = shed LAST.
                   Asserted at config load, never inferred.
      `cost_data`  a PRICE -- what any unserved kWh costs.

    Priorities must be unique across the fleet, or ties broken deterministically,
    or the shed is nondeterministic and reproducibility is gone (D6a, D10).
    """

    id: str  # unique within the fleet; validated at config load
    priority: int
    cost_data: LoadCostData
    # This feeder's meter (ADR-0012). Read only in observe(), never in profile()
    # or step().
    noise: LoadNoise

    # Which channel of LoadCommand this load accepts. Validated in both
    # directions like an asset's.
    #
    # Defaults TRUE, where every asset capability defaults False: D6a's
    # involuntary path already assumes the plant can drop a load by priority, so
    # defaulting False would make the voluntary half unreachable. A genuinely
    # unsheddable load sets it False, and then only physics can drop it.
    on_off_controllable: bool = True

    @classmethod
    def from_config(cls, resolved: ResolvedLoad, dt_minutes: int) -> Self:
        """Build this load from its validated fleet entry. Same one-signature
        contract as Asset.from_config, for the same registry-driven assembly."""
        raise NotImplementedError

    def build_cost(self) -> LoadCost:
        """This load's own cost model, built from its declared terms.

        Every load type implements this: there is no ledger-side term library
        left to fall back to.
        """
        raise NotImplementedError

    def initial_state(self, carry_in: Mapping[str, float]) -> LoadState:
        """True state at k=0, validated. Same contract as Asset.initial_state:
        a stateless load refuses a non-empty block."""
        raise NotImplementedError

    def required_scenario_fields(self) -> frozenset[str]:
        """The exogenous columns this load needs. Same contract as Asset's."""
        raise NotImplementedError

    def profile(self, state: LoadState, row: ScenarioRow) -> LoadDemand:
        """True demand this interval. Plant-side only -- the dispatcher sees
        the degraded forecast that observe() builds from this."""
        raise NotImplementedError

    def step(
        self, state: LoadState, cmd: LoadCommand, row: ScenarioRow, served_p_kw: float
    ) -> tuple[LoadState, LoadDelivered]:
        """Record what was actually served against what was demanded.

        `cmd.on` is the dispatcher's CONTACTOR -- switched off is a voluntary
        shed of the load's whole demand; `served_p_kw` is what the plant could
        actually deliver after the balance closed. The difference between them
        is exactly the voluntary/involuntary split (D6a): a shed the dispatcher
        chose and paid for, versus physics leaving no choice. Both are priced by
        the same shed cost, and they are kept apart because they say opposite
        things about the dispatcher.

        The plant NEVER sheds voluntarily to ease its own balancing (D6a). Do
        not add plant-side "helpful" shedding here; it rescues a bad dispatch
        for free.
        """
        raise NotImplementedError

    def observe(
        self,
        state: LoadState,
        row: ScenarioRow,
        measured_prev: LoadMeasurement | None,
        rng: random.Random,
    ) -> LoadObservation:
        """What the dispatcher may see of this load. The only thing that
        crosses to dispatch/.

        `measured_prev` is this load's served outcome for k-1, already projected
        through LoadDelivered.measurement(). None only at k=0. `rng` is this
        load's OWN stream (ADR-0012 D3)."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class ResolvedLoad:
    """One fleet load after validation, parallel to ResolvedAsset. Validated
    config, not truth.

    Carries the type NAME rather than a registry type: holding one here would
    close the cycle fleet -> registry -> load -> fleet. The registry resolves
    name -> load_cls at construction.
    """

    id: str
    type_name: str
    priority: int
    cost_terms: tuple[CostTerm, ...]  # in full, params included; see ResolvedAsset
    # The type's declared contactor channel, copied off the registry at build so
    # Fleet.validate_commands() can check it without constructing the load.
    on_off_controllable: bool = True
    # Declared state at k=0, raw; validated in initial_state(). Empty for every
    # load type that exists today.
    carry_in: Mapping[str, float] = field(default_factory=dict)
    # This feeder's validated meter noise (ADR-0012 D2). No default.
    noise: LoadNoise


@dataclass(frozen=True, kw_only=True)
class FixedLoad(Load):
    """The one load type today: demand IS ScenarioRow.load_p_kw, no state. The
    D6a bookkeeping in step() splits who chose not to serve from who could not.
    """

    @classmethod
    def from_config(cls, resolved: ResolvedLoad, dt_minutes: int) -> FixedLoad:
        return cls(
            id=resolved.id,
            priority=resolved.priority,
            on_off_controllable=resolved.on_off_controllable,
            # PER LOAD, off this load's own `unserved` term (D6).
            cost_data=LoadCostData(terms=resolved.cost_terms),
            noise=resolved.noise,
        )

    def build_cost(self) -> LoadCost:
        return UnservedCost.build(member_id=self.id, declared=self.cost_data.terms)

    def required_scenario_fields(self) -> frozenset[str]:
        return frozenset({"load_p_kw"})

    def initial_state(self, carry_in: Mapping[str, float]) -> LoadState:
        if carry_in:
            raise ValueError(
                f"load {self.id!r}: a fixed load carries no state, but carry_in declares "
                f"{sorted(carry_in)}. Refused rather than ignored (decision 15)."
            )
        return LoadState()

    def profile(self, state: LoadState, row: ScenarioRow) -> LoadDemand:
        self._narrow(state)
        return LoadDemand(p_kw=row.load_p_kw)  # reactive demand pinned to zero (D6)

    def step(
        self, state: LoadState, cmd: LoadCommand, row: ScenarioRow, served_p_kw: float
    ) -> tuple[LoadState, LoadDelivered]:
        load_state = self._narrow(state)
        demand_kw = row.load_p_kw
        violations: list[str] = []

        # The contactor. Closed, the load draws what it draws; open, it draws
        # nothing and the dispatcher pays the whole demand as a voluntary shed.
        # An uncommanded channel (None) is a load with no contactor: it stays
        # connected, and only physics can drop it.
        chosen_kw = 0.0 if cmd.on is False else demand_kw

        # No negative-served or over-served check: a load is SWITCHED, not
        # dialled (D6a), so a boolean makes those unrepresentable.

        # The plant can serve less than the dispatcher allowed (D5 secondary
        # slack) but never more: more would mean the plant invented a shed
        # reversal it was never commanded. Internal contract, so an assert.
        assert served_p_kw <= chosen_kw + _EPS, (
            f"load {self.id}: plant served {served_p_kw} kW above the allowed " f"{chosen_kw} kW"
        )

        # The D6a split. Voluntary is the dispatcher's economic choice,
        # involuntary is physics leaving no choice; priced identically,
        # recorded separately, because they say opposite things about the
        # dispatcher.
        shed_kw = demand_kw - chosen_kw
        unserved_kw = max(0.0, chosen_kw - served_p_kw)

        return load_state, LoadDelivered(
            served_p_kw=served_p_kw,
            served_q_kvar=0.0,
            shed_p_kw=shed_kw,
            unserved_p_kw=unserved_kw,
            violations=tuple(violations),
        )

    def observe(
        self,
        state: LoadState,
        row: ScenarioRow,
        measured_prev: LoadMeasurement | None,
        rng: random.Random,
    ) -> LoadObservation:
        """This feeder's meter, applied to its own last served record. ADR-0012.

        `LoadObservation` carries NO forecast field; building one is dispatch/'s
        job. `unserved_p_kw` is not clamped at zero -- clamping would bias the
        one channel a policy watches to learn whether it is shedding.
        """
        self._narrow(state)
        return LoadObservation(
            priority=self.priority,
            measured_prev=self._metered(measured_prev, rng),
        )

    def _metered(
        self, measured_prev: LoadMeasurement | None, rng: random.Random
    ) -> LoadMeasurement | None:
        """The load-side twin of `Asset.metered`. None passes through: at k=0
        there is no previous interval, which is not a reading of zero."""
        if measured_prev is None:
            return None
        return LoadMeasurement(
            served_p_kw=add_noise(measured_prev.served_p_kw, self.noise.served_p_sigma_kw, rng),
            served_q_kvar=add_noise(
                measured_prev.served_q_kvar, self.noise.served_q_sigma_kvar, rng
            ),
            unserved_p_kw=add_noise(
                measured_prev.unserved_p_kw, self.noise.unserved_p_sigma_kw, rng
            ),
        )

    def _narrow(self, state: LoadState) -> LoadState:
        if not isinstance(state, LoadState):
            raise TypeError(f"load {self.id!r} expects LoadState, got {type(state).__name__}")
        return state


class UnservedParams(BaseModel):
    """This load's shed price. ADR-0004 D6: one price, both shedding paths.
    Per LOAD, not per site, so it can disagree with `priority`."""

    model_config = ConfigDict(frozen=True)

    shed_cost_inr_per_kwh: float = Field(ge=0)


# ---------------------------------------------------------------------------
# Asset terms
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class UnservedCost:
    """Energy this load asked for and did not get, BY EITHER PATH. ADR-0004 D6.

    One price on both halves of D6a's split. They cannot double-count:
    `FixedLoad.step()` books `shed = demand - chosen` and
    `unserved = max(0, chosen - served)`, partitioning `demand - served`.
    """

    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {"unserved": UnservedParams}
    KINDS: ClassVar[frozenset[str]] = frozenset(KIND_PARAMS)

    declared: tuple[str, ...]
    params: UnservedParams
    _total_inr: float = 0.0

    @classmethod
    def build(cls, *, member_id: str, declared: tuple[CostTerm, ...]) -> UnservedCost:
        kinds = [term.kind for term in declared]
        unknown = [k for k in kinds if k not in cls.KINDS]
        if unknown:
            raise ValueError(
                f"load {member_id!r}: prices {sorted(cls.KINDS)}, but the fleet declares "
                f"{sorted(set(unknown))}. A term nobody implements must be refused, not "
                f"billed as zero."
            )
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"load {member_id!r}: duplicate cost term kinds {sorted(kinds)}")

        term = next((t for t in declared if t.kind == "unserved"), None)
        if term is None:
            # REFUSED, never defaulted to zero. A load with no shed price is a
            # load that is free to drop, which makes involuntary shedding the
            # cheapest way to balance and never says so (ADR-0004 D6).
            raise ValueError(
                f"load {member_id!r} declares no `unserved` cost term, so it has no shed "
                f"price. A load that is free to drop makes involuntary shedding the "
                f"cheapest balance."
            )
        params = UnservedParams.model_validate(dict(term.params))
        return cls(declared=("unserved",), params=params)

    def update(self, delivered: LoadDelivered, iv: Interval) -> None:
        dropped_kwh = (delivered.shed_p_kw + delivered.unserved_p_kw) * hours_in(iv.dt_minutes)
        self._total_inr += dropped_kwh * self.params.shed_cost_inr_per_kwh

    def close_period(self) -> None:
        """Nothing settles at a billing boundary: shedding is priced per interval."""

    def lines(self) -> Mapping[str, float]:
        return {"unserved": self._total_inr} if self.declared else {}
