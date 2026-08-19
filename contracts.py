"""Shared vocabulary between the testbed and the dispatcher.

This module imports nothing from sim/ or dispatch/. It is the ONLY thing both
sides are allowed to import. See CLAUDE.md and docs/adr/0002.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

# ---------------------------------------------------------------------------
# The clock.
#
# A run is described by three primitives and nothing else: a tz-aware ISO
# `timestamp` on every row and observation, `dt_minutes` (the plant step) and
# `dispatch_interval_multiple` (the control rate, in plant steps). Every other
# time-shaped quantity -- an hour, a billing month, a slot of the day, the
# number of intervals in a day, the hours a kW is delivered for -- is DERIVED,
# here, at the point of use.
#
# They live in contracts.py because it is the only module `dispatch/` may
# import, so it is the one place both sides can share a definition of the clock.
# Nothing here is truth: a calendar is a fact about the run, not about the
# plant, and reconstructs nothing a policy is meant to be uncertain about.
#
# `k` is a ROW COUNT. It is not a clock, and nothing here takes one.
# ---------------------------------------------------------------------------

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR


def parse_timestamp(timestamp: str) -> datetime:
    """The one parse. Every consumer of a timestamp comes through here."""
    return datetime.fromisoformat(timestamp)


def minutes_of_day(timestamp: str) -> int:
    """Minutes since local midnight.

    The general form of what `dispatch/forecast.py` does for its slots and what
    the tariff needs for its bands. From the CLOCK, never from a running count:
    a run starting at 06:00 lands in the same band as one starting at midnight,
    and neither has to say so.
    """
    ts = parse_timestamp(timestamp)
    return ts.hour * MINUTES_PER_HOUR + ts.minute


def billing_month(timestamp: str) -> tuple[int, int]:
    """The `(year, month)` a timestamp bills into.

    From the calendar, never from a counted number of intervals: months are not
    the same length, so a charge resetting every N intervals drifts off the
    billing calendar and never says so.
    """
    ts = parse_timestamp(timestamp)
    return (ts.year, ts.month)


def intervals_per_day(dt_minutes: int) -> int:
    """How many steps of `dt_minutes` a day holds.

    Refuses a step that does not divide the day. Slot-of-day indexing, a ToU
    band and a demand-charge block all need whole intervals, and a step that
    leaves a remainder makes each of them silently wrong at a different point.
    """
    if dt_minutes <= 0:
        raise ValueError(f"dt_minutes={dt_minutes} is not positive")
    if MINUTES_PER_DAY % dt_minutes != 0:
        raise ValueError(
            f"dt_minutes={dt_minutes} does not divide the day into whole intervals "
            f"({MINUTES_PER_DAY / dt_minutes} of them). Slot, band and block indexing all "
            f"need it to."
        )
    return MINUTES_PER_DAY // dt_minutes


def hours_in(dt_minutes: int) -> float:
    """One interval as a fraction of an hour -- the kW -> kWh factor.

    A CONVERSION computed where it is used, never a stored second
    representation of the plant step. The whole point of the three primitives
    is that `dt_minutes` is the only declaration of how long an interval is.

    Callers must apply it as `p_kw * hours_in(dt)`, computing the factor first:
    `p_kw * dt / 60` is not the same float. At dt = 15 the factor is 0.25, an
    exact power of two, and scaling by it is exact where the two-step form is
    not.
    """
    return dt_minutes / MINUTES_PER_HOUR


def clock_to_minutes(clock: str) -> int:
    """`"HH:MM"` -> minutes since midnight, for a declared band boundary.

    A tariff document states clock times, so a config that declares them reads
    like the document it came from. `"24:00"` is accepted and is the only way to
    write the end of the last band of the day -- `datetime.time` cannot hold it,
    which is why this is a parse of its own rather than one.
    """
    try:
        hours, _, minutes = clock.partition(":")
        total = int(hours) * MINUTES_PER_HOUR + int(minutes)
    except ValueError:
        raise ValueError(f"clock time {clock!r} is not HH:MM") from None
    if not clock.count(":") == 1 or not 0 <= total <= MINUTES_PER_DAY:
        raise ValueError(f"clock time {clock!r} is not a HH:MM in [00:00, 24:00]")
    return total


def step_local(timestamp: str, intervals: int, dt_minutes: int) -> datetime:
    """`intervals` steps on from `timestamp`, stepped in UTC.

    Stepping a LOCAL tz-aware datetime by a timedelta adds wall-clock time and
    silently skips or repeats an hour across a DST boundary; stepping in UTC and
    converting back is the only form that survives one. `synth/generate.py` has
    always done it this way and everything else has not.

    Asia/Kolkata has no DST, so site_a would never have noticed -- which is
    precisely the kind of assumption that survives until it is somebody else's
    problem.
    """
    ts = parse_timestamp(timestamp)
    moved = ts.astimezone(UTC) + intervals * timedelta(minutes=dt_minutes)
    return moved.astimezone(ts.tzinfo)


class Quality(StrEnum):
    """Provenance of one ScenarioRow's exogenous values. ADR-0003 D1 item 4.

    A run built from imputed data is not the same evidence as a run built from
    meter readings, so the difference has to survive into the log rather than
    being absorbed at load time. There is deliberately NO default: every row
    states where its numbers came from, because a defaulted MEASURED is exactly
    how synthetic data starts getting quoted as measurement.

    Only values with a real producer are listed. Add one when something
    actually emits it, not before.
    """

    MEASURED = "measured"  # straight from the source, unmodified
    IMPUTED = "imputed"  # carried forward from the previous interval
    SYNTHETIC = "synthetic"  # generated; never measured anywhere


@dataclass(frozen=True, kw_only=True)
class Observation:
    """What the dispatcher is allowed to see: one degraded view PER FLEET
    MEMBER, keyed by id. ADR-0004 D9, live since Stage E.

    The keys are the fleet, fixed at config load, in no promised order --
    iterate the FleetView (declaration order, D10) and index into these. The
    per-member values are the typed AssetObservation/LoadObservation
    subclasses; what each may carry is decided there.

    The pre-Stage-E `delivered_prev` leak is CLOSED: what crosses about the
    previous interval is each member's `measured_prev` (an AssetMeasurement /
    LoadMeasurement -- the projection), never a full delivered record. The
    forecast-uncertainty reasoning is at AssetMeasurement.
    """

    k: int
    # WHAT TIME IT IS, tz-aware local, ISO-8601 -- the same string the scenario
    # row carries and the ledger bills against.
    #
    # A MEASUREMENT of the calendar, not of the plant, which is why it may cross
    # (D9's membership rule). It reconstructs nothing: a policy that knows the
    # hour learns nothing about load, irradiance or SOC it did not already have.
    #
    # Withholding it was an accident with teeth. D7 hands the dispatcher the
    # TARIFF on AssetView.cost_terms so its own cost model can price against the
    # same declared data as the ledger -- and that tariff is keyed by clock hour
    # (`start_hour`/`end_hour`). Without this field the dispatcher held a
    # time-of-use schedule and no way to know the time, so the independent 5.2
    # cost model D7 requires was not expressible. A real EMS reads a clock.
    #
    # A STRING, matching ScenarioRow.timestamp, so nothing here has to agree with
    # sim/ about a datetime representation; a policy parses it as it likes.
    timestamp: str
    assets: Mapping[str, AssetObservation]
    loads: Mapping[str, LoadObservation]


@dataclass(frozen=True, kw_only=True)
class Commands:
    """What the dispatcher requests: one AssetCommand per asset, one
    LoadCommand per load, keyed by id. ADR-0004 D9, live since Stage E.

    EVERY fleet member must be commanded on EVERY channel it declares --
    Fleet.validate_commands() runs in the loop, and an absent command is an
    error, never a default (a defaulted zero to PV silently means "curtail
    everything" and still balances, D4). The slack-bearing asset's P setpoint
    is advisory (D5): the plant delivers the balancing value.
    """

    assets: Mapping[str, AssetCommand]
    loads: Mapping[str, LoadCommand]


# `Delivered` and `StepLog` USED TO LIVE HERE, as the flat one-of-each-type
# shapes the run log and the ledger spoke through Stage E (implementation-plan
# decisions 9 and 20). Both are gone at Stage F, and with them the restriction
# they carried: the plant needed exactly one asset of each type to build a
# legacy row, so a two-battery fleet could be VALIDATED but never RUN. The keyed
# log in sim/plant.py replaces them and lifts that.
#
# They are not replaced HERE, and that is the point of the move rather than a
# side effect. A step log carries realized truth -- a battery's true SOC, the
# violation strings with their magnitudes -- so under the D9 rule it belongs
# where the import graph keeps it away from dispatch/. What crosses to a
# dispatcher about a past interval is `measured_prev` on the observations
# above, and nothing else.


@dataclass(frozen=True, kw_only=True)
class CostBreakdown:
    """The final tally: one line per (fleet member, cost term). ADR-0004 D8.

    Re-keyed at Stage F from the seven fixed fields
    (`energy_inr`/`demand_charge_inr`/...) that hardcoded one fleet into the
    accounting, which is the same weld ADR-0004 removed from the contracts and
    the physics. Keys are `"<member_id>.<term_kind>"` -- `grid_1.energy`,
    `dg_1.fuel`, `load_1.unserved` -- so two batteries produce two degradation
    lines instead of silently sharing one.

    ORDER IS DECLARATION ORDER (D10), and is not cosmetic: `total_inr` adds
    these in iteration order and floating-point addition is not associative.
    The ledger builds the mapping by iterating the fleet, never a set.

    Every line is rounded to the paisa by the ledger, and the total adds the
    ROUNDED lines, so the printed items always add up to the printed total.

    Never report a bare scalar -- always the items (CLAUDE.md). `total_inr` is
    a convenience for a headline figure, not the output.

    D8 also gives the system ledger genuinely cross-asset lines -- a renewable
    mandate is a ratio over the whole fleet and cannot attach to one member.
    None are implemented, so every key here is a member line today. They would
    be keyed `"system.<term>"`, which is why member ids may not be `system`.
    """

    lines: Mapping[str, float]

    @property
    def total_inr(self) -> float:
        return sum(self.lines.values())


# ---------------------------------------------------------------------------
# ADR-0004 -- the configurable fleet. The SHARED half.
#
# Everything below this line landed ADDITIVELY at Stage B, unwired, so that the
# asset ports (Stage D) and the plant rewrite (Stage E) were mechanical rather
# than one large simultaneous diff. It is all WIRED now: `Commands` and
# `Observation` above are keyed collections of these per-member types (Stage E),
# and the cost vocabulary below drives the ledger's term library (Stage F).
#
# WHAT IS HERE AND WHAT IS NOT. This module is the only one dispatch/ may
# import, so membership of it is a permission, not a filing decision. Anything
# that is or reveals TRUTH lives in sim/ instead, where the import graph makes
# it unreachable from dispatch/ -- not a lint rule, a fact. In sim/asset.py:
# AssetState and its subclasses, AssetEnvelope, AssetDelivered, KvaPriority,
# Asset. In sim/load.py: LoadState, LoadDemand, LoadDelivered, Load. In
# sim/plant.py: PlantState.
#
# The line between the two halves is drawn three times over, and each cut is a
# different kind of secret:
#
#   AssetRatings vs AssetEnvelope   Nameplate limits are on a datasheet and
#                                   leak nothing. The same limits narrowed by
#                                   true state do: a BESS's p_max_kw inverts to
#                                   its exact SOC. Two types, one visible.
#   AssetMeasurement vs AssetDelivered
#                                   What an asset was seen to do, versus the
#                                   full truth of what it did. `curtailed_kw`
#                                   is the reason: curtailed + delivered IS the
#                                   true available irradiance, which is the
#                                   very thing pv_forecast_kw is supposed to be
#                                   a degraded view of. Publishing both would
#                                   hand the dispatcher perfect hindsight and
#                                   make the forecast-error model decorative.
#   LoadMeasurement vs LoadDelivered
#                                   Same cut, drawn by a different test: could
#                                   a real site METER it? Served and unserved
#                                   power, yes. The voluntary/involuntary split
#                                   and the violation strings, no.
#
# The design test these types must pass: every field of the hardcoded contracts
# above has a home, or the fleet is still welded in somewhere.
#
#     Commands.bess_kw / .genset_kw   -> AssetCommand.p_setpoint_kw
#     Commands.pv_curtail_kw          -> AssetCommand.p_setpoint_kw  (D4: there is
#                                        no curtail channel; see the warning there)
#     Delivered.{bess,genset,grid,pv}_kw -> AssetDelivered.p_net_kw   (sim/asset.py)
#     Delivered.curtailed_kw          -> AssetDelivered.curtailed_kw  (sim/asset.py)
#     Delivered.load_kw               -> LoadDelivered.served_p_kw    (sim/load.py)
#     Delivered.unserved_kw           -> LoadDelivered.unserved_p_kw  (sim/load.py;
#                                        now per-load and per-path, D6a)
#     CostBreakdown.energy_inr, ...   -> CostBreakdown.lines["grid_1.energy"], ...
#                                        (D8, Stage F: keyed by member and term)
#     PlantState.soc_kwh/.soh/.cumulative_throughput_kwh -> BessState  (sim/asset.py)
#     PlantState.genset_on/.genset_run_hours            -> GensetState (sim/asset.py)
#     Observation.soc_observed_kwh    -> BessObservation
#     Observation.pv_forecast_kw      -> PvObservation
#     Observation.load_forecast_kw    -> LoadObservation
#     Observation.*_available         -> AssetObservation.available   (D11)
#     Observation.delivered_prev      -> Asset/LoadObservation.measured_prev,
#                                        narrowed: see the three cuts above
#     Observation.k                   -> stays on the keyed container (D9, Stage E)
#
# The delivered_prev KNOWN HOLE that used to be documented here CLOSED at
# Stage E: Observation carries per-member measured_prev projections and nothing
# else about the previous interval. The `Delivered`/`StepLog` legacy log shapes
# that outlived it were deleted at Stage F, so the mapping above is now the
# whole story rather than a plan.
#
# These are kw_only so that a subclass may add a required field after a base
# class's defaulted one. Without it the per-type subclasses of D1 are simply
# not expressible, and the workaround -- defaulting fields that have no sane
# default, like a BESS's observed SOC -- is exactly the silent-default trap the
# rest of this module is written to avoid.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class AssetCommand:
    """Everything the dispatcher may ask of one asset: a P setpoint, a Q
    setpoint, and an on/off. ADR-0004 D4.

    There is deliberately NO curtailment or shed channel. Commanding a PV asset
    below its available power IS the curtailment, and the asset reports the
    difference as a measurement.

    THE TRAP, and it is silent: today `pv_curtail_kw = 0.0` means PV runs flat
    out, whereas here `p_setpoint_kw = 0.0` means curtail EVERYTHING. Both
    close the energy balance, so nothing fires -- PV output simply vanishes.
    Whoever ports the PV dispatcher path reads this line first.

    ALL THREE channels are None when NOT COMMANDED, which is distinct from a
    commanded zero or a commanded False. That distinction is what makes the D3
    validation possible, and every channel is checked in BOTH directions: one
    commanded on an asset that does not declare it, and one left absent on an
    asset that does, are both errors. Fleet.validate_commands() enforces it.

      `p_setpoint_kw`    Optional so a REACTIVE-ONLY asset is expressible. A
                         STATCOM or synchronous condenser is q_controllable and
                         not p_controllable, and would otherwise be forced to
                         carry a P setpoint it cannot honour.

                         NOTE THE TRADE, because it is a real loss. This was a
                         mandatory field, so omitting it was a TypeError at
                         construction and could not reach the plant. The
                         guarantee now lives in the validator instead of the
                         type system, so it holds only where validate_commands()
                         is actually called. What it buys is an asset class the
                         fleet previously could not describe at all. Stage E is
                         what paid it back: runner.simulate() calls
                         Fleet.validate_commands() on EVERY interval, so the
                         check runs wherever a run does.

      `q_setpoint_kvar`  Pinned to zero in config for now, so the ADR-0004
                         refactor stays bit-identical against the goldens.

      `on`               An explicit commit channel. This EXTENDS D4, which
                         named setpoints only. It exists because "genset off"
                         and "genset commanded 0 kW" are different intentions
                         that today's genset.deliver() conflates by shutting
                         off below minimum stable load -- so a dispatcher
                         cannot currently ask for a spinning idle, and cannot
                         be held responsible for a start it did not order.
    """

    p_setpoint_kw: float | None = None
    q_setpoint_kvar: float | None = None
    on: bool | None = None


@dataclass(frozen=True, kw_only=True)
class AssetCapabilities:
    """What an asset TYPE can do. ADR-0004 D3/D5.

    A statement about EQUIPMENT, not about one site's choices: a PV array can
    never hold a bus, a rotating genset has no commandable reactive channel.
    Config chooses within this ceiling and may never exceed it.

    In contracts.py, and deliberately so. Apply the D9 rule -- move it to sim/
    only if knowing it would defeat uncertainty the experiment depends on -- and
    capability fails that test completely: it is a datasheet fact. A dispatcher
    that cannot read `q_controllable` cannot avoid issuing the Q command D3 calls
    a construction error, so withholding it would be withholding the one thing
    that makes the rule followable.

    Defaults are the conservative end of every axis: a type grants nothing it
    has not asked for, so a forgotten declaration surfaces as a config rejection
    rather than as a command the physics cannot honour.

    TERMINOLOGY: ADR-0004 D5 calls this "grid-forming". The code says
    SLACK-BEARING, which is what the testbed actually models: the asset whose
    real power is determined by the balance rather than by its command
    (`P = sum(load served) - sum(P_net over the others)`). That is the slack bus
    of a load flow. Grid-forming is a narrower control-mode claim -- a
    voltage-source converter holding frequency and voltage -- and an asset can
    be one without being the other. At 15-minute energy bookkeeping we never
    model the control mode, only who absorbs the imbalance, so naming the flag
    after the control mode would promise physics this testbed does not have.
    """

    name: str
    p_controllable: bool = True
    q_controllable: bool = False
    on_off_controllable: bool = False
    # D5's "grid-forming" capability, under the more accurate name. See above.
    slack_bearing_capable: bool = False
    # Does this asset's output follow an exogenous resource it cannot store?
    #
    # The distinction a policy actually needs, and the one no other flag here
    # carries: does commanding ZERO mean "do nothing", or "actively suppress
    # something that would happen anyway"? Sun arrives whether it is dispatched
    # or not, so zero to PV is curtail-everything; a genset produces nothing
    # until told to, so zero to it is idle. Those are opposite instructions, and
    # a policy that cannot tell them apart falls into the D4 trap silently --
    # PV output vanishes and the balance still closes.
    #
    # It also decides what a P SETPOINT MEANS for this asset. On a
    # resource-limited injector the setpoint is a CEILING, not a target: above
    # available is "run free" and no error, below available is curtailment. That
    # is how an inverter is actually operated -- you set a limit, not a demand.
    resource_limited: bool = False


@dataclass(frozen=True, kw_only=True)
class AssetRatings:
    """One asset's NAMEPLATE limits, in the D2 sign convention. ADR-0004 D3.

    Static: the same every interval of the run, because it comes off a
    datasheet. That is exactly why it is safe to publish to the dispatcher, and
    why it is a different type from the interval envelope in sim/asset.py.
    Confusing the two is how the SOC leaks, so they cannot be confused: this
    one is here, that one is somewhere dispatch/ cannot import from.

    `s_max_kva` is None for an asset with no apparent-power rating (a rotating
    genset, today's unlimited grid slack); such an asset is not kVA-clipped.
    How an asset resolves a command outside its kVA circle is `kva_priority` on
    the asset itself.

    `energy_capacity_kwh` is None for everything that is not a store. It landed
    with ADR-0006's rule-based dispatcher, which needs it for a reason worth
    stating: `BessObservation.soc_observed_kwh` is ABSOLUTE, so without a
    capacity to divide by, no policy can express a threshold as a state of
    charge -- and "keep 20% in reserve" is how every operator, datasheet and
    control narrative in this domain is written. A policy could only have used
    absolute kWh, which is the same number welded to one battery's size.

    It is a NAMEPLATE and passes the D9 test cleanly: a battery's rated energy is
    printed on it, the same way its power limits are. What stays behind is the
    interval envelope in sim/asset.py, whose p bounds invert to the exact SOC --
    capacity says how big the tank is, never how full it is.
    """

    p_min_kw: float  # most negative injection the asset is rated for
    p_max_kw: float  # largest injection the asset is rated for
    q_min_kvar: float = 0.0
    q_max_kvar: float = 0.0
    s_max_kva: float | None = None
    energy_capacity_kwh: float | None = None

    def __post_init__(self) -> None:
        if self.p_min_kw > self.p_max_kw:
            raise ValueError(
                f"ratings inverted: p_min_kw={self.p_min_kw} > p_max_kw={self.p_max_kw}"
            )
        if self.q_min_kvar > self.q_max_kvar:
            raise ValueError(
                f"ratings inverted: q_min_kvar={self.q_min_kvar} > q_max_kvar={self.q_max_kvar}"
            )
        if self.s_max_kva is not None and self.s_max_kva <= 0.0:
            raise ValueError(f"s_max_kva must be positive when set, got {self.s_max_kva}")
        if self.energy_capacity_kwh is not None and self.energy_capacity_kwh <= 0.0:
            raise ValueError(
                f"energy_capacity_kwh must be positive when set, got {self.energy_capacity_kwh}. "
                f"A store with no capacity is not a store; leave it None for an asset that "
                f"holds no energy."
            )


@dataclass(frozen=True, kw_only=True)
class AssetMeasurement:
    """What one asset was OBSERVED to do, in the D2 sign convention. D1/D4.

        p_net_kw   > 0 injecting into the bus,  < 0 drawing from it
        q_net_kvar > 0 supplying reactive,      < 0 absorbing it

    The dispatcher-visible projection of AssetDelivered (sim/asset.py), and
    deliberately a strict subset of it. What is held back is `curtailed_kw`:
    curtailed + delivered reconstructs the true available power exactly, and
    for PV that is the true irradiance -- the quantity `pv_forecast_kw` exists
    to give the dispatcher a DEGRADED view of. Publish both and the dispatcher
    gets perfect hindsight on the resource one interval late, and the forecast
    error model stops meaning anything.

    `violations` is held back for the same reason: the strings carry magnitudes
    (`island_pv_curtailed:45.0`). The dispatcher still learns it was clipped,
    the way D4 says it should -- by comparing its own setpoint to p_net_kw.
    That delta is the intended channel; it just does not come with the answer.
    """

    p_net_kw: float
    q_net_kvar: float = 0.0


@dataclass(frozen=True, kw_only=True)
class AssetObservation:
    """The degraded per-asset view the dispatcher may see. ADR-0004 D1.

    `ratings` is the nameplate. It cannot accidentally be the interval envelope
    -- that type lives in sim/asset.py, which this module cannot import.

    `available` mirrors the exogenous per-asset flag (D11) and is safe to
    expose: a policy has to know an asset is down so it does not command it.

    Controllability (D3) is deliberately NOT here. Which channels an asset
    accepts is a static property, so it lives on the asset and is validated
    once at config load rather than being restated 5787 times per run. How the
    dispatcher learns it is Stage H's problem, when the dispatcher stops
    hardcoding asset names.

    `measured_prev` is truth for interval k-1, never k, so it is causally
    available before the dispatcher commands k. It is noiseless and ideal --
    the same deliberate exception the system-level Observation already makes --
    but it is the MEASUREMENT, not the full delivered record. See
    AssetMeasurement for what is held back and why.
    """

    available: bool
    ratings: AssetRatings
    measured_prev: AssetMeasurement | None  # None only at k=0


@dataclass(frozen=True, kw_only=True)
class BessObservation(AssetObservation):
    """`soc_observed_kwh` is the DEGRADED view, never BessState.soc_kwh. The
    two are equal today only because observe() is still noiseless."""

    soc_observed_kwh: float


# One parameter of one declared cost term.
#
# WIDENED AT STAGE F, as the Stage C shape reserved. The flat alias could not
# express the tariff's structured parameters -- a ToU window is a label, two
# hour bounds and a multiplier; a PF penalty band is two bounds and a
# percentage -- and D7 requires the grid asset to declare its ENTIRE tariff as
# cost data. So the alias is now recursive: a param is a scalar, a sequence of
# params, or a mapping of params. That is the shape YAML produces and the shape
# both cost libraries (ledger, and dispatch/'s 5.2) read.
#
# Deliberately still a VALUE type, not a schema. contracts.py grants dispatch/
# the right to NAME a parameter, and nothing more: what a given `kind`'s params
# must contain is the business of whichever library is pricing them, and each
# validates for itself. Putting the tariff's schema here would make one side's
# reading of it authoritative for the other, which is the back-door unification
# D7 exists to prevent.
#
# The nested containers are frozen on the way in (sim/fleet.py freezes lists to
# tuples and mappings to read-only views), so a structured param cannot be
# mutated after construction any more than a scalar one could -- and so no
# CostTerm instance aliases a live config object.
CostParam = float | int | str | bool | tuple["CostParam", ...] | Mapping[str, "CostParam"]


@dataclass(frozen=True, kw_only=True)
class CostTerm:
    """One entry selected from the ledger's cost-term library. ADR-0004 D7.

    `kind` is a NAME, not a function, and that is the point of maximum danger
    in the whole ADR. If an asset owned *the* cost function and both the ledger
    and the dispatcher's 5.2 called it, the two implementations would be
    unified through the back door and every cost-model error would go invisible
    (CLAUDE.md). Instead both sides implement `kind` independently and share
    only this vocabulary; their disagreement stays a measured output.

    A `kind` with no implementation in the ledger's library is rejected at
    config load -- one of the five rejections the ADR's gate names.
    """

    kind: str
    params: Mapping[str, CostParam] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class AssetCostData:
    """Declared cost data, never a cost method. See CostTerm."""

    terms: tuple[CostTerm, ...] = ()


@dataclass(frozen=True, kw_only=True)
class LoadCommand:
    """The CONTACTOR. ADR-0004 D6a, reshaped 2026-07-29.

    This was `served_p_kw: float` -- a continuous level the dispatcher chose to
    serve the load at -- and that was physically wrong for every load type that
    exists. You cannot serve a motor at 73%. You open a contactor or you do not,
    so the voluntary half of D6a is a switch and always was.

    Commanding `on=False` IS the voluntary shed, and the dispatcher pays that
    load's shed cost for the whole of its demand. Finer granularity comes from
    DECLARING MORE LOADS -- a feeder with three sheddable groups is three fleet
    members with their own priorities and shed prices -- not from a continuous
    knob no contactor could honour. That is also what makes the priority
    ordering mean something.

    Nothing is permanently given up. A genuinely modulatable load (EV charging,
    a variable-speed drive, a batch process that can be slowed) is a future load
    TYPE that declares `p_controllable` and carries a setpoint, exactly as an
    asset does. Today's `fixed` type simply must not claim a channel it has no
    means of honouring.

    None when NOT COMMANDED, which is distinct from a commanded False -- the
    same both-directions rule AssetCommand carries, and Fleet.validate_commands()
    enforces it for loads now too.
    """

    on: bool | None = None


@dataclass(frozen=True, kw_only=True)
class LoadMeasurement:
    """What one load was OBSERVED to get. ADR-0004 D6a.

    The dispatcher-visible projection of LoadDelivered (sim/load.py), and the
    test for membership here is whether a real site could METER it:

      `served_p_kw`   yes -- a meter at the load reads exactly this. Note it
                      does reveal true past demand on any interval that was
                      fully served, and that is fine: a real EMS knows what the
                      site consumed last interval. This is not the same as
                      AssetDelivered.curtailed_kw, which is COUNTERFACTUAL --
                      what PV could have made is not on any meter, and handing
                      it over would give away the resource the forecast is
                      supposed to be uncertain about.
      `unserved_p_kw` yes -- an outage is not subtle, and a dispatcher has to
                      know load was dropped on it.
      `shed_p_kw`     held back, and lossless to hold back: a voluntary shed is
                      the dispatcher's own command, so it already knows.
      `violations`    held back; the strings carry magnitudes.
    """

    served_p_kw: float
    served_q_kvar: float = 0.0
    unserved_p_kw: float = 0.0


@dataclass(frozen=True, kw_only=True)
class LoadObservation:
    """The degraded per-load view the dispatcher may see. ADR-0004 D6.

    `priority` is visible because the dispatcher is entitled to know the
    ranking that will be applied to it if physics forces an involuntary shed --
    even though it need not respect that ranking in its own voluntary shedding.
    Convention, asserted at config load and never inferred: LOWER number =
    more critical = shed LAST. A flipped convention sheds the most critical
    load first, costs the maximum every interval, and looks like a bad scenario
    rather than a sign bug.
    """

    priority: int
    measured_prev: LoadMeasurement | None  # None only at k=0


@dataclass(frozen=True, kw_only=True)
class AssetView:
    """One fleet asset, as the dispatcher may know it. ADR-0004 Stage E
    (implementation-plan decision 12).

    The controllability flags are the RESOLVED site choices, not the type's
    capability ceiling: a dispatcher that read the ceiling would command a
    channel this site declined, and validate_commands() would correctly
    refuse it. What crosses here is exactly what the dispatcher may act on.

    `ratings` is the nameplate -- static, datasheet, safe (D9). `cost_terms`
    is the declared vocabulary D7 requires the dispatcher's own cost model to
    price under its own independent implementation. What is deliberately NOT
    here: the spec (true efficiencies and derates -- handing them over makes
    the dispatcher's internal model correct by construction and destroys
    cost-model error as a measured output), state, and the interval envelope.
    """

    id: str
    type_name: str  # e.g. "pv"; lets a policy know a battery is a battery
    p_controllable: bool
    q_controllable: bool
    on_off_controllable: bool
    slack_bearing_capable: bool
    # See AssetCapabilities.resource_limited. Here because it is what lets a
    # policy iterate by CAPABILITY rather than by `type_name` -- it is the one
    # distinction that decides whether commanding zero suppresses anything, and
    # a policy that branches on the name instead silently mis-commands the first
    # new type anyone registers.
    resource_limited: bool
    ratings: AssetRatings
    cost_terms: tuple[CostTerm, ...]


@dataclass(frozen=True, kw_only=True)
class LoadView:
    """One fleet load, as the dispatcher may know it. `priority` is the
    involuntary-shed ordering the dispatcher is entitled to know (D6), even
    though its own voluntary shedding need not respect it.

    `on_off_controllable` is the load's declared channel, and it is here for the
    same reason AssetView carries the asset's: a policy iterating by capability
    must be able to see which members it may command before it commands them.
    Loads declaring their channels is what D6's "first-class and symmetric to
    assets" always claimed and did not deliver -- until 2026-07-29 an asset
    declared its channels and was validated in both directions while a load
    silently accepted a continuous setpoint it could not honour.

    There is deliberately no `p_controllable` yet. It belongs to the first
    genuinely modulatable load type, and declaring it now would be a capability
    nothing can honour -- the exact fault this field exists to fix.
    """

    id: str
    type_name: str
    priority: int
    on_off_controllable: bool
    cost_terms: tuple[CostTerm, ...]


@dataclass(frozen=True, kw_only=True)
class PlantModel:
    """What the dispatcher BELIEVES about one member's physics. ADR-0009 D3.

    THE DISTINCTION AGAINST AssetRatings, and it is the whole reason these are
    two types rather than one. A rating is a DATASHEET FACT: a 300 kW inverter is
    a 300 kW inverter on the day it ships and on the day it is decommissioned, an
    operator reads it off a nameplate, and an estimator has no business
    overwriting it. A believed parameter is the opposite in every respect. Usable
    capacity falls with cycling, round-trip efficiency falls as cell impedance
    grows, a fuel curve drifts with injector fouling, an array's performance ratio
    falls with soiling and recovers after a wash. NOBODY KNOWS THEIR CURRENT
    VALUES. They are inferred from operating data and they change over a run.

    Merge the two and the information that tells a recalibration layer which
    numbers it may touch is gone, and is not recoverable.

    THE BELIEF IS ALLOWED TO BE WRONG, and that is the point rather than a
    tolerated defect. Handing the dispatcher `SiteConfig` would make its internal
    model correct by construction, every horizon plan exactly realisable, and the
    estimator's target identically zero -- the same trap D12 avoided one level up,
    and the same principle that keeps sim/ledger.py and dispatch/'s cost model as
    two independent implementations. A belief EQUAL to the truth is legal: it is
    the control condition and must be expressible. What is not legal is arriving
    at it by omission, which is why `believed:` has no default anywhere.

    This is the belief at k=0 -- a PRIOR. A recalibrating estimator (Stage 5+)
    holds its own updated copy in dispatcher state; FleetView stays frozen and is
    never mutated.
    """


@dataclass(frozen=True, kw_only=True)
class EmptyPlantModel(PlantModel):
    """Nothing to estimate: the grid, and every load today.

    A real value rather than a missing key, so `believed[member_id]` is total and
    a dispatcher never branches on absence. It is also what makes D2's mandatory
    empty block meaningful -- a member whose beliefs were FORGOTTEN would
    otherwise be indistinguishable from one that correctly has none.
    """


@dataclass(frozen=True, kw_only=True)
class BessPlantModel(PlantModel):
    """`usable_capacity_kwh` is the band the policy can actually cycle, NOT the
    nameplate -- `AssetRatings.energy_capacity_kwh` already publishes that, and a
    belief restating a rating would be a parameter with nothing to be wrong about.
    """

    usable_capacity_kwh: float
    round_trip_efficiency: float

    def __post_init__(self) -> None:
        if self.usable_capacity_kwh <= 0.0:
            raise ValueError(
                f"usable_capacity_kwh={self.usable_capacity_kwh} must be positive; a store "
                f"believed to hold nothing is not a store"
            )
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError(
                f"round_trip_efficiency={self.round_trip_efficiency} must lie in (0, 1]. A "
                f"believed efficiency above 1 is a battery that makes energy."
            )


@dataclass(frozen=True, kw_only=True)
class BelievedFuelCurve:
    """L/h = a + b*P_kw + c*P_kw^2. The same SHAPE as config's FuelCurve.

    One parameter, not three. Flattening the coefficients into sibling fields
    invites a belief carrying two of the three, and forces a translation table on
    anyone reading the belief beside the truth.

    An INDEPENDENT restatement, necessarily: contracts.py imports nothing internal
    and so cannot reach `config.schema.FuelCurve`. The field names match on
    purpose -- `a` is the L/h term, `c` the L/h/kW^2 one, in that order -- because
    the one thing worse than restating a shape is restating it transposed.
    """

    a: float
    b: float
    c: float

    def __post_init__(self) -> None:
        # Concavity is the shape of the fit, not a range check: a genset burns
        # proportionally LESS per kW as it loads up. A convex believed curve would
        # send an optimiser the opposite way at every commitment decision, and
        # ADR-0008's tangent-hull reasoning assumes the sign.
        if self.c >= 0.0:
            raise ValueError(
                f"fuel curve c={self.c} must be negative: the curve is concave (ADR-0003 D3). "
                f"A convex belief inverts every part-load decision an optimiser makes."
            )
        if self.b <= 0.0:
            raise ValueError(f"fuel curve b={self.b} must be positive: more load burns more fuel")


@dataclass(frozen=True, kw_only=True)
class GensetPlantModel(PlantModel):
    """`min_stable_load_fraction` is the one entry here that is not about
    degradation, and it is worth stating rather than hiding.

    ADR-0006 records the gap it closes: minimum stable load lives in `GensetSpec`,
    `AssetRatings.p_min_kw` is 0.0 because a box rating cannot express a FORBIDDEN
    BAND, and a dispatcher therefore cannot see a below-minimum commit coming --
    hence `GENSET_COMMIT_AT_RATED` as a workaround. This is the only channel that
    can carry it, so it goes here and the workaround becomes optional. It is
    estimable in principle (from observed refusals) and effectively fixed in
    practice; an estimator should leave it alone.
    """

    fuel_curve: BelievedFuelCurve
    min_stable_load_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.min_stable_load_fraction < 1.0:
            raise ValueError(
                f"min_stable_load_fraction={self.min_stable_load_fraction} must lie strictly "
                f"in (0, 1); it is a fraction of the rating, not a kW figure"
            )


@dataclass(frozen=True, kw_only=True)
class PvPlantModel(PlantModel):
    """`performance_ratio` has NO counterpart in the true spec, and that is legal.

    Truth applies no soiling or degradation term today (ADR-0009 follow-up): the
    field is added here as the dispatcher's ESTIMATE, and the plant physics behind
    it is built later. A belief need not mirror a spec field -- that requirement
    was dropped -- so this is a belief about something the plant does not yet
    model, which is the clearest case of a counterpart missing rather than merely
    differing.
    """

    performance_ratio: float
    bifacial_gain_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.performance_ratio <= 1.0:
            raise ValueError(
                f"performance_ratio={self.performance_ratio} must lie in (0, 1]. Above 1 is an "
                f"array believed to out-perform its own nameplate."
            )
        if self.bifacial_gain_fraction < 0.0:
            raise ValueError(
                f"bifacial_gain_fraction={self.bifacial_gain_fraction} is negative: a rear face "
                f"collecting light cannot subtract from the front"
            )


@dataclass(frozen=True, kw_only=True)
class FleetView:
    """What fleet am I driving? Delivered ONCE, at dispatcher construction.

    The dispatcher-visible face of the fleet: ordered ids (declaration order,
    D10 -- iterating this and summing is reproducible), per-member resolved
    channels, nameplates, priorities and the declared cost vocabulary.
    Everything a policy needs to iterate by capability instead of hardcoding
    asset names (Stage H), and nothing that would let it reconstruct truth.

    NOT SiteConfig, and that is the load-bearing exclusion: config holds the
    plant's true efficiencies, so handing it over would make the dispatcher's
    model correct by construction (D7, CLAUDE.md).

    `believed` is how a policy gets a plant model anyway, without getting the
    TRUE one. See PlantModel.
    """

    assets: tuple[AssetView, ...]
    loads: tuple[LoadView, ...]
    # What the dispatcher is assumed to know about each member's physics
    # (ADR-0009 D3). TOTAL over `asset_ids + load_ids`: every member has an entry,
    # `EmptyPlantModel` where there is nothing to estimate, so indexing this never
    # branches on absence.
    #
    # REQUIRED, with no default, and that is D2 at the type level. A default empty
    # mapping would make a fleet whose beliefs were forgotten indistinguishable
    # from one that correctly has none -- and unlike a forgotten rating, a
    # forgotten belief is invisible in the RESULTS rather than merely wrong.
    #
    # Deliberately NOT merged into AssetRatings: ratings are datasheet facts and
    # beliefs decay, and that distinction is what tells a recalibration layer
    # which numbers it may overwrite. No parameter appears in both.
    believed: Mapping[str, PlantModel]
    # The config slack PRIORITY ORDER (decision 29), best first. Not truth: it
    # is a config choice, and D5 calls the advisory-versus-delivered delta a
    # primary diagnostic -- which presumes the dispatcher knows where to aim it.
    #
    # This is how a policy learns who is holding the bus WITHOUT a per-interval
    # channel: the plant takes the first entry that is available and not
    # commanded off, and availability already crosses on every AssetObservation.
    # So the designation is derivable from what the dispatcher legitimately
    # knows, which is what decision 29 bought and why Stage H item 2 needed no
    # observation field in the end.
    slack_priority: tuple[str, ...] = ()
    # THE PLANT STEP, in minutes: how often an observation arrives and how wide
    # an interval the plant integrates over. A declared config parameter both
    # sides are MEANT to know -- a policy converting a setpoint to energy needs
    # it to say anything at all, and one that did not know it could not tell how
    # often it is being sampled.
    #
    # MINUTES, and not hours, because `dt_minutes` is the one declaration of how
    # long an interval is. Hours are a CONVERSION, computed where a kW becomes a
    # kWh -- `contracts.hours_in()` -- never a stored second representation that
    # can drift from the first.
    #
    # Here because it was previously nowhere: `dispatch/rule_based.py` carried
    # `_DT_HOURS = 0.25` as a literal, which is the bare float in physics code
    # CLAUDE.md calls a bug, and it silently assumed a 15-minute plant. A run at
    # dt_minutes: 5 would have had every stored-energy calculation in `dispatch/`
    # wrong by 3x with nothing to say so.
    #
    # No default, for that exact reason. The dataclass is kw_only, so a
    # defaultless field is free to follow a defaulted one.
    dt_minutes: int
    # HOW MANY PLANT STEPS THIS POLICY'S COMMANDS APPLY FOR. The runner calls a
    # dispatcher every interval; a dispatcher whose multiple is above 1
    # re-optimises every that-many intervals and its earlier commands are held in
    # between. So:
    #
    #     observation period = dt_minutes
    #     control period     = dt_minutes * dispatch_interval_multiple
    #
    # BOTH cross, and neither is derivable from the other. A policy converting a
    # setpoint to energy needs the CONTROL period -- that is how long the command
    # it is about to issue will actually be applied for. A policy building a
    # history, a forecast or a state estimate needs the OBSERVATION period,
    # because that is the rate its measurements arrive at. Publishing only one
    # would leave the other assumed.
    #
    # 1 is the ordinary case and makes the two identical, which is what every run
    # before this was.
    dispatch_interval_multiple: int = 1

    def __post_init__(self) -> None:
        if self.dt_minutes <= 0:
            raise ValueError(
                f"dt_minutes={self.dt_minutes} is not positive. The plant step is a "
                f"declared parameter of the run, not something a policy may assume."
            )
        if self.dispatch_interval_multiple < 1:
            raise ValueError(
                f"dispatch_interval_multiple={self.dispatch_interval_multiple} is below 1. "
                f"A policy cannot decide more often than the plant steps."
            )

    @property
    def control_period_minutes(self) -> int:
        """How long a command issued now will be applied for.

        Derived, never declared: it is the plant step times the multiple, and a
        third field carrying it could disagree with the two it comes from.
        """
        return self.dt_minutes * self.dispatch_interval_multiple

    def slack_id(self, available: Mapping[str, bool]) -> str:
        """Who the plant will treat as the bus-holder, given what is available.

        The dispatcher's own copy of the plant's resolution rule -- deliberately
        a copy, not a shared call: `sim/` and `dispatch/` cannot import each
        other, and this is the same independent-implementation discipline D7
        applies to cost. If the two ever disagree, `slack_delta_kw` says so,
        which is exactly what that diagnostic is for.

        It cannot see the commanded-off test the plant also applies, because a
        policy calls this while deciding what to command. That is a real limit:
        a policy that switches a genset off and expects to keep the bus will be
        wrong for one interval and learn it from the delta.
        """
        for asset_id in self.slack_priority:
            if available.get(asset_id, False):
                return asset_id
        return self.slack_priority[0] if self.slack_priority else ""

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.assets)

    @property
    def load_ids(self) -> tuple[str, ...]:
        return tuple(ln.id for ln in self.loads)

    def asset(self, asset_id: str) -> AssetView:
        for a in self.assets:
            if a.id == asset_id:
                return a
        raise KeyError(f"no asset {asset_id!r} in the fleet view")

    def load(self, load_id: str) -> LoadView:
        for ln in self.loads:
            if ln.id == load_id:
                return ln
        raise KeyError(f"no load {load_id!r} in the fleet view")


@dataclass(frozen=True, kw_only=True)
class LoadCostData:
    """Declared cost data, never a cost method. The mirror of AssetCostData.

    ONE representation of the shed price, and it is the declared term itself.
    This carried an extracted `shed_cost_inr_per_kwh` float beside the terms for
    a while, which meant one object held the same number twice with nothing
    checking they agreed -- and the extraction had its own refusals, separate
    from the ones the params model already enforces. `UnservedParams` in
    sim/load.py is now the single reading, and everything that needs the price
    goes through it.

    Kept separate from `priority` on purpose. Priority is an ORDERING (who gets
    dropped first); a price is what any unserved kWh costs, however it came to
    be unserved. They usually agree, and the point is the case where they do
    not: a furnace mid-melt with a moderate per-kWh cost should still be shed
    last, because cost per kWh cannot express restart pain.
    """

    terms: tuple[CostTerm, ...] = ()
