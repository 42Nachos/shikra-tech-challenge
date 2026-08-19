"""The TNERC HT-I utility connection: its physics and its tariff.

ONE MODULE FOR BOTH HALVES. The physics is a PCC interface transformer with a
symmetric apparent-power rating; the cost is the four declared tariff terms as
one object. Swap this module and you have a different utility, with a different
tariff STRUCTURE and possibly different equipment.

THE PHYSICS IS THIN. Every kW crossing the PCC in either direction goes through
the same windings, so one rating bounds import and export alike and `step()` is a
clip. The rating is LIVE but has never fired on a shipped scenario: peak
|grid_kw| across the goldens is 786.6 kW against 1000 kVA.

A DOWN GRID CARRIES NO FLOW IN EITHER DIRECTION. Islanded surplus backoff is the
plant's job (D5 secondary slack), not this model's.

THE TARIFF: four declared terms, ONE object (ADR-0016 D4), because three of the
four are not independent:

    energy         ToU-rated import. Sets the rate the other two read.
    export         credited at the IMPORT rate.
    demand_charge  the month's largest block-integrated kVA, floored.
    pf_penalty     a percentage of the month's CONSUMPTION CHARGES.

As four objects that coupling needed a shared mutable context and a
writer-before-reader rank. Here the rate is a local, the period's charges are a
field, and the ordering is the order of the statements in `update()`.

THIS IS ONE SITE'S TARIFF STRUCTURE. An 11-month demand ratchet, or a block rate
whose marginal price falls with consumption, is a DIFFERENT structure and belongs
in a second class beside this one.

Config is untouched: the four terms are still declared separately under
`cost_terms:`, because `dispatch/` reads exactly those off `FleetView` to build
its own independent 5.2 model (ADR-0004 D7).
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from dataclasses import dataclass, field
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from contracts import AssetCommand, MINUTES_PER_DAY, CostTerm, hours_in, minutes_of_day
from sim.asset import AssetDelivered, AssetEnvelope, AssetState
from sim.scenario import Interval, ScenarioRow
from sim.tariff import TouRateSchedule


@dataclass(frozen=True, kw_only=True)
class GridState(AssetState):
    """Stateless. Declared anyway, so the first thing the connection must
    remember lands here without moving a signature."""


class TnercHtSpec(BaseModel):
    """The connection's equipment rating.

    EQUIPMENT, not tariff: this rates what CAN flow, while contracted demand on
    the `demand_charge` term prices what does.
    """

    model_config = ConfigDict(frozen=True)

    transformer_s_max_kva: float = Field(gt=0)

    def build(self, member_id: str, dt_minutes: int) -> TnercHtModel:
        """This spec's model, bound to its member and the plant step.

        ON THE SPEC, so there is no second table mapping names to classes.
        """
        return TnercHtModel(member_id=member_id, spec=self, dt_minutes=dt_minutes)


@dataclass(frozen=True, kw_only=True)
class TnercHtModel:
    """The connection's physics, bound to its rating and the plant step."""

    member_id: str
    spec: TnercHtSpec
    dt_minutes: int

    @property
    def s_max_kva(self) -> float:
        return self.spec.transformer_s_max_kva

    @property
    def p_limits_kw(self) -> tuple[float, float]:
        """Symmetric: the same windings carry both directions."""
        return -self.spec.transformer_s_max_kva, self.spec.transformer_s_max_kva

    def required_scenario_fields(self) -> frozenset[str]:
        """Nothing exogenous: the tariff is declared, and `timestamp` is on every row."""
        return frozenset()

    def initial_state(self, carry_in: Mapping[str, float]) -> GridState:
        if carry_in:
            raise ValueError(
                f"asset {self.member_id!r}: the grid carries no state, but carry_in declares "
                f"{sorted(carry_in)}. Refused rather than ignored."
            )
        return GridState()

    def envelope(self, state: AssetState, row: ScenarioRow) -> AssetEnvelope:
        self.narrow(state)
        if not row.available[self.member_id]:
            return AssetEnvelope(p_min_kw=0.0, p_max_kw=0.0)
        s_max_kva = self.spec.transformer_s_max_kva
        return AssetEnvelope(p_min_kw=-s_max_kva, p_max_kw=s_max_kva, s_max_kva=s_max_kva)

    def step(
        self, state: AssetState, cmd: AssetCommand, row: ScenarioRow
    ) -> tuple[GridState, AssetDelivered]:
        grid_state = self.narrow(state)
        assert cmd.p_setpoint_kw is not None, "the port refuses an absent setpoint"
        setpoint_kw = cmd.p_setpoint_kw
        violations: list[str] = []

        if not row.available[self.member_id]:
            # No flow either direction. A nonzero setpoint to a dead grid is a
            # dispatcher error and is recorded.
            if setpoint_kw != 0.0:
                violations.append(f"grid_unavailable:{setpoint_kw:.1f}")
            delivered_kw = 0.0
        else:
            # Clip-and-report against the transformer, both directions. While
            # slack-bearing, the plant passes the balancing value here; a
            # balance the transformer cannot carry comes back clipped, and the
            # residual is the plant's D5 secondary slack to spill.
            s_max_kva = self.spec.transformer_s_max_kva
            delivered_kw = min(max(setpoint_kw, -s_max_kva), s_max_kva)
            if delivered_kw != setpoint_kw:
                violations.append(f"grid_transformer_clipped:{setpoint_kw:.1f}->{delivered_kw:.1f}")

        return grid_state, AssetDelivered(
            p_net_kw=delivered_kw,
            q_net_kvar=0.0,  # the port clips reactive against the kVA circle
            violations=tuple(violations),
        )

    def build_cost(self, declared: tuple[CostTerm, ...]) -> TnercHtCost:
        return TnercHtCost.build(member_id=self.member_id, declared=declared)

    def check_delivery(
        self,
        state_before: AssetState,
        state_after: AssetState,
        delivered: AssetDelivered,
        row: ScenarioRow,
    ) -> None:
        """A connection has no conservation law of its own: `step()` bounds it."""

    def narrow(self, state: AssetState) -> GridState:
        if not isinstance(state, GridState):
            raise TypeError(f"the tnerc_ht model expects GridState, got {type(state).__name__}")
        return state


# Float residue in a shortfall/step division, absorbed before the ceiling. NOT a
# tariff quantity: that is `PfPenaltyParams.pf_step`, declared in config.
_PF_STEP_TOL = 1e-9


class ExportParams(BaseModel):
    """Net metering. `credit_at_import_rate` is a flag rather than a rate because
    that IS the tariff's rule; a second rate schedule could disagree."""

    model_config = ConfigDict(frozen=True)

    credit_at_import_rate: bool


class DemandChargeParams(BaseModel):
    """Max recorded kVA over the billing month, floored at a percentage of
    contracted demand (TNERC clause 3.1.1.7). There is no 11-month ratchet in
    this tariff (CLAUDE.md).

    `assumed_power_factor` is here because the meter records kVA and the plant
    produces kW: with no reactive data, kVA = kW / pf.
    """

    model_config = ConfigDict(frozen=True)

    inr_per_kva_month: float = Field(ge=0)
    contracted_demand_kva: float = Field(gt=0)
    billable_demand_floor_pct: float = Field(ge=0, le=1)
    assumed_power_factor: float = Field(gt=0, le=1)
    # THE INTEGRATION WINDOW (TNERC clause 3.1.1.8). A maximum demand is the
    # highest AVERAGE power over a fixed block, not the highest instantaneous
    # reading, and the block is anchored to the clock -- 00:00-00:15, 00:15-00:30
    # -- exactly as a real MD meter integrates.
    #
    # Must divide the day, so a block can never straddle midnight and therefore
    # never straddles a month boundary. That is what lets `close_period()` settle
    # without a carry rule.
    demand_window_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_window_divides_the_day(self) -> Self:
        if MINUTES_PER_DAY % self.demand_window_minutes != 0:
            raise ValueError(
                f"demand_window_minutes={self.demand_window_minutes} does not divide the day "
                f"into whole blocks. A block that straddles midnight would also straddle a "
                f"month boundary, and the period's maximum would depend on where the run "
                f"happened to start."
            )
        return self


class PfPenaltyBand(BaseModel):
    """Low-power-factor compensation band (TNERC clause 3.1.1.6). Applies when
    the average PF falls in [pf_low, pf_high). The charge is
    `pct_of_charges_per_step` percent of the period consumption charges for
    every `pf_step` below the limit (always measured from the limit, and rounded
    up -- see `PfPenaltyParams.steps_below_limit`)."""

    model_config = ConfigDict(frozen=True)

    pf_low: float = Field(ge=0, le=1)
    pf_high: float = Field(gt=0, le=1)
    pct_of_charges_per_step: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_band_is_non_empty(self) -> Self:
        if self.pf_low >= self.pf_high:
            raise ValueError(f"pf band [{self.pf_low}, {self.pf_high}) is empty or inverted")
        return self


class PfPenaltyParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_power_factor: float = Field(gt=0, le=1)
    assumed_power_factor: float = Field(gt=0, le=1)
    # The tariff's QUANTUM: the penalty is charged per this much power factor
    # below the limit. TNERC 3.1.1.6 says 0.01; another regulator may not, and a
    # step written into the code cannot be re-declared without editing both this
    # module and the oracle's mirror of it.
    pf_step: float = Field(gt=0, le=1)
    bands: tuple[PfPenaltyBand, ...] = Field(min_length=1)

    def steps_below_limit(self, pf: float) -> int:
        """How many whole `pf_step` steps `pf` falls short of the limit.

        CEILING, not nearest: rounding to nearest hands back a free half-step.

        On the params class so the ledger and the ADR-0008 oracle produce the
        SAME integer. Not shared with `dispatch/`, which implements 5.2
        independently (D7).

        The tolerance absorbs float residue: `(0.90 - 0.87) / 0.01` evaluates to
        3.0000000000000027, and a bare `ceil` would charge four steps for three.
        """
        shortfall = self.min_power_factor - pf
        if shortfall <= 0.0:
            return 0
        return math.ceil(shortfall / self.pf_step - _PF_STEP_TOL)


# The order a member's lines are emitted in, and it is not cosmetic:
# `CostBreakdown.total_inr` sums in iteration order and floating-point addition
# is not associative, so this is part of "same config = bit-identical output".
# It matches the order the term library emitted (writers before readers, then
# declaration order), which is why the move costs nothing.
# Derived from KIND_PARAMS below, which is an ordered mapping: one declaration
# of the order, two readers (this, and the oracle's `declared_line_keys`).


@dataclass(kw_only=True)
class TnercHtCost:
    """One connection's bill, accumulated over a run.

    MUTABLE, and held by the ledger rather than by the frozen asset (ADR-0004
    D1); an accumulator on `GridState` would be declarable in `carry_in:`.
    """

    KIND_PARAMS: ClassVar[Mapping[str, type[BaseModel]]] = {
        "energy": TouRateSchedule,
        "export": ExportParams,
        "demand_charge": DemandChargeParams,
        "pf_penalty": PfPenaltyParams,
    }
    KINDS: ClassVar[frozenset[str]] = frozenset(KIND_PARAMS)

    member_id: str
    declared: tuple[str, ...]
    energy: TouRateSchedule | None = None
    export: ExportParams | None = None
    demand_charge: DemandChargeParams | None = None
    pf_penalty: PfPenaltyParams | None = None

    _totals: dict[str, float] = field(default_factory=dict, init=False)
    # Accumulated across the BILLING PERIOD, reset when it closes. What the PF
    # penalty is a percentage of.
    _period_energy_charge_inr: float = field(default=0.0, init=False)
    # The open maximum-demand block, and the readings inside it.
    _period_max_kva: float = field(default=0.0, init=False)
    _block: tuple[int, int, int] | None = field(default=None, init=False)
    _block_kw: list[float] = field(default_factory=list, init=False)

    @classmethod
    def build(cls, *, member_id: str, declared: tuple[CostTerm, ...]) -> TnercHtCost:
        """Validate this member's declared terms into one tariff, or refuse.

        Validated at construction, which is startup, so a mistyped tariff refuses
        to start rather than mispricing a run. `export` and `pf_penalty` need a
        sibling `energy` term for their rate and base; a repeated kind would
        collapse into one line.
        """
        kinds = [term.kind for term in declared]
        unknown = [k for k in kinds if k not in cls.KINDS]
        if unknown:
            raise ValueError(
                f"{member_id!r}: the grid tariff prices {sorted(cls.KINDS)}, but the fleet "
                f"declares {sorted(set(unknown))}. A term nobody implements must be refused, "
                f"not billed as zero."
            )
        if len(set(kinds)) != len(kinds):
            raise ValueError(
                f"{member_id!r}: cost term kind declared twice ({sorted(kinds)}). Each kind is "
                f"one line in the breakdown, so a repeat would collapse into one."
            )

        params: dict[str, BaseModel] = {}
        for term in declared:
            try:
                params[term.kind] = cls.KIND_PARAMS[term.kind].model_validate(dict(term.params))
            except ValidationError as exc:
                raise ValueError(
                    f"{member_id!r}: cost term {term.kind!r} has invalid params: {exc}"
                ) from None

        # THE RULE IS ABOUT THE DEPENDENCY, NOT THE TERM. An `export` that does
        # NOT credit at the import rate reads nothing from `energy`, so refusing
        # it would be a rule with no reason behind it.
        export = params.get("export")
        needs_energy = [
            kind
            for kind, needed in (
                ("export", isinstance(export, ExportParams) and export.credit_at_import_rate),
                ("pf_penalty", "pf_penalty" in params),
            )
            if needed
        ]
        for kind in needs_energy:
            if "energy" not in params:
                reads = (
                    "exports credit at the import rate"
                    if kind == "export"
                    else "the penalty is a percentage of the period's consumption charges"
                )
                raise ValueError(
                    f"{member_id!r} declares a {kind!r} term but no `energy` term to set that "
                    f"rate: {reads}, so without one it would price at zero rather than fail."
                )

        return cls(
            member_id=member_id,
            declared=tuple(k for k in cls.KIND_PARAMS if k in params),
            **params,  # type: ignore[arg-type]
        )

    def update(self, delivered: AssetDelivered, iv: Interval) -> None:
        """Price one interval. Statement order IS the dependency order: `energy`
        first, because both readers below need the rate it computes and the
        running total it accumulates.
        """
        rate_inr_per_kwh = 0.0
        if self.energy is not None:
            rate_inr_per_kwh = self.energy.rate_inr_per_kwh(minutes_of_day(iv.row.timestamp))
            if delivered.p_net_kw >= 0.0:
                charge = delivered.p_net_kw * hours_in(iv.dt_minutes) * rate_inr_per_kwh
                self._add("energy", charge)
                self._period_energy_charge_inr += charge

        if self.export is not None and self.export.credit_at_import_rate:
            if delivered.p_net_kw < 0.0:
                exported_kw = -delivered.p_net_kw
                self._add("export", -(exported_kw * hours_in(iv.dt_minutes) * rate_inr_per_kwh))

        if self.demand_charge is not None:
            block = (
                iv.ts.year,
                iv.ts.timetuple().tm_yday,
                minutes_of_day(iv.row.timestamp) // self.demand_charge.demand_window_minutes,
            )
            if self._block is not None and block != self._block:
                self._close_block()
            self._block = block
            self._block_kw.append(delivered.p_net_kw)

    def close_period(self) -> None:
        """Settle the two path-dependent terms. The LEDGER decides when a billing
        period ends and calls this."""
        if self.demand_charge is not None:
            self._close_block()
            floor_kva = (
                self.demand_charge.billable_demand_floor_pct
                * self.demand_charge.contracted_demand_kva
            )
            billable_kva = max(self._period_max_kva, floor_kva)
            self._add("demand_charge", billable_kva * self.demand_charge.inr_per_kva_month)
            self._period_max_kva = 0.0
            self._block = None

        if self.pf_penalty is not None:
            p = self.pf_penalty
            pf = p.assumed_power_factor
            if pf < p.min_power_factor:
                steps = p.steps_below_limit(pf)
                for band in p.bands:
                    if band.pf_low <= pf < band.pf_high:
                        self._add(
                            "pf_penalty",
                            self._period_energy_charge_inr
                            * (band.pct_of_charges_per_step / 100.0)
                            * steps,
                        )
                        break

        # LAST, after every term that reads it has settled.
        self._period_energy_charge_inr = 0.0

    def lines(self) -> Mapping[str, float]:
        """One entry per DECLARED term, in `KIND_PARAMS` order. A term declared
        but never spent on still gets a zero line."""
        return {kind: self._totals.get(kind, 0.0) for kind in self.declared}

    def _add(self, kind: str, amount_inr: float) -> None:
        self._totals[kind] = self._totals.get(kind, 0.0) + amount_inr

    def _close_block(self) -> None:
        """A maximum-demand meter integrates over its window and reports the
        block's average; the month's charge is the largest of those.

        NET over the block, not per-interval `max(0, .)`, so an exporting
        interval offsets an importing one inside the same block. Floored at zero
        at the block's END. At `window == dt` this is `max(0, p)` exactly.
        """
        if not self._block_kw:
            return
        assert self.demand_charge is not None
        mean_kw = sum(self._block_kw) / len(self._block_kw)
        kva = max(0.0, mean_kw) / self.demand_charge.assumed_power_factor
        self._period_max_kva = max(self._period_max_kva, kva)
        self._block_kw = []
