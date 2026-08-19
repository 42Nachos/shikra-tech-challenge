"""The plant: truth + physics + clip-and-report, over an assembled fleet.

ADR-0004. Every asset clips its own command, the slack-bearing asset absorbs the
balance within its envelope (D5 primary slack), and whatever no asset can carry
becomes curtailment or involuntary shed (D5 secondary slack, D6a). The energy
balance is a signed sum over the fleet.

Every record here is keyed by member (D9). The plant never sheds a load
voluntarily (D6a).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass

from contracts import Commands, Quality
from sim.asset import Asset, AssetDelivered, AssetState
from sim.fleet import AssembledFleet, Fleet
from sim.load import Load, LoadDelivered, LoadState
from sim.scenario import ScenarioRow

# Numerical tolerance on comparisons, the same figure sim/asset.py and
# sim/load.py use. Not physics.
_EPS = 1e-9

# The energy-balance tolerance, deliberately looser than `_EPS`: the balance is
# a sum over the whole fleet and accumulates rounding from every member.
# CLAUDE.md: never loosen this to make a test pass.
_BALANCE_TOL_KW = 1e-6


@dataclass(frozen=True, kw_only=True)
class PlantState:
    """The TRUE internal state: one AssetState/LoadState per member, keyed by id
    (D9). dispatch/ must never see it. Nothing may iterate these dicts for
    arithmetic -- iterate the fleet (declaration order, D10) and index in."""

    assets: Mapping[str, AssetState]
    loads: Mapping[str, LoadState]


@dataclass(frozen=True, kw_only=True)
class FleetDelivered:
    """The keyed truth of one interval: what every member did. Plant-side, and
    the source of the measured_prev projections observe() hands the dispatcher."""

    assets: Mapping[str, AssetDelivered]
    loads: Mapping[str, LoadDelivered]


@dataclass(frozen=True, kw_only=True)
class StepLog:
    """One interval of TRUE, KEYED record. ADR-0004 D9.

    Truth-side, not in contracts.py: it carries every asset's curtailed power and
    every asset's end-of-interval state.

    `asset_states` is the state AFTER the interval.

    `slack_advisory_kw` / `slack_delta_kw` are D5's primary diagnostic, both None
    when the slack asset declares no P channel.

    `violations` is the PLANT's own, in append order: island backoff,
    discretionary-draw relief, involuntary shed. Each member's own
    clip-and-report strings stay on that member's record.
    """

    k: int
    timestamp: str
    slack_id: str
    assets: Mapping[str, AssetDelivered]
    loads: Mapping[str, LoadDelivered]
    asset_states: Mapping[str, AssetState]
    violations: tuple[str, ...]
    slack_advisory_kw: float | None
    slack_delta_kw: float | None
    # Pass-through from the ScenarioRow, so data-quality figures can be
    # recomputed over an existing log. Not physics.
    quality: Quality


@dataclass(frozen=True, kw_only=True)
class Plant:
    """The frozen plant: fleet members plus the slack designation.

    `slack_priority` is the config-declared failover order (D5); the bus-holder
    is resolved from it EVERY interval, best entry first. `type_names` feeds the
    island violation strings, which name the TYPE that backed off.
    """

    assets: tuple[Asset, ...]
    loads: tuple[Load, ...]
    slack_priority: tuple[str, ...]
    type_names: Mapping[str, str]
    by_id: Mapping[str, Asset]

    @classmethod
    def build(cls, assembled: AssembledFleet, slack_priority: tuple[str, ...]) -> Plant:
        return cls(
            assets=assembled.assets,
            loads=assembled.loads,
            slack_priority=slack_priority,
            type_names={v.id: v.type_name for v in assembled.view.assets},
            by_id={a.id: a for a in assembled.assets},
        )

    def _resolve_slack(self, cmd: Commands, row: ScenarioRow) -> str:
        """Who holds the bus THIS interval. ADR-0004 D5.

        The first entry of the config priority order passing two tests:

          SCENARIO  available this interval
          COMMAND   not commanded OFF

        Envelope width is NOT a test: a battery too empty to help is still the
        bus-holder and simply carries nothing.

        FALLBACK: if nothing qualifies, the FIRST entry is returned as the
        nominal holder, and the secondary tier absorbs everything.
        """
        for asset_id in self.slack_priority:
            if not row.available[asset_id]:
                continue
            if cmd.assets[asset_id].on is False:
                continue
            return asset_id
        return self.slack_priority[0]

    def initial_state(self, fleet: Fleet) -> PlantState:
        """Every member's state at k=0, validated from its own carry_in block.
        Where an invalid carry-in refuses to start."""
        return PlantState(
            assets={a.id: a.initial_state(fleet.asset(a.id).carry_in) for a in self.assets},
            loads={
                ln.id: ln.initial_state(rl.carry_in)
                for ln, rl in zip(self.loads, fleet.loads, strict=True)
            },
        )

    def step(
        self, state: PlantState, cmd: Commands, row: ScenarioRow
    ) -> tuple[PlantState, FleetDelivered, StepLog]:
        next_assets: dict[str, AssetState] = {}
        delivered: dict[str, AssetDelivered] = {}
        plant_violations: list[str] = []

        # 0. Who holds the bus this interval. Resolved BEFORE anything steps.
        slack_id = self._resolve_slack(cmd, row)
        slack = self.by_id[slack_id]

        # 1. Every non-slack asset steps against its own command; each clips
        #    and reports for itself.
        for asset in self.assets:
            if asset.id == slack_id:
                continue
            s_next, d = asset.step(state.assets[asset.id], cmd.assets[asset.id], row)
            next_assets[asset.id] = s_next
            delivered[asset.id] = d

        # 2. What the dispatcher chose to serve (D6a: shed below demand is its
        #    economic choice; the plant only ever reduces further by physics).
        demand_kw: dict[str, float] = {}
        chosen_kw: dict[str, float] = {}
        total_chosen_kw = 0.0
        for load in self.loads:
            demand_kw[load.id] = load.profile(state.loads[load.id], row).p_kw
            # The contactor (D6a): open means the dispatcher shed the whole load
            # and pays for it, closed means the load draws what it draws.
            on = cmd.loads[load.id].on
            chosen_kw[load.id] = 0.0 if on is False else demand_kw[load.id]
            total_chosen_kw += chosen_kw[load.id]

        supply_wo_slack_kw = 0.0
        for asset in self.assets:
            if asset.id != slack_id:
                supply_wo_slack_kw += delivered[asset.id].p_net_kw

        # 3. Primary slack (D5): the slack asset's real power is determined by
        #    the balance; its commanded setpoint is advisory and overridden.
        #    A zero-width envelope means it cannot carry anything this
        #    interval (a down grid): step it at zero -- quiet by design -- and
        #    let secondary slack absorb the imbalance.
        balance_kw = total_chosen_kw - supply_wo_slack_kw
        slack_env = slack.envelope(state.assets[slack.id], row)
        slack_can_carry = not (slack_env.p_min_kw == 0.0 == slack_env.p_max_kw)
        advisory_kw = cmd.assets[slack.id].p_setpoint_kw
        effective = dataclasses.replace(
            cmd.assets[slack.id], p_setpoint_kw=balance_kw if slack_can_carry else 0.0
        )
        s_next, d = slack.step(state.assets[slack.id], effective, row)
        next_assets[slack.id] = s_next
        delivered[slack.id] = d

        # D5's primary diagnostic: the gap between the dispatcher's advisory and
        # what the bus actually demanded. Never "fix" a large delta by forcing
        # the command to equal the output -- the delta IS the measurement.
        #
        # Taken from the delivered record rather than `balance_kw`, so it
        # includes whatever the slack's own envelope clipped.
        slack_delta_kw = None if advisory_kw is None else d.p_net_kw - advisory_kw

        # 4. Secondary slack (D5). Surplus first: back injections off in FLEET
        #    DECLARATION ORDER, which config controls.
        def _supply_kw() -> float:
            total = 0.0
            for a in self.assets:
                if a.id != slack_id:
                    total += delivered[a.id].p_net_kw
            return total

        residual_kw = total_chosen_kw - (_supply_kw() + delivered[slack_id].p_net_kw)
        if residual_kw < -_EPS:
            surplus_kw = -residual_kw
            for asset in self.assets:
                if surplus_kw <= _EPS:
                    break
                if asset.id == slack_id:
                    continue
                d0 = delivered[asset.id]
                if d0.p_net_kw <= 0.0:
                    continue
                reduction_kw = min(surplus_kw, d0.p_net_kw)
                target_kw = d0.p_net_kw - reduction_kw
                if next_assets[asset.id] is state.assets[asset.id]:
                    # Stateless: adjust the record arithmetically; re-stepping
                    # would double-round curtailed_kw.
                    final = dataclasses.replace(
                        d0,
                        p_net_kw=target_kw,
                        curtailed_kw=d0.curtailed_kw + reduction_kw,
                    )
                else:
                    # Stateful: re-run the physics at the reduced setpoint. The
                    # asset may refuse the target. The re-step's violation
                    # strings are discarded and the island record replaces them;
                    # the original step's strings are kept.
                    eff2 = dataclasses.replace(cmd.assets[asset.id], p_setpoint_kw=target_kw)
                    s2, d2 = asset.step(state.assets[asset.id], eff2, row)
                    final = dataclasses.replace(d2, violations=d0.violations)
                    next_assets[asset.id] = s2
                delta_kw = d0.p_net_kw - final.p_net_kw
                surplus_kw -= delta_kw
                kind = "curtailed" if final.curtailed_kw > d0.curtailed_kw else "backed_off"
                plant_violations.append(f"island_{self.type_names[asset.id]}_{kind}:{delta_kw:.1f}")
                delivered[asset.id] = final
            residual_kw = total_chosen_kw - (_supply_kw() + delivered[slack_id].p_net_kw)

        # 5. Deficit. FIRST: a drawing asset (a charging BESS) is discretionary
        #    consumption, so its draw backs off, declaration order, before any
        #    load is shed. A charge continuing into a deficit would balance only
        #    by booking unserved ABOVE demand. No golden pins this; the
        #    hypothesis tests do.
        if residual_kw > _EPS:
            for asset in self.assets:
                if residual_kw <= _EPS:
                    break
                # THE SLACK IS NOT EXEMPT: with a battery as slack, its own
                # draw was set from a balance the backoff above has since
                # changed, so it must track the new one. Unreachable while the
                # slack is always the grid, which cannot draw discretionarily.
                d0 = delivered[asset.id]
                if d0.p_net_kw >= 0.0:
                    continue
                relief_kw = min(residual_kw, -d0.p_net_kw)
                target_kw = d0.p_net_kw + relief_kw
                eff3 = dataclasses.replace(cmd.assets[asset.id], p_setpoint_kw=target_kw)
                s3, d3 = asset.step(state.assets[asset.id], eff3, row)
                final = dataclasses.replace(d3, violations=d0.violations)
                next_assets[asset.id] = s3
                delta_kw = final.p_net_kw - d0.p_net_kw
                residual_kw -= delta_kw
                plant_violations.append(
                    f"deficit_{self.type_names[asset.id]}_charge_backed_off:{delta_kw:.1f}"
                )
                delivered[asset.id] = final

        #    THEN: involuntary shed, highest priority number first (D6a --
        #    physics, not a choice; the one case the plant sheds uncommanded).
        unserved_kw: dict[str, float] = {load.id: 0.0 for load in self.loads}
        if residual_kw > _EPS:
            remaining_kw = residual_kw
            for load in sorted(self.loads, key=lambda ln: -ln.priority):
                cut_kw = min(remaining_kw, chosen_kw[load.id])
                unserved_kw[load.id] = cut_kw
                remaining_kw -= cut_kw
                if remaining_kw <= _EPS:
                    break
            assert (
                remaining_kw <= _EPS
            ), f"k={row.k}: deficit {remaining_kw} kW exceeds everything served"
            # The string names the CONDITION -- nobody could form the bus --
            # rather than a type, which under a priority list could be any of
            # them.
            label = "no_slack_unserved" if not slack_can_carry else "slack_exhausted_unserved"
            plant_violations.append(f"{label}:{residual_kw:.1f}")

        # 6. Loads record what actually arrived; the voluntary/involuntary
        #    split is theirs to book (D6a).
        next_loads: dict[str, LoadState] = {}
        load_delivered: dict[str, LoadDelivered] = {}
        for load in self.loads:
            served_kw = chosen_kw[load.id] - unserved_kw[load.id]
            ls_next, ld = load.step(state.loads[load.id], cmd.loads[load.id], row, served_kw)
            next_loads[load.id] = ls_next
            load_delivered[load.id] = ld

        # 7. THE assertion: a pure conservation law over the fleet (D2).
        #    Fires every interval. Never loosen to make a test pass.
        served_total_kw = 0.0
        for load in self.loads:
            served_total_kw += load_delivered[load.id].served_p_kw
        supply_total_kw = _supply_kw() + delivered[slack_id].p_net_kw
        assert abs(supply_total_kw - served_total_kw) < _BALANCE_TOL_KW, (
            f"energy balance broken at k={row.k}: "
            f"supply={supply_total_kw} served={served_total_kw}"
        )

        # DECLARATION ORDER (D10) on the way out. The dicts above were filled
        # non-slack-first, and floating-point addition is not associative.
        ordered_assets = {a.id: delivered[a.id] for a in self.assets}
        ordered_states = {a.id: next_assets[a.id] for a in self.assets}
        ordered_loads = {ln.id: load_delivered[ln.id] for ln in self.loads}

        return (
            PlantState(
                assets=ordered_states, loads={ln.id: next_loads[ln.id] for ln in self.loads}
            ),
            FleetDelivered(assets=ordered_assets, loads=ordered_loads),
            StepLog(
                k=row.k,
                timestamp=row.timestamp,
                slack_id=slack_id,
                assets=ordered_assets,
                loads=ordered_loads,
                asset_states=ordered_states,
                violations=tuple(plant_violations),
                slack_advisory_kw=advisory_kw,
                slack_delta_kw=slack_delta_kw,
                quality=row.quality,
            ),
        )
