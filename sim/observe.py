"""Turn truth into what the dispatcher sees, per member. ADR-0004.

Each member degrades its own truth in its own observe(). This module is only the
fan-out: it cuts the previous interval's keyed delivered records down to their
measurement() projections and hands every member its own.

It also owns WHICH STREAM each member draws from (ADR-0012 D3): one per member,
so changing one member's instruments cannot reshuffle another's draws.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from contracts import Observation
from sim.asset import Asset
from sim.load import Load
from sim.noise import member_rng
from sim.plant import FleetDelivered, PlantState
from sim.scenario import ScenarioRow


def observe(
    assets: tuple[Asset, ...],
    loads: tuple[Load, ...],
    state: PlantState,
    row: ScenarioRow,
    delivered_prev: FleetDelivered | None,
    rngs: Mapping[str, random.Random],
) -> Observation:
    """Build the keyed Observation for interval k.

    `delivered_prev` is truth for k-1; each member receives ONLY its own
    measurement() projection, so no full delivered record crosses this return.

    `rngs` is keyed by member id and TOTAL over the fleet, so a missing key is a
    KeyError rather than a member sharing someone else's stream.
    """
    return Observation(
        k=row.k,
        # Un-noised: the clock is not an instrument reading (ADR-0013).
        timestamp=row.timestamp,
        assets={
            a.id: a.observe(
                state.assets[a.id],
                row,
                delivered_prev.assets[a.id].measurement() if delivered_prev else None,
                rngs[a.id],
            )
            for a in assets
        },
        loads={
            ln.id: ln.observe(
                state.loads[ln.id],
                row,
                delivered_prev.loads[ln.id].measurement() if delivered_prev else None,
                rngs[ln.id],
            )
            for ln in loads
        },
    )


def member_rngs(
    seed: int, assets: tuple[Asset, ...], loads: tuple[Load, ...]
) -> Mapping[str, random.Random]:
    """One independent stream per fleet member, derived from the run seed.

    Built ONCE per run and advanced across intervals; reseeding every interval
    would draw the same value every interval. Ids are unique across assets AND
    loads, so one flat mapping cannot collide.
    """
    members: tuple[Asset | Load, ...] = (*assets, *loads)
    return {member.id: member_rng(seed, member.id) for member in members}
