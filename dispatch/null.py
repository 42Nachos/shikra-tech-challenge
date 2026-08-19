"""The null dispatcher: move nothing, let the slack serve the load.

"Null" cannot mean "command nothing": an absent command is an error (D4), so
doing nothing is itself a full set of commands, one per declared channel
(implementation-plan decision 13). What nothing means, BY CAPABILITY -- no
branch below names a type, which is the whole point of Stage H:

    resource_limited   p = its own nameplate. The setpoint is a CEILING on such
                       an asset, so the rating is always a valid "run free" at
                       any light level. THE D4 TRAP read twice: a zero would
                       mean curtail-everything, and before the ceiling existed
                       the only complaint-free alternative was commanding the
                       forecast -- which is why a forecast used to cross the
                       boundary at all.
    the slack          p = persistence from its own measured_prev. Advisory
                       (D5): the plant delivers the balancing value, and the
                       gap between the two is the diagnostic. WHICH member is
                       the slack is derived from the FleetView's priority order
                       and this interval's availability, so a fleet whose bus
                       is held by a battery aims the advisory at the battery.
    everything else    p = 0, q = 0 and on = False where the channel is
                       declared. Off, and commanded off, distinctly (D4).
    loads              contactor closed: serve everything, shed nothing.

NO FORECAST IS READ, because none is published: the plant publishes what a site
can measure, and a policy builds its own model from that. Null's model is the
crudest possible one -- last interval's slack flow -- and the advisory delta
records exactly how wrong it is, every interval.

Deterministic and measurement-only: no truth, no state, no randomness. A newly
registered asset type gets a correct command here with no edit to this file.
"""

from __future__ import annotations

from contracts import (
    AssetCommand,
    Commands,
    FleetView,
    LoadCommand,
    Observation,
)


class NullDispatcher:
    def __init__(self, view: FleetView) -> None:
        self._view = view

    def step(self, obs: Observation) -> Commands:
        # Who the plant will treat as the bus-holder, derived from the config
        # priority order and this interval's availability -- both of which the
        # dispatcher legitimately knows (decision 29).
        available = {v.id: obs.assets[v.id].available for v in self._view.assets}
        slack_id = self._view.slack_id(available)

        # PERSISTENCE, from the slack's own last measurement. This is the whole
        # of null's "forecast", and it is built HERE, from a meter reading,
        # because forecasting is a dispatcher-side model. It used to be
        # `load_forecast - pv_forecast`, computed by the plant and handed over
        # exactly right -- which was the plant doing the dispatcher's job
        # perfectly and calling the result a forecast.
        #
        # None at k=0, when there is no previous interval: 0.0 then, and the
        # advisory delta records how wrong that was, which is what D5's primary
        # diagnostic is for.
        prev = obs.assets[slack_id].measured_prev
        slack_advisory_kw = prev.p_net_kw if prev is not None else 0.0

        asset_cmds: dict[str, AssetCommand] = {}
        for asset_view in self._view.assets:
            if asset_view.resource_limited:
                # Run free. The setpoint is a CEILING on this asset, so its own
                # nameplate is always a valid "do not curtail" instruction --
                # at any light level, with nothing recorded. No forecast needed,
                # which is the point: doing nothing to PV must not require
                # knowing what the sun is doing.
                p_kw: float | None = asset_view.ratings.p_max_kw
            elif asset_view.id == slack_id:
                # Advisory only (D5): the plant delivers the balancing value.
                # Identified by the RESOLUTION rather than by `type_name ==
                # "grid"`, so a fleet whose bus is held by a battery aims the
                # diagnostic at the battery.
                p_kw = slack_advisory_kw
            else:
                p_kw = 0.0
            asset_cmds[asset_view.id] = AssetCommand(
                p_setpoint_kw=p_kw if asset_view.p_controllable else None,
                q_setpoint_kvar=0.0 if asset_view.q_controllable else None,
                on=False if asset_view.on_off_controllable else None,
            )

        # Every contactor closed: null sheds nothing. This needs no forecast and
        # no magnitude -- "serve it all" is what a closed contactor MEANS, where
        # the old continuous command had to be handed a kW figure to say the
        # same thing (and was handed the true demand to do it).
        load_cmds = {
            load_view.id: LoadCommand(on=True if load_view.on_off_controllable else None)
            for load_view in self._view.loads
        }
        return Commands(assets=asset_cmds, loads=load_cmds)
