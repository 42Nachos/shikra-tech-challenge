"""Discharge the battery whenever it can. The dumbest policy that is not `null`.

REFERENCE SUBMISSION, never shipped in the pack. It exists so the scorer has
something to load by path, run under the audit hook and rank. It is not good --
it empties the store early and has nothing left for the evening peak.
"""

from __future__ import annotations

from contracts import AssetCommand, Commands, FleetView, LoadCommand, Observation


class ReferenceDispatcher:
    """Battery flat out, everything else as `null` would leave it."""

    def __init__(self, view: FleetView) -> None:
        self._view = view

    def step(self, obs: Observation) -> Commands:
        available = {v.id: obs.assets[v.id].available for v in self._view.assets}
        slack_id = self._view.slack_id(available)
        prev = obs.assets[slack_id].measured_prev
        slack_advisory_kw = prev.p_net_kw if prev is not None else 0.0

        assets: dict[str, AssetCommand] = {}
        for view in self._view.assets:
            if view.resource_limited:
                p_kw: float | None = view.ratings.p_max_kw
            elif view.id == slack_id:
                p_kw = slack_advisory_kw
            elif view.ratings.energy_capacity_kwh is not None:
                # The one decision this policy makes: storage runs flat out.
                # The plant clips it against the envelope and records that.
                p_kw = view.ratings.p_max_kw
            else:
                p_kw = 0.0
            assets[view.id] = AssetCommand(
                p_setpoint_kw=p_kw if view.p_controllable else None,
                q_setpoint_kvar=0.0 if view.q_controllable else None,
                on=False if view.on_off_controllable else None,
            )
        loads = {
            v.id: LoadCommand(on=True if v.on_off_controllable else None) for v in self._view.loads
        }
        return Commands(assets=assets, loads=loads)


# The entry point score_track1.py loads by path. A class taking the FleetView is
# already a `Callable[[FleetView], Dispatcher]`, which is the factory shape the
# catalog's own values have.
DISPATCHER = ReferenceDispatcher
