"""A zero-order hold, so a policy can run slower than the plant.

WHAT IT IS FOR. A real EMS samples fast and re-optimises slowly: meters report
every few seconds, the optimiser runs every quarter hour or half hour, and the
setpoints it produced are held on the equipment in between. Until this existed
the two rates were one number in this repo, and nothing could say otherwise.

`SiteConfig.dispatch_interval_multiple` declares the ratio, and `FleetView`
carries it beside `dt_minutes`, so a policy sees both its OBSERVATION period and
its CONTROL period and neither is assumed.

THE INNER POLICY IS CALLED EVERY INTERVAL. Only its COMMANDS are held. That
distinction is the whole design:

  - a forecaster learns from every measurement, not one in N (`dispatch/
    forecast.py` records what arrives, at the rate it arrives);
  - the MPC's month-to-date peak sees every interval, so the demand charge it is
    shaving against is the one the ledger will actually bill;
  - a policy that wants to react to a measurement it took between decisions can,
    because it was given it.

A slower loop that also skipped the observations would be a different and worse
experiment: it would conflate "decides rarely" with "sees rarely".

NOTHING IN THE LOG SAYS WHICH INTERVALS WERE FRESH, and that is deliberate. The
run CSV records what was COMMANDED and what was DELIVERED; whether a command was
newly computed or held is a property of the policy, not of the plant, and a
column for it would be the policy's own bookkeeping leaking into the plant's
record. It stays derivable -- `k % multiple == 0` -- and `runner.manifest()`
carries the multiple, so a reader has what they need without a column.

WHAT A HELD COMMAND MEANS WHEN THE PLANT HAS MOVED. Nothing special. A battery
commanded to discharge 100 kW at the sample interval may exceed its envelope two
intervals later as SOC drops, and the plant clips it and records the violation
exactly as it would any other over-command. That is not an artefact to be
smoothed away: a slow controller genuinely drives into limits between samples,
and the violation is a measured output of the control period. Re-clipping the
hold to the live envelope would rescue a slow policy for free, which is the
plant-side-shedding failure ADR-0004 D6a exists to prevent, one channel over.

AT A MULTIPLE OF 1 NO WRAPPER IS BUILT AT ALL. `held()` returns the bare
dispatcher rather than a pass-through, so the ordinary path is the code that was
always there and no golden can move for a reason nobody chose.
"""

from __future__ import annotations

from contracts import Commands, FleetView, Observation
from dispatch.base import Dispatcher


class ZeroOrderHold:
    """Wraps a policy that decides every `multiple` intervals.

    Keyed on `obs.k`, which is a ROW COUNT -- the one thing `k` is for. A hold is
    a fact about position in the run, not about the clock: it is "how many plant
    steps since the last decision", and deriving that from a timestamp would
    reintroduce the k-as-clock confusion this pass removed everywhere else.
    `dispatch/playback.py` keys on `obs.k` for the same reason.
    """

    def __init__(self, inner: Dispatcher, multiple: int) -> None:
        if multiple < 2:
            raise ValueError(
                f"ZeroOrderHold needs a multiple of 2 or more, got {multiple}. "
                f"At 1 there is nothing to hold -- use the bare dispatcher."
            )
        self._inner = inner
        self._multiple = multiple
        self._held: Commands | None = None

    def step(self, obs: Observation) -> Commands:
        # ALWAYS, before deciding whether to keep the answer. The inner policy
        # accumulates its history, its month-to-date peak and its own state from
        # every interval; what the multiple gates is the ACTUATION.
        fresh = self._inner.step(obs)
        if self._held is None or obs.k % self._multiple == 0:
            self._held = fresh
        return self._held


def held(inner: Dispatcher, view: FleetView) -> Dispatcher:
    """`inner`, wrapped iff the site declares a control rate slower than the plant."""
    if view.dispatch_interval_multiple == 1:
        return inner
    return ZeroOrderHold(inner, view.dispatch_interval_multiple)
