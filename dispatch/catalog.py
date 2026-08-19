"""Name -> dispatcher, so a policy can be chosen on the command line.

The table lives HERE rather than in `runner.py` deliberately. runner.py is the
testbed side, built once and frozen; `dispatch/` is the side that churns. A new
policy is then a new file in this package plus one line below, and no edit to
the frozen half -- the same trade `sim/registry.py` makes for asset types, and
for the same reason.

Values are FACTORIES of the `Callable[[FleetView], Dispatcher]` shape the runner
and the golden harness already use (`dispatch/base.py`), so the classes
themselves fit as they are, and a policy needing extra construction arguments
registers a closure rather than bending the interface.

This is a CLI convenience, NOT a contract. `studies/regression/golden.py` goes on
naming its dispatcher classes directly: a golden records which policy it froze,
and a lookup table between the two would let a rename quietly repoint it.
"""

from __future__ import annotations

from collections.abc import Callable

from contracts import FleetView
from dispatch.base import Dispatcher
from dispatch.hold import held
from dispatch.null import NullDispatcher

DispatcherFactory = Callable[[FleetView], Dispatcher]

# What `python runner.py` runs when nobody says otherwise -- unchanged from when
# runner.py named the class directly, so the flag adds a choice without moving
# what the bare command does.
DEFAULT_DISPATCHER = "null"

_DISPATCHERS: dict[str, DispatcherFactory] = {
    "null": NullDispatcher,
}


def dispatcher(name: str) -> DispatcherFactory:
    """Look up a policy, or refuse to start.

    A typo must not fall through to a default: a run attributed to the wrong
    policy is worse than no run at all, and this project is judged by whether
    run 12 can be compared to run 37. `sim/registry.py` takes the same line for
    a typo in an asset `type:`.

    THE ZERO-ORDER HOLD IS APPLIED HERE, so it reaches every policy uniformly
    and no individual dispatcher knows it exists. A site declaring a control rate
    slower than its plant step gets its commands held between decisions
    (`dispatch/hold.py`); at the ordinary multiple of 1 the bare policy comes
    back untouched.
    """
    try:
        factory = _DISPATCHERS[name]
    except KeyError:
        raise ValueError(
            f"unknown dispatcher {name!r}. Known: {', '.join(known_dispatchers())}"
        ) from None
    return lambda view: held(factory(view), view)


def known_dispatchers() -> tuple[str, ...]:
    """Sorted, so the CLI's help text and its error messages read the same way
    regardless of registration order."""
    return tuple(sorted(_DISPATCHERS))
