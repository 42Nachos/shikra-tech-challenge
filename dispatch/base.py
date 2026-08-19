"""The one interface every dispatcher implements. Imports contracts only.

Since Stage E a dispatcher is CONSTRUCTED with the FleetView (decision 12) --
the one delivery of what fleet it is driving -- and then stepped with keyed
observations, returning keyed commands. The Protocol pins only step(); the
constructor convention is carried by the factory type the runner and the
golden harness use (Callable[[FleetView], Dispatcher]), so a dispatcher that
needs extra construction arguments wraps itself in a closure rather than
bending the interface.

Every declared channel of every member must be commanded, every interval:
Fleet.validate_commands() runs in the loop and absence is an error, not a
default (D4). The greedy/rule-based/MPC/oracle policies deleted at Stage E
were empty NotImplementedError skeletons; Stage H rebuilds them against the
FleetView, iterating by capability instead of hardcoding names.
"""

from __future__ import annotations

from typing import Protocol

from contracts import Commands, Observation


class Dispatcher(Protocol):
    def step(self, obs: Observation) -> Commands: ...
