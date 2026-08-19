"""One noisy reading, and the per-member seeding that makes it reproducible.

ADR-0012. Its own module because both ports need it and neither owns it. Not in
`sim/observe.py`, which imports both ports and so cannot be imported by either.
"""

from __future__ import annotations

import hashlib
import random


def add_noise(value: float, sigma_of_the_same_unit: float, rng: random.Random) -> float:
    """`value` as an instrument would report it: one gaussian draw added.

    ALWAYS DRAWS, including at sigma zero, so draw counts depend on the fleet's
    shape and never on the values in a noise block. The ADD is then guarded on
    the draw being nonzero, which is not redundant: `-0.0 + 0.0` is `0.0`, so an
    unguarded add would normalise signed zeros into a spurious golden diff.
    """
    drawn = rng.gauss(0.0, sigma_of_the_same_unit)
    return value if drawn == 0.0 else value + drawn


def member_rng(seed: int, member_id: str) -> random.Random:
    """This member's own stream. ADR-0012 D3.

    One `random.Random` per member, so editing one member's noise cannot
    reshuffle another's draws. sha256 rather than `hash()`, which is salted per
    process. Duplicates `synth/generate.py:_sub_seed`, since `sim/` may not
    import `synth/`.
    """
    digest = hashlib.sha256(f"{seed}:{member_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
