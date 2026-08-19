"""Time-of-Day rate evaluation, and the declared schedule it evaluates.

The single source of truth for turning a local clock time into an energy rate.
Both the grid's `energy` term and the scenario builder call through here, so the
ToU schedule is never re-typed as literals in two places.

The schedule is declared as params on the grid asset's own `energy` cost term
(ADR-0004 D7); `TouRateSchedule` is the model those params validate against.
`config/` declares no tariff schema, since it cannot import `sim/` and a second
definition would drift.

Testbed-side: dispatch/'s 5.2 reads the same declared params and implements its
own rating, and cannot import this module at all.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from contracts import MINUTES_PER_DAY, CostTerm, clock_to_minutes


class TouWindow(BaseModel):
    """One Time-of-Day energy band. `multiplier` scales the base energy rate
    (TNERC T.O. 6/2025: peak +25% -> 1.25, night -5% -> 0.95, normal 1.0).

    Half-open [start, end) in LOCAL CLOCK TIME, `"HH:MM"`, and the set of windows
    must partition the whole day (validated on TouRateSchedule).

    Minutes rather than whole hours, so the tariff's resolution is not coarser
    than the meter's. Nothing in TNERC needs an 06:30 band today.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    start: str
    end: str
    multiplier: float = Field(gt=0)

    @property
    def start_minute(self) -> int:
        return clock_to_minutes(self.start)

    @property
    def end_minute(self) -> int:
        return clock_to_minutes(self.end)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        # Parses both, so a malformed clock time is refused here rather than at
        # the first interval that falls in this band.
        if self.start_minute >= self.end_minute:
            raise ValueError(f"tou window {self.label}: start must be before end")
        return self


class TouRateSchedule(BaseModel):
    """A base energy rate and the ToD bands that scale it.

    The `energy` cost term's params model (ADR-0004 D7), and the ONE declaration
    of a member's rate schedule -- `export` and `pf_penalty` read it rather than
    declaring copies.

    The windows must PARTITION the day, both directions checked: an overlap makes
    the rate depend on declaration order, a gap makes some hour unpriceable.
    """

    model_config = ConfigDict(frozen=True)

    base_rate_inr_per_kwh: float = Field(gt=0)
    tou_windows: tuple[TouWindow, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_windows_partition_day(self) -> Self:
        # Per MINUTE, not per hour: a half-hour gap would otherwise load
        # cleanly and raise at whichever interval first fell into it.
        covered = [False] * MINUTES_PER_DAY
        for w in self.tou_windows:
            for m in range(w.start_minute, w.end_minute):
                if covered[m]:
                    raise ValueError(f"tou windows overlap at {m // 60:02d}:{m % 60:02d}")
                covered[m] = True
        if not all(covered):
            first = covered.index(False)
            raise ValueError(
                f"tou windows leave the day uncovered from {first // 60:02d}:{first % 60:02d}"
            )
        return self

    def multiplier(self, minute_of_day: int) -> float:
        """Multiplier on the base rate for the ToD band containing this minute.

        `minute_of_day` is `contracts.minutes_of_day(row.timestamp)` -- the
        interval's START. A band boundary inside an interval takes effect at the
        next one, which is what a meter integrating over the interval does.
        """
        for w in self.tou_windows:
            if w.start_minute <= minute_of_day < w.end_minute:
                return w.multiplier
        raise ValueError(f"no ToU window covers minute {minute_of_day} of the day")

    def rate_inr_per_kwh(self, minute_of_day: int) -> float:
        """Applicable import energy rate (INR/kWh) at this minute of the day."""
        return self.base_rate_inr_per_kwh * self.multiplier(minute_of_day)


def import_rate_schedule(cost_terms: tuple[CostTerm, ...]) -> TouRateSchedule | None:
    """The rate schedule declared by a member's `energy` term, if it has one.

    Returns None rather than raising: the caller knows whether a fleet with no
    energy term is a problem for it.
    """
    for term in cost_terms:
        if term.kind == "energy":
            return TouRateSchedule.model_validate(dict(term.params))
    return None
