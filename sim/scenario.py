"""Scenario source, and the exogenous truth it produces. Imports contracts only.

`from_csv` reads a processed CSV built by tools/build_scenario.py, which is
where resampling, gap-filling and provenance flagging happen. The two
generators at the bottom are synthetic fixtures: `synthetic_day` so `python
runner.py` works with no data at all, and `stress_days` to exercise the paths
the real scenario never reaches. PV reconstruction (GHI->POA transposition)
still belongs here and is not built. See Module 3.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from contracts import (
    MINUTES_PER_HOUR,
    Quality,
    intervals_per_day,
    minutes_of_day,
    parse_timestamp,
    step_local,
)

# Suffix that marks an availability column as belonging to one asset:
# `bess_1.available`. One constant, because the producer (build_scenario), the
# reader (from_csv) and the requirement (Asset.required_scenario_fields) must
# agree on it exactly, and three string literals would eventually not.
AVAILABILITY_SUFFIX = ".available"


def availability_field(member_id: str) -> str:
    """The scenario column name carrying `member_id`'s availability."""
    return f"{member_id}{AVAILABILITY_SUFFIX}"


@dataclass(frozen=True)
class ScenarioRow:
    """One row of exogenous TRUTH, fed into the plant.

    Here rather than in contracts.py: `load_p_kw` and `poa_irradiance_w_m2` are
    the exact quantities a dispatcher is meant to be uncertain about.
    """

    k: int
    timestamp: str  # tz-aware ISO string (tighten to datetime later)
    load_p_kw: float
    poa_irradiance_w_m2: float  # plane-of-array; GHI->POA transposition lives in sim/scenario
    # AMBIENT air temperature, not cell temperature (ADR-0011 D10): ambient is
    # what a site measures, cell temperature is what a MODEL computes from it.
    # Site-level, like irradiance -- one air mass over one site.
    temp_air_c: float
    quality: Quality
    # PER-ASSET availability, keyed by asset ID (ADR-0004 D11). One battery can
    # be down while another runs, which a type-level flag cannot say.
    #
    # NO DEFAULT: absence is caught by
    # AssembledFleet.validate_scenario_row(), never filled. Every field above is
    # site-level.
    available: Mapping[str, bool]


@dataclass(frozen=True, kw_only=True)
class Interval:
    """One interval, as anything pricing it needs to see: the row, the parsed
    clock, and the plant step.

    `ts` is parsed ONCE by the ledger rather than per cost model.
    """

    row: ScenarioRow
    ts: datetime
    dt_minutes: int


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("true", "1", "yes")


def _parse_quality(row: dict[str, str], path: str | Path) -> Quality:
    """Provenance is never guessed: a CSV without a quality column is rebuilt,
    never defaulted to MEASURED."""
    raw = row.get("quality")
    if raw is None:
        raise ValueError(
            f"{path}: no 'quality' column. Rebuild it with tools/build_scenario.py "
            f"(ADR-0003 D1 item 4); provenance is not defaulted."
        )
    try:
        return Quality(raw.strip().lower())
    except ValueError:
        valid = ", ".join(q.value for q in Quality)
        raise ValueError(f"{path}: unknown quality {raw!r} (expected one of: {valid})") from None


def _parse_temp_air_c(row: dict[str, str], path: str | Path) -> float:
    """Ambient temperature is never guessed, like `quality`. The tempting
    default is 25 degC, the pinned value this column exists to retire."""
    raw = row.get("temp_air_c")
    if raw is None:
        raise ValueError(
            f"{path}: no 'temp_air_c' column. Rebuild it with tools/build_scenario.py "
            f"(ADR-0011 D10); ambient temperature is not defaulted, least of all to 25."
        )
    return float(raw)


def from_csv(path: str | Path) -> Iterator[ScenarioRow]:
    """Stream ScenarioRows from a processed scenario CSV.

    Produced by tools/build_scenario.py, already resampled, tz-aware and named to
    match ScenarioRow. All tariff and physics processing happens upstream.
    """
    with Path(path).open(newline="") as f:
        for r in csv.DictReader(f):
            # Availability is read by PATTERN: whichever `<id>.available`
            # columns the file carries become the mapping. This module sits below
            # sim/asset.py and cannot know the fleet. Sorted, so `scenario_hash`
            # does not depend on column order.
            available = {
                col[: -len(AVAILABILITY_SUFFIX)]: _parse_bool(value)
                for col, value in sorted(r.items())
                if col.endswith(AVAILABILITY_SUFFIX)
            }
            yield ScenarioRow(
                k=int(r["k"]),
                timestamp=r["timestamp"],
                load_p_kw=float(r["load_p_kw"]),
                poa_irradiance_w_m2=float(r["poa_irradiance_w_m2"]),
                temp_air_c=_parse_temp_air_c(r, path),
                quality=_parse_quality(r, path),
                available=available,
            )


# The stress windows, chosen against the ScriptedStressDispatcher's schedule so
# that every island branch fires. THE OVERLAPS ARE THE POINT: under a slack
# priority order a grid outage alone does not reach D5's SECONDARY tier, since
# the battery takes the bus. Every slack candidate must be out at once.
#
#   k 52..56   grid + bess down at MIDDAY, genset commanded off -> island_pv_curtailed
#   k 172..176 grid + bess + dg all down in the evening         -> no_slack_unserved
#
# `island_bess_backed_off` is NOT reachable here: under this priority an
# available battery during an outage IS the bus. tests/test_plant.py covers it.
#
# These generators take no config -- this module sits below sim/fleet.py -- so a
# fixture declares the step it is written on. `synth/` is the generator that
# takes a step from config; these are fixtures.
_FIXTURE_DT_MINUTES = 15
_FIXTURE_INTERVALS_PER_DAY = intervals_per_day(_FIXTURE_DT_MINUTES)

_SYNTHETIC_DAY_START = "2025-07-14T00:00:00+05:30"
_STRESS_START = "2025-07-14T00:00:00+05:30"

# (day, start_hour, stop_hour), half-open on the hour. CLOCK positions rather
# than k ranges, since k is a row count and says nothing about the time.
_STRESS_GRID_DOWN = ((0, 12.0, 14.0), (1, 2.0, 3.0), (1, 18.0, 20.0))
_STRESS_DG_DOWN = ((1, 19.0, 21.0),)
_STRESS_BESS_DOWN = ((0, 13.0, 14.0), (1, 19.0, 20.0))


def _in_windows(day: int, hour: float, windows: tuple[tuple[int, float, float], ...]) -> bool:
    return any(d == day and start <= hour < stop for d, start, stop in windows)


# Diurnal ambient temperature for the two fixtures below: a shape, not a
# measurement. Peaks mid-afternoon rather than at solar noon.
_AMBIENT_MEAN_C = 28.0
_AMBIENT_SWING_C = 6.0  # so 22-34 degC, an ordinary Chennai day
_AMBIENT_PEAK_HOUR = 15.0


def _ambient_c(hour: float) -> float:
    """Ambient air temperature at `hour` (0-24) of a synthetic day."""
    phase = math.pi * (hour - _AMBIENT_PEAK_HOUR) / 12.0
    return _AMBIENT_MEAN_C + _AMBIENT_SWING_C * math.cos(phase)


def stress_days() -> Iterator[ScenarioRow]:
    """Two synthetic days exercising what the null runs never touch: genset
    dispatch, BESS cycling, curtailment, islanding, involuntary shed.

    Same shapes as synthetic_day, two days long, with availability windows per
    the module-level tables. tests/test_stress_scenario.py asserts the case
    still exercises every intended path.
    """
    start = parse_timestamp(_STRESS_START)
    for k in range(2 * _FIXTURE_INTERVALS_PER_DAY):
        # The timestamp first, the clock from the timestamp. k counts rows.
        stamp = step_local(_STRESS_START, k, _FIXTURE_DT_MINUTES)
        timestamp = stamp.isoformat()
        hour = minutes_of_day(timestamp) / MINUTES_PER_HOUR
        day = (stamp.date() - start.date()).days
        poa = max(0.0, 1000.0 * math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        load = 150.0 + 80.0 * math.sin(math.pi * (hour - 8) / 10)
        yield ScenarioRow(
            k=k,
            timestamp=timestamp,
            load_p_kw=round(load, 2),
            poa_irradiance_w_m2=round(poa, 2),
            temp_air_c=round(_ambient_c(hour), 2),
            quality=Quality.SYNTHETIC,
            # Keyed by site_a's asset IDS, since the windows above were chosen
            # against exactly its genset, battery and grid. Unlike
            # `synthetic_day` this does not take the ids: a list of ids does not
            # say which member plays which ROLE, and resolving roles needs the
            # registry, which sits above this module.
            available={
                "bess_1": not _in_windows(day, hour, _STRESS_BESS_DOWN),
                "dg_1": not _in_windows(day, hour, _STRESS_DG_DOWN),
                "grid_1": not _in_windows(day, hour, _STRESS_GRID_DOWN),
                "pv_1": True,
            },
        )


def synthetic_day(
    asset_ids: Sequence[str], n: int = _FIXTURE_INTERVALS_PER_DAY
) -> Iterator[ScenarioRow]:
    """One day at the fixture step, with every asset available.

    `asset_ids` comes from the fleet, since this is what `python runner.py` runs
    with no arguments. NO DEFAULT: absence must be caught, not filled.

    THE MAPPING IS EMITTED SORTED, the same invariant `from_csv` applies when
    reading, so `scenario_hash` does not depend on declaration order.
    """
    available = {asset_id: True for asset_id in sorted(asset_ids)}
    for k in range(n):
        # The timestamp first, the clock from the timestamp. k counts rows.
        timestamp = step_local(_SYNTHETIC_DAY_START, k, _FIXTURE_DT_MINUTES).isoformat()
        hour = minutes_of_day(timestamp) / MINUTES_PER_HOUR
        # Synthetic clear-sky-shaped POA irradiance, peaking ~1000 W/m2 at noon.
        poa = max(0.0, 1000.0 * math.sin(math.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        load = 150.0 + 80.0 * math.sin(math.pi * (hour - 8) / 10)
        yield ScenarioRow(
            k=k,
            timestamp=timestamp,
            load_p_kw=round(load, 2),
            poa_irradiance_w_m2=round(poa, 2),
            temp_air_c=round(_ambient_c(hour), 2),
            quality=Quality.SYNTHETIC,  # generated here; nothing about it is measured
            # Built once above and shared: the same flags on every interval.
            available=available,
        )
