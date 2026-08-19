"""A track 3 suite that asserts almost nothing. The floor.

REFERENCE SUBMISSION, never shipped in the pack. It exists so the track 3
pipeline has something to run: clean master, then each bug build, with a kill
rate at the end.

DELIBERATELY WEAK. Two of these tests pass no matter what the plant does, and
that is the point of having them -- a suite that passes everywhere kills nothing,
and a scorer that reports a high kill rate for this fixture is broken. Only
`test_energy_balance_holds` asserts anything real, so the expected result is a
LOW kill rate rather than zero: it is the check that the machinery discriminates
at all.

The one thing it must do is PASS ON CLEAN MASTER, which is the track 3 gate. A
suite that fails on correct code has asserted something untrue, and without that
gate the winning strategy is `assert False` everywhere.
"""

from __future__ import annotations

from pathlib import Path

from config.schema import load_site_config
from dispatch.null import NullDispatcher
from runner import simulate
from sim.scenario import from_csv

_CASE = "hack_public_2026_03"
# Found by walking up to `contracts.py` rather than counting directories, so this
# file works unchanged from `examples/` and from a slot, which sit at different
# depths. A hardcoded depth was wrong the first time it moved.
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "contracts.py").exists())


def test_the_plant_imports() -> None:
    """Passes on every build, including all fifteen. Kills nothing, on purpose."""
    import sim.plant

    assert sim.plant is not None


def test_arithmetic_still_works() -> None:
    """Passes on every build. The floor of the floor."""
    assert 2 + 2 == 4


def test_energy_balance_holds() -> None:
    """The one real assertion: a month runs and the bus closes every interval.

    `plant.step()` asserts the balance itself, so completing the run IS the
    check. Anything that makes the plant contradict its own books stops here.
    """
    case_dir = _ROOT / "studies" / "scenarios" / "synthetic" / _CASE
    config = load_site_config(
        str(_ROOT / "config" / "sites" / "hack_public.yaml"),
        str(case_dir / f"{_CASE}.carry_in.yaml"),
    )
    rows = list(from_csv(str(case_dir / f"{_CASE}.csv")))
    result = simulate(config, rows, NullDispatcher, 0)
    assert len(result.logs) == len(rows)
    assert result.breakdown.total_inr > 0.0


def test_soc_stays_inside_the_declared_band() -> None:
    """The one test here that can kill something.

    Checked against the band the config declares rather than against the plant's
    own opinion of it, so a model that moves its own limits cannot move the
    assertion with them. Its presence is what makes a zero kill rate on this
    fixture meaningful: without it, zero would be indistinguishable from a scorer
    that never detects anything.
    """
    case_dir = _ROOT / "studies" / "scenarios" / "synthetic" / _CASE
    config = load_site_config(
        str(_ROOT / "config" / "sites" / "hack_public.yaml"),
        str(case_dir / f"{_CASE}.carry_in.yaml"),
    )
    rows = list(from_csv(str(case_dir / f"{_CASE}.csv")))
    result = simulate(config, rows, NullDispatcher, 0)

    tank = next(a for a in config.fleet.assets if a.type == "bess").spec["params"]["tank"]
    for log in result.logs:
        soc_kwh = getattr(log.asset_states["bess_1"], "soc_kwh")
        assert tank["soc_min_kwh"] - 1e-6 <= soc_kwh <= tank["soc_max_kwh"] + 1e-6, (
            f"k={log.k}: soc {soc_kwh} outside " f"[{tank['soc_min_kwh']}, {tank['soc_max_kwh']}]"
        )
