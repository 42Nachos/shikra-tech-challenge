"""The loop. Deterministic; the only module that touches both sides.

Run it:  python runner.py
Prints the CostBreakdown, then asks what to file the run under and writes a log
to runs/. This satisfies the Stage 0 gate: the loop runs a scenario end to end
and writes a log.

The log records what the dispatcher COMMANDED beside what the plant DELIVERED,
which is what tools/report.py draws.

`simulate()` is the loop itself, with no I/O. `run()` is the CLI wrapper around
it. The split exists so the regression harness in studies/regression drives
exactly the same code path the CLI does -- a golden that came from a different
loop would prove nothing.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from config.schema import SiteConfig, load_site_config
from contracts import Commands, CostBreakdown, FleetView, Quality
from dispatch import catalog
from dispatch.base import Dispatcher
from sim import scenario
from sim.fleet import assemble, build_fleet
from sim.ledger import Ledger
from sim.observe import member_rngs, observe
from sim.plant import FleetDelivered, Plant, StepLog
from sim.scenario import ScenarioRow

DEFAULT_SEED = 0
# REPO-RELATIVE, and deliberately still a plain string: studies/regression
# writes this into every golden manifest as the config's identity, so an
# absolute path here would pin three goldens to one machine. Resolution against
# the repo happens at the point of use, below.
DEFAULT_CONFIG_PATH = "config/sites/site_a.yaml"

# This file's own directory, which IS the repo root. Everything the program
# owns -- its default config, its output directory -- is resolved against this
# rather than against the process's cwd, so `python /path/to/runner.py` works
# from anywhere instead of only from the repo root. The old behaviour failed
# loudly on the config (FileNotFoundError) and quietly on the output: a run log
# would be written into a fresh `runs/` beside wherever you happened to be
# standing. studies/regression/golden.py has always done it this way; this is
# that pattern, applied to the CLI path it never covered.
#
# NOT applied to `--scenario`: a path the user types resolves against the
# user's cwd, which is ordinary CLI behaviour and the least surprising thing.
# There is deliberately no fallback that retries a failed path against the repo
# root -- a silent second guess is how a run comes to read a file nobody meant.
_REPO_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced. No I/O, no formatting."""

    logs: tuple[StepLog, ...]
    # What the dispatcher ASKED for, interval by interval, parallel to `logs`.
    # The step log carries only what the plant DELIVERED, and the two differ
    # wherever a command was clipped, overridden on the slack, or refused --
    # which is the whole diagnostic. Kept here rather than on StepLog because a
    # command is the policy's output, not the plant's truth, and sim/ has no
    # business storing it.
    commands: tuple[Commands, ...]
    breakdown: CostBreakdown
    seed: int


def _cell(value: Any, decimals: int | None) -> Any:
    """One CSV cell. `decimals=None` writes floats via repr(), which round-trips
    a double exactly -- the golden files use that; the human-readable run log
    rounds to 2."""
    if value is None:
        # An empty cell, and empty is what None must ALREADY round-trip to:
        # csv writes None as "", so a None left in the in-memory row compares
        # as "None" against a golden's "" and can never match. Every not-
        # commanded channel is None (contracts.py), so without this a golden
        # carrying command columns is unblessable -- it fails the moment it is
        # read back. Written here rather than fixed in the comparison because
        # this function is what decides a value's CSV representation.
        return ""
    if isinstance(value, tuple):
        return "|".join(str(v) for v in value)
    if isinstance(value, float):
        return repr(value) if decimals is None else round(value, decimals)
    return value


def _member_columns(row: dict[str, Any], member_id: str, record: Any, decimals: int | None) -> None:
    """Flatten one member's dataclass record into `<member_id>.<field>` columns.

    Keyed rather than flat since Stage F. The old log named one asset of each
    type (`bess_kw`, `soc_kwh`), which is the fleet welded into the log format:
    a second battery had nowhere to be written, so the plant refused to run one.
    """
    for f in fields(record):
        column = f"{member_id}.{f.name}"
        assert column not in row, f"log column {column!r} written twice"
        row[column] = _cell(getattr(record, f.name), decimals)


def flatten(log: StepLog, cmd: Commands, decimals: int | None = 2) -> dict[str, Any]:
    """One CSV row per interval: what was COMMANDED beside what was DELIVERED.

    Column order is DECLARATION ORDER (D10): the log's mappings are built by
    iterating the fleet, and this walks them in that order, so the header is a
    stable function of the config rather than of dict insertion luck. Within a
    member it is command, then delivered, then state, so each asset reads
    request-then-result across the row.

    A stateless asset contributes no state columns at all -- PvState and
    GridState are empty dataclasses -- so the header grows only when an asset
    actually starts remembering something.

    The command columns are keyed `<member_id>.cmd.<field>`. The infix is not
    cosmetic: `dg_1.on` is already taken by GensetState (what the genset IS),
    and `dg_1.cmd.on` is what it was ASKED to be. Those two disagreeing is a
    measurement -- a refused start, a commit below minimum stable load -- so
    the log has to be able to say both.

    AN EMPTY COMMAND CELL MEANS NOT COMMANDED, and is not a zero. All three
    AssetCommand channels are None when the asset does not declare them, which
    contracts.py distinguishes from a commanded zero for good reason: a zero P
    setpoint on PV means curtail everything. `_cell` passes None through and
    csv writes it empty; anything reading this file back must not coerce that
    to 0.0.
    """
    row: dict[str, Any] = {"k": log.k, "timestamp": log.timestamp}
    for member_id, delivered in log.assets.items():
        _member_columns(row, f"{member_id}.cmd", cmd.assets[member_id], decimals)
        _member_columns(row, member_id, delivered, decimals)
        _member_columns(row, member_id, log.asset_states[member_id], decimals)
    for load_id, load_delivered in log.loads.items():
        _member_columns(row, f"{load_id}.cmd", cmd.loads[load_id], decimals)
        _member_columns(row, load_id, load_delivered, decimals)
    row["slack_id"] = log.slack_id
    row["slack_advisory_kw"] = _cell(log.slack_advisory_kw, decimals)
    row["slack_delta_kw"] = _cell(log.slack_delta_kw, decimals)
    row["violations"] = _cell(log.violations, decimals)
    row["quality"] = log.quality
    return row


def simulate(
    config: SiteConfig,
    rows: Iterable[ScenarioRow],
    make_dispatcher: Callable[[FleetView], Dispatcher],
    seed: int = DEFAULT_SEED,
) -> RunResult:
    """Run one scenario end to end. Pure: no files read or written.

    ADR-0004 Stage E: the loop drives the assembled fleet. Construction-time
    validation completes up front (build_fleet types the specs, assemble
    constructs and checks the slack, initial_state validates carry-in), the
    dispatcher is constructed with its one FleetView (decision 12), and
    Fleet.validate_commands() runs EVERY interval -- the D9 trade, wired: an
    absent or extra command channel stops the run rather than defaulting.

    Seeded because CLAUDE.md requires same seed + same config to give a
    bit-identical log. Since ADR-0012 the seed is actually consumed: it derives
    one independent stream PER FLEET MEMBER, which each member's observe() draws
    its sensor noise from. One shared stream would have made every member's
    numbers depend on every other member's noise block.
    """
    fleet = build_fleet(config.fleet)
    assembled = assemble(fleet, config)
    plant = Plant.build(assembled, config.slack_priority)
    state = plant.initial_state(fleet)
    # Construction-time validation completes here: the ledger builds one term
    # per declared (member, kind), validating each term's params and refusing a
    # term the member cannot support. An invalid tariff stops the run now, not
    # after 5,787 intervals. The one-genset restriction this used to carry is
    # gone with the flat fuel line -- fuel is priced per declaring asset.
    ledger = Ledger(assembled, config)

    # Built once, here, and advanced across intervals -- see member_rngs().
    rngs = member_rngs(seed, assembled.assets, assembled.loads)

    dispatcher = make_dispatcher(assembled.view)
    logs: list[StepLog] = []
    commands: list[Commands] = []
    delivered_prev: FleetDelivered | None = None  # no previous interval at k=0

    # ADR-0004 D1, Stage G: the fleet's declared exogenous needs are checked
    # against what the scenario actually supplies, once, before interval 0. The
    # first row is peeked and pushed back rather than the whole scenario being
    # materialised -- `rows` may be a generator over a 5,787-line CSV, and
    # simulate() promises to stream it.
    stream = iter(rows)
    first = next(stream, None)
    if first is not None:
        assembled.validate_scenario_row(first)
        stream = itertools.chain([first], stream)

    for row in stream:
        obs = observe(assembled.assets, assembled.loads, state, row, delivered_prev, rngs)
        cmd = dispatcher.step(obs)
        fleet.validate_commands(cmd.assets, cmd.loads)
        state, delivered, log = plant.step(state, cmd, row)
        ledger.update(delivered, row)  # the keyed truth, per member (Stage F)
        logs.append(log)
        commands.append(cmd)
        delivered_prev = delivered

    ledger.close_billing_period()
    return RunResult(
        logs=tuple(logs),
        commands=tuple(commands),
        breakdown=ledger.finalize(),
        seed=seed,
    )


def scenario_rows(scenario_path: str | None, config: SiteConfig) -> Iterator[ScenarioRow]:
    """The rows for this run: a CSV if one was named, else the built-in day.

    `config` is here only for the built-in generator, which needs the fleet's
    asset ids to fill availability -- a scenario is written FOR a fleet, and the
    generated one now says so instead of naming site_a's members. A CSV supplies
    its own `<id>.available` columns and the fleet checks them at interval 0.
    """
    if scenario_path is None:
        return scenario.synthetic_day([a.id for a in config.fleet.assets])
    return scenario.from_csv(scenario_path)


def _print_quality(logs: tuple[StepLog, ...]) -> None:
    """ADR-0003 D1 item 4: the count of non-MEASURED rows is REPORTED, never
    silently absorbed. A cost computed over imputed or synthetic inputs is not
    the same evidence as one computed over meter readings."""
    counts: Counter[Quality] = Counter(log.quality for log in logs)
    non_measured = len(logs) - counts[Quality.MEASURED]
    print("Scenario data quality:")
    for q in Quality:
        if counts[q]:
            print(f"  {q.value:<10} {counts[q]:>7} ({100.0 * counts[q] / len(logs):5.1f}%)")
    if non_measured:
        print(f"  WARNING: {non_measured} of {len(logs)} intervals are not measured data.")


def _print_breakdown(breakdown: CostBreakdown) -> None:
    # Never a bare scalar -- always the itemised CostBreakdown (CLAUDE.md).
    # One line per (member, term) since Stage F, in fleet declaration order, so
    # the printout says WHICH asset the money went to and not merely what for.
    print("CostBreakdown (INR):")
    width = max((len(k) for k in breakdown.lines), default=0)
    for key, amount in breakdown.lines.items():
        print(f"  {key:<{width}}  {amount:14,.2f}")
    print(f"  {'TOTAL':<{width}}  {breakdown.total_inr:14,.2f}")


_NAME_RE = re.compile(r"\A[A-Za-z0-9._-]+\Z")

# What a run directory holds. Fixed names, because the DIRECTORY carries the
# run's identity -- `runs/tuned_v2/log.csv` says everything `tuned_v2.csv` did
# and leaves room for the manifest and the page beside it. tools/report.py
# repeats these two rather than importing this module: it reads CSVs and
# imports nothing internal, which is a property worth two string constants.
LOG_NAME = "log.csv"
MANIFEST_NAME = "manifest.json"


def _ask_run_name() -> str | None:
    """The name to file this run under, or None to write only `latest`.

    Returns None on a blank line, which is the common case: you ran the loop to
    look at it, not to keep it.

    Rejects rather than rewrites a name with a separator or a `..` in it. A
    silently sanitised name is a file written somewhere the user did not ask
    for, and `runs/` is the only directory this program owns.
    """
    while True:
        name = input("Run name (blank = runs/latest/ only): ").strip()
        if not name:
            return None
        if _NAME_RE.match(name) and name not in {".", ".."}:
            return name
        print("  Use letters, digits, dot, dash or underscore only.")


def _scenario_ref(scenario_path: str | None) -> str | None:
    """The scenario path written into the manifest, REPO-ROOT-RELATIVE.

    Not the string as typed. `--scenario` resolves against the user's cwd, so
    `python ../runner.py --scenario scenarios/foo.csv` from `studies/` records
    a path that means nothing anywhere else -- and tools/report.py, which
    resolves against the repo root, then reports the scenario as missing and
    silently draws no inputs.

    Absolute only when the file lives outside the repo, since an absolute path
    for a repo file would pin the manifest to one machine.
    """
    if scenario_path is None:
        return None
    resolved = Path(scenario_path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _config_ref(config_path: str | None) -> str:
    """The config path written into the manifest, repo-root-relative.

    Same rule and same reason as `_scenario_ref`, one paragraph shorter: a
    manifest recording a path relative to whichever directory somebody happened
    to be standing in identifies the run on exactly one machine.

    `None` means the default, and is recorded as the default's own string rather
    than resolved -- `DEFAULT_CONFIG_PATH` is already repo-root-relative and
    goldens embed it (see its definition), so round-tripping it through
    `Path.resolve()` would be a chance to change it for nothing.
    """
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    resolved = Path(config_path).resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _model_block(config: SiteConfig) -> dict[str, dict[str, Any]]:
    """Each member's declared BELIEF beside its declared SPEC. ADR-0009 D5.

    Side by side and UNMATCHED: no difference is computed and no parameter is
    paired with another. A belief need not have a counterpart in the spec at all
    -- PV's `performance_ratio` has none, because truth applies no soiling term --
    so a mismatch vector would have to invent one, and inventing it is how a
    belief with nothing to be wrong about comes to look rigorous.

    What this buys is that "did this run actually contain model error?" is
    answerable from the artifact, after the fact, instead of by finding the config
    that produced it and reading it by hand. Two runs differing only in belief
    must not produce indistinguishable manifests.

    This is a departure from ADR-0007 D3, which keeps the run manifest to what a
    report needs to RENDER a run and leaves `--carry-in` out on those grounds.
    The belief is not an input to be drawn; it is a property of the experiment.

    Declaration order (D10), and loads are included with both blocks empty --
    omitting a member with nothing to say would be the same silent default D2
    refuses in the config.
    """
    return {
        entry.id: {"believed": dict(entry.believed), "spec": dict(entry.spec)}
        for entry in config.fleet.assets
    } | {entry.id: {"believed": dict(entry.believed), "spec": {}} for entry in config.fleet.loads}


def manifest(
    result: RunResult,
    name: str,
    scenario_path: str | None,
    dispatcher_name: str,
    config: SiteConfig,
    config_path: str | None = None,
) -> dict[str, Any]:
    """What tools/report.py needs that the log itself cannot carry: which
    scenario supplied the inputs, and the itemised bill.

    Public since the ADR-0008 oracle, which writes a run directory of its own
    and must write the SAME manifest a policy run does -- tools/report.py reads
    one shape, and a second producer with its own idea of the format is how the
    two drift. A rename, nothing more.

    `scenario` is repo-root-relative (see _scenario_ref) and is null for the
    built-in synthetic day -- which has no file, so a report over such a run
    simply has no exogenous inputs to draw. Deliberately carries no config or
    scenario hash: this identifies a run well enough to render it, and run
    IDENTITY is studies/regression's job, where the hashes live.

    `--carry-in` is NOT recorded, and a carry-in overlay changes the run (a
    different starting SOC is a different run, ADR-0005 D7). It is not needed to
    draw one, so it is left out rather than half-tracked; if a manifest ever
    has to identify a run, that and the hashes arrive together.

    `config_path` defaults to None meaning DEFAULT_CONFIG_PATH, which is what
    this field was hardcoded to before the site config became selectable. It is
    the one input here that decides what the FLEET is, so a run against a second
    site must not be indistinguishable from a run against the first -- the same
    argument `model` above makes for beliefs.
    """
    counts: Counter[Quality] = Counter(log.quality for log in result.logs)
    return {
        "name": name,
        "dispatcher": dispatcher_name,
        "seed": result.seed,
        "scenario": _scenario_ref(scenario_path),
        "config_path": _config_ref(config_path),
        "intervals": len(result.logs),
        # The two rates, so a reader can tell how often the policy actually
        # DECIDED as against how often it was asked. Above a multiple of 1 the
        # commanded columns repeat within each group of `multiple` intervals,
        # and nothing in the log marks which one was fresh -- deliberately, since
        # that is the policy's bookkeeping and not the plant's record. It stays
        # derivable from these two numbers: fresh iff `k % multiple == 0`.
        "dt_minutes": config.dt_minutes,
        "dispatch_interval_multiple": config.dispatch_interval_multiple,
        "quality_counts": {q.value: counts[q] for q in Quality if counts[q]},
        # What the dispatcher was assumed to believe, beside what the plant
        # actually is (ADR-0009 D5). See _model_block.
        "model": _model_block(config),
        # One line per (member, term), in fleet declaration order. Never a bare
        # scalar; `total_inr` is the headline, not the output (CLAUDE.md).
        "cost_breakdown_inr": dict(result.breakdown.lines),
        "total_inr": result.breakdown.total_inr,
    }


def run(
    scenario_path: str | None = None,
    seed: int = DEFAULT_SEED,
    carry_in_path: str | None = None,
    dispatcher_name: str = catalog.DEFAULT_DISPATCHER,
    config_path: str | None = None,
) -> None:
    # `--config` names the SITE. It defaults to None meaning DEFAULT_CONFIG_PATH,
    # so every existing invocation and every golden is unmoved -- but a second
    # site is now expressible, which it was not while the path was a module
    # constant read directly here. Running a scenario built for one fleet against
    # another site's config produced a plausible answer rather than an error, and
    # the plant would not have complained: `validate_scenario_row` only checks
    # that the columns the fleet NEEDS are present, so a 1.4 kW off-grid trace
    # runs perfectly happily against a 300 kW array.
    #
    # Resolved against the USER's cwd, like `--scenario` and `--carry-in`; the
    # default alone keeps its repo-root anchor, because it is a repo file named
    # by a constant rather than a path somebody typed.
    #
    # `--carry-in` is a generated scenario's companion overlay (ADR-0005 D7):
    # the site config stays the base and authoritative for every parameter, and
    # only initial state is replaced. It moves `config_hash`, which is correct --
    # a different starting SOC is a different run.
    #
    # Resolved against the USER's cwd, like `--scenario` and for the same
    # reason: a path someone types is theirs, and a silent retry against the
    # repo root is how a run comes to read a file nobody meant.
    site_yaml = _REPO_ROOT / DEFAULT_CONFIG_PATH if config_path is None else Path(config_path)
    config = load_site_config(site_yaml, carry_in_path)
    # Resolved BEFORE the scenario is opened, so an unknown name costs nothing
    # and reports itself immediately. argparse's `choices` already catches this
    # on the CLI path; the lookup is what guards `run()` called from Python.
    make_dispatcher = catalog.dispatcher(dispatcher_name)
    result = simulate(config, scenario_rows(scenario_path, config), make_dispatcher, seed)

    rows = [flatten(log, cmd) for log, cmd in zip(result.logs, result.commands, strict=True)]
    print(f"Ran {len(rows)} intervals.")
    _print_quality(result.logs)
    _print_breakdown(result.breakdown)

    # NOTHING IS WRITTEN unless a human is at the keyboard to name the run.
    # `pytest`, a piped invocation and `runner.run()` called from Python all
    # land here and leave `runs/` untouched, so a test suite never litters it.
    # The tests drive simulate() directly today; this is what keeps that from
    # being a matter of habit.
    if not sys.stdin.isatty():
        print("Not a terminal -- nothing written. Run it interactively to save.")
        return

    name = _ask_run_name()
    runs_dir = _REPO_ROOT / "runs"

    # ONE DIRECTORY PER RUN, holding the log, the manifest, and -- once
    # tools/report.py has been over it -- the page, so everything about a run
    # travels together and can be copied or deleted as one thing.
    #
    # `latest` is always written, and a named run is written TWICE, as a real
    # copy rather than a link. Two small directories beat a symlink that reads
    # as a directory until the day something follows it somewhere unexpected.
    targets = ["latest"] if name is None else ["latest", name]
    run_manifest = manifest(
        result, name or "latest", scenario_path, dispatcher_name, config, config_path
    )
    for stem in targets:
        out_dir = runs_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / LOG_NAME).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out_dir / MANIFEST_NAME).write_text(json.dumps(run_manifest, indent=2) + "\n")
        print(f"  -> {out_dir}{os.sep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EMS testbed loop.")
    parser.add_argument(
        "--scenario",
        default=None,
        help="Processed scenario CSV (from tools/build_scenario.py). "
        "Omit to use the built-in synthetic day.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Run seed (default {DEFAULT_SEED}). Same seed + same config = bit-identical log.",
    )
    parser.add_argument(
        "--carry-in",
        default=None,
        help="Carry-in overlay YAML (from tools/build_synth_scenario.py). Sets fleet members' "
        "state at k=0 on top of the site config, and nothing else.",
    )
    parser.add_argument(
        "--dispatcher",
        default=catalog.DEFAULT_DISPATCHER,
        choices=catalog.known_dispatchers(),
        help=f"Policy under test (default {catalog.DEFAULT_DISPATCHER}). "
        "Registered in dispatch/catalog.py.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Site config YAML (default {DEFAULT_CONFIG_PATH}). A scenario is built for one "
        "fleet; running it against another site's config is not refused anywhere.",
    )
    args = parser.parse_args()
    run(args.scenario, args.seed, args.carry_in, args.dispatcher, args.config)
