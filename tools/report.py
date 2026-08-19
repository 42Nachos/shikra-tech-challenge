"""Draw a run: the scenario that went in, what the dispatcher asked for, and
what the plant delivered -- on one page, on one time axis.

    python tools/report.py                    # runs/latest/  -> report.html
    python tools/report.py --run tuned_v2     # runs/tuned_v2/ -> report.html
    python tools/report.py --out /tmp/x.html

A run is a DIRECTORY that runner.py wrote -- `log.csv` and `manifest.json` --
and the page is written beside them, so everything about a run stays together
and can be copied or deleted as one thing.

Reads CSVs and nothing else. It imports neither sim/ nor dispatch/, and it
never re-runs the simulation -- a report is recomputed over an existing log
(CLAUDE.md), so re-rendering is free and cannot change what a run said.

WHY IT DISCOVERS COLUMNS INSTEAD OF KNOWING THEM. The run log's header is a
function of the fleet config: a second battery adds `bess_2.*`, a site without
a genset has no `dg_1.*` at all. So nothing here may name a member. Columns are
classified by pattern only -- `.cmd.` marks a command, the unit is read off the
suffix -- which is exactly what the units-in-every-name convention is for.

WHY PANELS INSTEAD OF ONE AXIS WITH TWO SCALES. Ticking kW and INR/kWh together
puts two incompatible scales on screen. Drawn on twin y-axes they appear to
cross somewhere, and where they cross is decided by the axis bounds rather than
by the data -- a correlation the reader did not have before and that is not
real. Same-unit signals share a panel and overlay honestly; a new unit opens a
new panel below. One drag zooms all of them, and one crosshair reads all of
them at the same instant, which is the comparison the twin axis was reaching
for.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]

_REPO_ROOT = Path(__file__).resolve().parent.parent

# What runner.py puts in a run directory, and what this adds to it. Repeated
# here rather than imported: this module reads CSVs and imports nothing
# internal, which is worth three string constants. Change them in both places.
LOG_NAME = "log.csv"
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "report.html"

# The command infix written by runner.flatten(). One reliable test for "this
# column is what the policy ASKED for" rather than what the plant did.
CMD_INFIX = ".cmd."
AVAILABILITY_SUFFIX = ".available"

# Unit by column suffix, LONGEST FIRST: a compound suffix like `_inr_per_kwh`
# ends in `_kwh` and is not an energy, so it must win.
UNITS: tuple[tuple[str, str], ...] = (
    ("_inr_per_kwh", "INR/kWh"),
    ("_w_m2", "W/m²"),
    ("_kvar", "kvar"),
    ("_kwh", "kWh"),
    ("_kva", "kVA"),
    ("_kw", "kW"),
    ("_pct", "%"),
    ("_hours", "h"),
    # LAST, and it must stay last: `_c` is a suffix of nothing else here, but it
    # is one character and would shadow a future `_c`-ending compound if it were
    # moved up. Live since ADR-0011 D10 put `temp_air_c` on the scenario.
    ("_c", "°C"),
)

# Which entity a scenario column describes. The scenario schema is fixed
# (sim/scenario.py's ScenarioRow), unlike the fleet, so naming these five is
# not the fleet-welded-into-the-tool mistake -- and the two prices are ONE
# entity, a tariff quoted in both directions.
SCENARIO_ENTITY: dict[str, str] = {
    "load_p_kw": "demand",
    "poa_irradiance_w_m2": "irradiance",
    # Its OWN entity rather than sharing irradiance's. They are both weather and
    # they drive one asset between them, but they are separately measured and
    # they peak hours apart -- seating them together would draw the lag out of
    # the picture, and the lag is why the column exists (ADR-0011 D10).
    "temp_air_c": "weather",
}

# Plant-level columns that belong to no member and are not entities: which
# asset held the bus, how far it moved from its advisory, the provenance flag.
# They are drawn in recessive ink ON PURPOSE rather than competing for a
# categorical hue -- a diagnostic that looks like a series reads as one.
DIAGNOSTIC = "\x00diagnostic"
DIAGNOSTIC_COLUMNS = frozenset(
    {"slack_id", "slack_advisory_kw", "slack_delta_kw", "violations", "quality"}
)

# Validated categorical palette, in slot order. Colour follows the ENTITY, so
# a member keeps its hue across every panel and ticking a signal off never
# repaints the survivors. Commanded-vs-delivered is carried by the DASH, not by
# a second hue -- which is what keeps a five-member fleet inside eight slots.
PALETTE_LIGHT = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
PALETTE_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)
MUTED = "#898781"

GROUP_DASH = {"scenario": "dot", "commanded": "dash", "delivered": "solid", "overlay": "dash"}
GROUP_LABEL = {
    "scenario": "Scenario",
    "delivered": "Delivered",
    "commanded": "Commanded",
    "overlay": "Overlay",
}

# The overlaid run's signal ids are prefixed, because two runs contribute the
# same column names and the sidebar keys on the id. The prefix never reaches a
# label -- the legend says which run in words.
OVERLAY_PREFIX = "overlay::"


@dataclass
class Signal:
    """One drawable column."""

    id: str
    label: str
    group: str  # scenario | delivered | commanded
    entity: str  # what it describes -- the hue is chosen by this
    unit: str  # "" when not a physical quantity
    kind: str  # numeric | categorical
    values: list[Any] = field(default_factory=list)
    # categorical only: the ordered category names its integer values index
    categories: list[str] = field(default_factory=list)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _unit_of(column: str) -> str:
    for suffix, unit in UNITS:
        if column.endswith(suffix):
            return unit
    return ""


def _entity_of(column: str, group: str) -> str:
    """Which physical thing this column is about.

    Member columns are `<member_id>.…`, so the member falls out of the name.
    `bess_1.cmd.p_setpoint_kw` and `bess_1.soc_kwh` are the same battery and
    must draw in the same colour.
    """
    if column in DIAGNOSTIC_COLUMNS:
        return DIAGNOSTIC
    if group == "scenario":
        if column.endswith(AVAILABILITY_SUFFIX):
            return column[: -len(AVAILABILITY_SUFFIX)]
        return SCENARIO_ENTITY.get(column, column)
    if "." in column:
        return column.split(".", 1)[0]
    return DIAGNOSTIC


def _classify(column: str, raw: list[str], group: str) -> Signal | None:
    """One column -> one Signal, or None for the columns that are not signals.

    An EMPTY CELL IS NOT A ZERO. runner.flatten() writes an un-commanded
    channel as empty, which is a different fact from a commanded zero -- a zero
    P setpoint on PV means curtail everything. Numeric signals therefore carry
    None (a gap in the line), never 0.0, and a channel that is empty in every
    interval is dropped rather than drawn as a flat line at zero.
    """
    if column in {"k", "timestamp"}:
        return None

    present = [v for v in raw if v != ""]
    if not present:
        return None  # never commanded anywhere -- nothing to draw

    label = column
    entity = _entity_of(column, group)
    unit = _unit_of(column)

    try:
        numbers: list[Any] = [float(v) if v != "" else None for v in raw]
    except ValueError:
        # Not numeric: a bool, a quality flag, a slack holder, a violation
        # string. Drawn as a step over its own categories rather than forced
        # onto a numeric axis it has no meaning on.
        categories = sorted({v for v in raw})
        index = {c: i for i, c in enumerate(categories)}
        return Signal(
            id=column,
            label=label,
            group=group,
            entity=entity,
            unit="",
            kind="categorical",
            values=[index[v] for v in raw],
            categories=[c if c != "" else "—" for c in categories],
        )

    return Signal(
        id=column,
        label=label,
        group=group,
        entity=entity,
        unit=unit or "value",
        kind="numeric",
        values=numbers,
    )


def build_signals(
    run_rows: list[dict[str, str]],
    scenario_rows: list[dict[str, str]] | None,
    overlay_rows: list[dict[str, str]] | None = None,
) -> list[Signal]:
    """Every drawable column, scenario first, then delivered, then commanded.

    Sidebar order is this order, so the groups read the way the interval does:
    what the world did, what the plant did about it, what it was told to do.

    OVERLAY MODE REALLOCATES THE DASH rather than adding a channel, and drops
    the commanded traces to pay for it. ADR-0007 D6 spends hue on the entity and
    dash on commanded-versus-delivered, and a second run needs a third
    distinction that the vocabulary does not have. Opacity was the obvious
    candidate and is a bad one: a faded line reads as an artefact rather than a
    deliberate statement, and it degrades exactly where a chart is most often
    read badly -- a projector, a print, low vision.

    So while overlaying, a comparison is between what two runs DID. Commanded is
    not drawn for either, dash carries the run, and hue still carries the
    entity, so a member keeps its colour across both runs and the eye compares
    like with like. The cost: commanded-versus-delivered -- the clip, refusal
    and slack-override diagnostic -- is unavailable in overlay mode, and needs a
    single-run report. That is the right trade for a page whose question is
    "where do these two differ", and the wrong one for "what did this policy get
    wrong", which is a different page.
    """
    signals: list[Signal] = []

    if scenario_rows:
        for column in scenario_rows[0]:
            raw = [r[column] for r in scenario_rows]
            sig = _classify(column, raw, "scenario")
            if sig is not None:
                signals.append(sig)

    for column in run_rows[0]:
        if overlay_rows is not None and CMD_INFIX in column:
            continue
        group = "commanded" if CMD_INFIX in column else "delivered"
        raw = [r[column] for r in run_rows]
        sig = _classify(column, raw, group)
        if sig is not None:
            signals.append(sig)

    if overlay_rows is not None:
        for column in overlay_rows[0]:
            if CMD_INFIX in column:
                continue
            raw = [r[column] for r in overlay_rows]
            sig = _classify(column, raw, "overlay")
            if sig is not None:
                # Same ENTITY as its counterpart in the base run, so the two
                # draw in one hue; a distinct ID, because the sidebar keys on it
                # and both runs contribute `bess_1.p_net_kw`.
                sig.id = OVERLAY_PREFIX + sig.id
                signals.append(sig)

    return signals


def assign_colours(signals: list[Signal]) -> dict[str, str]:
    """Slot index per entity. FLEET MEMBERS FIRST, in declaration order.

    The order matters and is not first-appearance: the scenario's columns are
    built before the run's, so appearance order spends slots on demand,
    irradiance and tariff and then runs out before the last member -- which is
    how load_1 ended up grey while a provenance flag held a hue. Members are
    the subject of the page, so they are seated first, and exogenous entities
    take what is left.

    Past eight the tail goes muted rather than getting a generated hue: a ninth
    categorical colour is indistinguishable from an existing one under
    colour-vision deficiency, so folding is the honest end of the ramp. Five
    members plus demand, irradiance and tariff is exactly eight, so a
    single-site fleet fits with nothing to spare.
    """
    # An entity is a MEMBER if anything the plant logged is about it. Decided
    # over all its signals rather than the first one seen: `bess_1.available`
    # is a scenario column about a member, and bucketing on that would file the
    # battery as exogenous.
    is_member = {s.entity for s in signals if s.group != "scenario"} - {DIAGNOSTIC}

    members: list[str] = []
    exogenous: list[str] = []
    for s in signals:
        if s.entity == DIAGNOSTIC:
            continue
        bucket = members if s.entity in is_member else exogenous
        if s.entity not in bucket:
            bucket.append(s.entity)

    slots = {e: str(i) for i, e in enumerate(members + exogenous) if i < len(PALETTE_LIGHT)}
    return {
        s.entity: slots.get(s.entity, "muted") if s.entity != DIAGNOSTIC else "muted"
        for s in signals
    }


def _default_ticks(signals: list[Signal]) -> list[str]:
    """What is on screen before anyone touches a checkbox: the power balance.

    Each asset's net power and the demand it was serving -- about six traces,
    which is a chart. Deliberately NOT every kW column: curtailment, shed,
    unserved and the two slack diagnostics are also kW, and ticking all of them
    opens on fourteen overlapping lines, which is a legend with a chart behind
    it. They are one click away in the sidebar.
    """
    return [
        s.id
        for s in signals
        if (s.group in ("delivered", "overlay") and s.id.endswith(".p_net_kw"))
        or (s.group in ("delivered", "overlay") and s.id.endswith(".served_p_kw"))
        or (s.group == "scenario" and s.id == "load_p_kw")
    ]


def cost_gap_rows(base: dict[str, Any], overlay: dict[str, Any] | None) -> tuple[str, str]:
    """The itemised bill, and beside it the gap against the overlaid run.

    THE HEADLINE OUTPUT of a comparison, and the reason `--overlay` is worth
    having at all: the traces show where two runs diverge, this shows what the
    divergence cost. One line per (member, term), because that is how the ledger
    reports and a scalar total hides which line moved (CLAUDE.md).

    Returns the table body and the header, so a single-run page keeps exactly
    the two columns it had.
    """
    cost = base.get("cost_breakdown_inr", {})
    if overlay is None:
        rows = "".join(
            f"<tr><td>{k}</td><td class='num'>{v:,.2f}</td></tr>" for k, v in cost.items()
        )
        return rows, "<tr><th>line</th><th class='num'>INR</th></tr>"

    other = overlay.get("cost_breakdown_inr", {})
    body: list[str] = []
    # Union, base order first: a term one run declares and the other does not is
    # a real difference and must not vanish from the table.
    keys = list(cost) + [k for k in other if k not in cost]
    for key in keys:
        mine, theirs = cost.get(key, 0.0), other.get(key, 0.0)
        gap = mine - theirs
        cls = " class='num gap'" if abs(gap) >= 0.005 else " class='num'"
        body.append(
            f"<tr><td>{key}</td><td class='num'>{mine:,.2f}</td>"
            f"<td class='num'>{theirs:,.2f}</td><td{cls}>{gap:+,.2f}</td></tr>"
        )
    header = (
        "<tr><th>line</th><th class='num'>this run</th>"
        "<th class='num'>overlay</th><th class='num'>gap</th></tr>"
    )
    return "".join(body), header


def _flatten_params(block: Any, prefix: str = "") -> list[tuple[str, str]]:
    """One declared block -> ordered (dotted name, formatted value) pairs.

    Dotted rather than reshaped, so a nested parameter reads the same on both
    sides: the genset's belief shows `fuel_curve.a` against the spec's
    `fuel_curve.a` without either being flattened into something it is not.
    """
    out: list[tuple[str, str]] = []
    if not isinstance(block, dict):
        return out
    for key, value in block.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out += _flatten_params(value, f"{name}.")
        elif isinstance(value, (list, tuple)):
            out.append((name, ", ".join(_number(v) for v in value)))
        else:
            out.append((name, _number(value)))
    return out


def _number(value: Any) -> str:
    """A parameter as text, without dropping digits that are the parameter.

    `%g` defaults to six significant figures, which turns ADR-0003 D8's
    sqrt(0.95) = 0.9746794 into 0.974679 -- a table of what the plant IS must not
    round the thing it is reporting. Ten figures covers every value in the specs
    and still drops the trailing zeros that make 300.0 read as 300.
    """
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def model_rows(manifest: dict[str, Any]) -> str:
    """The believed plant model beside the true spec, per member. ADR-0009 D5.

    SIDE BY SIDE AND UNMATCHED. The rows are aligned by position, not by name,
    and no difference is computed -- a belief need not have a counterpart in the
    spec at all, so pairing them would mean inventing one. What the reader gets
    is both blocks whole, which is enough to answer "did this run contain model
    error?" without going back to the config.

    Returns "" when the manifest predates the key, which is what keeps an older
    run renderable: this function reads the manifest and nothing else, so the
    card simply does not appear.
    """
    model = manifest.get("model") or {}
    if not model:
        return ""

    body: list[str] = []
    for member_id, sides in model.items():
        believed = _flatten_params(sides.get("believed"))
        spec = _flatten_params(sides.get("spec"))
        # At least one row per member, so a member with NOTHING to believe is
        # visible as such rather than absent -- the same reason D2 makes the
        # empty block mandatory in config.
        for i in range(max(len(believed), len(spec), 1)):
            b_name, b_value = believed[i] if i < len(believed) else ("", "")
            s_name, s_value = spec[i] if i < len(spec) else ("", "")
            first = f"<strong>{member_id}</strong>" if i == 0 else ""
            body.append(
                f"<tr><td>{first}</td><td>{b_name}</td><td class='num'>{b_value}</td>"
                f"<td>{s_name}</td><td class='num'>{s_value}</td></tr>"
            )

    head = (
        "<tr><th>member</th><th>believed</th><th class='num'>value</th>"
        "<th>spec (truth)</th><th class='num'>value</th></tr>"
    )
    return f"""
    <div class="card">
      <div class="bar"><strong style="font-size:13px">Plant model</strong>
        <span class="hint" style="margin:0">— what the dispatcher was assumed to
        believe, beside what the plant actually is. Listed side by side, not
        matched: a belief need not have a counterpart in the spec.</span>
      </div>
      <table>
        <thead>{head}</thead>
        <tbody>{"".join(body)}</tbody>
      </table>
    </div>"""


def render_html(
    signals: list[Signal],
    timestamps: list[str],
    manifest: dict[str, Any],
    run_path: Path,
    scenario_path: Path | None,
    scenario_note: str,
    overlay_manifest: dict[str, Any] | None = None,
) -> str:
    colours = assign_colours(signals)
    payload = {
        "x": timestamps,
        "signals": [
            {
                "id": s.id,
                "label": s.label,
                "group": s.group,
                "entity": s.entity,
                "unit": s.unit,
                "kind": s.kind,
                "slot": colours[s.entity],
                "dash": GROUP_DASH[s.group],
                "values": s.values,
                "categories": s.categories,
            }
            for s in signals
        ],
        "defaults": _default_ticks(signals),
        "paletteLight": list(PALETTE_LIGHT),
        "paletteDark": list(PALETTE_DARK),
        "muted": MUTED,
        "groupLabels": GROUP_LABEL,
    }

    cost_rows, cost_head = cost_gap_rows(manifest, overlay_manifest)
    overlay_note = ""
    if overlay_manifest is not None:
        overlay_note = (
            f"Overlaying <strong>{overlay_manifest.get('name', '?')}</strong> "
            f"(<code>{overlay_manifest.get('dispatcher', '?')}</code>) as DASHED lines. "
            f"Commanded traces are not drawn while overlaying — the dash carries the run."
        )

    total = manifest.get("total_inr", 0.0)
    total_row = f"<tr><td>TOTAL</td><td class='num'>{total:,.2f}</td></tr>"
    if overlay_manifest is not None:
        other_total = overlay_manifest.get("total_inr", 0.0)
        total_row = (
            f"<tr><td>TOTAL</td><td class='num'>{total:,.2f}</td>"
            f"<td class='num'>{other_total:,.2f}</td>"
            f"<td class='num gap'>{total - other_total:+,.2f}</td></tr>"
        )

    quality = manifest.get("quality_counts", {})
    intervals = manifest.get("intervals", len(timestamps))
    non_measured = sum(v for k, v in quality.items() if k != "measured")
    quality_bits = " · ".join(f"{k} {v:,}" for k, v in quality.items())
    quality_warn = (
        f"<span class='warn'>{non_measured:,} of {intervals:,} intervals are not measured data</span>"
        if non_measured
        else ""
    )

    span = f"{timestamps[0]} → {timestamps[-1]}" if timestamps else "—"
    sources = f"<code>{run_path}</code>"
    if scenario_path is not None:
        sources += f" · <code>{scenario_path}</code>"

    return _TEMPLATE.format(
        plotlyjs=get_plotlyjs(),
        payload=json.dumps(payload),
        name=manifest.get("name", "run"),
        dispatcher=manifest.get("dispatcher", "?"),
        seed=manifest.get("seed", "?"),
        intervals=f"{intervals:,}",
        span=span,
        quality_bits=quality_bits,
        quality_warn=quality_warn,
        scenario_note=scenario_note,
        cost_rows=cost_rows,
        cost_head=cost_head,
        total_row=total_row,
        model_card=model_rows(manifest),
        overlay_note=overlay_note,
        sources=sources,
    )


_TEMPLATE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — EMS run</title>
<style>
:root {{
  color-scheme: light;
  --surface: #fcfcfb; --page: #f9f9f7;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --warn: #d03b3b;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --surface: #1a1a19; --page: #0d0d0d;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --warn: #e66767;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface: #1a1a19; --page: #0d0d0d;
  --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  --warn: #e66767;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}}
header {{ padding: 20px 24px 16px; border-bottom: 1px solid var(--border); }}
h1 {{ font-size: 19px; margin: 0 0 4px; font-weight: 600; }}
.meta {{ color: var(--ink-2); font-size: 13px; }}
.meta code {{ color: var(--muted); font-size: 12px; }}
.warn {{ color: var(--warn); }}
.cols {{ display: flex; gap: 20px; align-items: flex-start; padding: 16px 24px 40px; }}
aside {{ flex: 0 0 250px; position: sticky; top: 12px; max-height: 92vh; overflow-y: auto; }}
main {{ flex: 1 1 auto; min-width: 0; }}
.card {{
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}}
.card + .card {{ margin-top: 14px; }}
.gname {{
  font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
  color: var(--muted); margin: 14px 0 6px;
}}
.gname:first-child {{ margin-top: 0; }}
label.sig {{
  display: flex; align-items: center; gap: 7px; padding: 3px 0;
  cursor: pointer; font-size: 12.5px; color: var(--ink-2);
  overflow-wrap: anywhere;
}}
label.sig:hover {{ color: var(--ink); }}
label.sig input {{ flex: 0 0 auto; margin: 0; accent-color: var(--ink-2); }}
.key {{ flex: 0 0 16px; height: 0; border-top-width: 2px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ text-align: left; padding: 3px 10px 3px 0; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.gap {{ font-weight: 600; }}
tfoot td {{ border-top: 1px solid var(--baseline); font-weight: 600; }}
.bar {{ display: flex; gap: 8px; align-items: center; margin-bottom: 10px; }}
button {{
  font: inherit; font-size: 12.5px; padding: 4px 11px; cursor: pointer;
  color: var(--ink-2); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px;
}}
button:hover {{ color: var(--ink); }}
input[type=range] {{
  accent-color: var(--ink-2); width: 190px; height: 18px; cursor: ew-resize;
}}
.ctl {{ font-size: 12.5px; color: var(--ink-2); }}
#tablewrap {{ max-height: 340px; overflow: auto; }}
#tablewrap table {{ font-variant-numeric: tabular-nums; font-size: 12px; }}
#tablewrap th {{
  position: sticky; top: 0; background: var(--surface);
  border-bottom: 1px solid var(--border); white-space: nowrap;
}}
#tablewrap td {{ white-space: nowrap; padding-right: 14px; }}
.hint {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
</style>

<header>
  <h1>{name}</h1>
  <div class="meta">
    {dispatcher} · seed {seed} · {intervals} intervals · {span}<br>
    {quality_bits} {quality_warn}<br>
    {scenario_note}
    {overlay_note}
    <div style="margin-top:6px">{sources}</div>
  </div>
</header>

<div class="cols">
  <aside>
    <div class="card">
      <div class="bar">
        <button id="none">Clear</button>
        <button id="theme">Theme</button>
      </div>
      <div id="picker"></div>
    </div>
  </aside>

  <main>
    <div class="card">
      <div class="bar">
        <label class="ctl" for="rowh">Panel height</label>
        <input type="range" id="rowh" min="140" max="700" step="20">
        <span class="ctl" id="rowhval" style="font-variant-numeric:tabular-nums"></span>
        <button id="rowhreset">Reset</button>
      </div>
      <div id="plot"></div>
      <div class="hint">
        Drag to zoom — every panel moves together. Hovering draws one line
        through all panels so you can read the same instant everywhere.
        Double-click to reset the zoom.
      </div>
    </div>

    <div class="card">
      <table>
        <thead>{cost_head}</thead>
        <tbody>{cost_rows}</tbody>
        <tfoot>{total_row}</tfoot>
      </table>
    </div>
{model_card}

    <div class="card">
      <div class="bar"><strong style="font-size:13px">Table view</strong>
        <span class="hint" style="margin:0">— the ticked signals over the visible range</span>
      </div>
      <div id="tablewrap"></div>
    </div>
  </main>
</div>

<script>{plotlyjs}</script>
<script>
const DATA = {payload};
const gd = document.getElementById('plot');
const checked = new Set(DATA.defaults);

// Height is PER PANEL, not for the plot as a whole: ticking a fourth unit
// should make the page taller, not shave a third off the three panels already
// on screen. A categorical strip counts as a fraction of one, since it only
// ever shows two or three states.
const HEIGHT_KEY = 'ems.panelHeight';
const DEFAULT_ROW_H = 260;
function storedHeight() {{
  try {{ return Number(localStorage.getItem(HEIGHT_KEY)) || 0; }} catch (e) {{ return 0; }}
}}
function storeHeight(v) {{
  try {{ localStorage.setItem(HEIGHT_KEY, String(v)); }} catch (e) {{ /* file:// */ }}
}}
let rowH = storedHeight() || DEFAULT_ROW_H;

function plotHeight(rows) {{
  if (!rows.length) return 320;
  const units = rows.reduce((a, r) => a + (r.cat ? 0.42 : 1), 0);
  return Math.round(Math.max(rowH, units * rowH));
}}

function dark() {{
  const stamped = document.documentElement.getAttribute('data-theme');
  if (stamped) return stamped === 'dark';
  return matchMedia('(prefers-color-scheme: dark)').matches;
}}
function ink(name) {{
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}}
function colourOf(sig) {{
  if (sig.slot === 'muted') return DATA.muted;
  return (dark() ? DATA.paletteDark : DATA.paletteLight)[Number(sig.slot)];
}}

// ---- sidebar. Signal ids come from a CSV header, so they are untrusted text
// and go in as textContent, never as markup.
function buildPicker() {{
  const host = document.getElementById('picker');
  host.textContent = '';
  for (const group of ['scenario', 'delivered', 'commanded']) {{
    const inGroup = DATA.signals.filter(s => s.group === group);
    if (!inGroup.length) continue;
    const h = document.createElement('div');
    h.className = 'gname';
    h.textContent = DATA.groupLabels[group];
    host.appendChild(h);
    for (const sig of inGroup) {{
      const label = document.createElement('label');
      label.className = 'sig';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = checked.has(sig.id);
      box.addEventListener('change', () => {{
        box.checked ? checked.add(sig.id) : checked.delete(sig.id);
        render();
      }});
      const key = document.createElement('span');
      key.className = 'key';
      key.style.borderTopStyle = sig.dash === 'solid' ? 'solid'
                               : sig.dash === 'dash' ? 'dashed' : 'dotted';
      key.style.borderTopColor = colourOf(sig);
      const text = document.createElement('span');
      text.textContent = sig.label;
      label.append(box, key, text);
      host.appendChild(label);
    }}
  }}
}}

// ---- one panel per unit; every categorical signal gets its own thin strip.
function layoutRows() {{
  const chosen = DATA.signals.filter(s => checked.has(s.id));
  const units = [];
  const byUnit = new Map();
  for (const s of chosen) {{
    if (s.kind !== 'numeric') continue;
    if (!byUnit.has(s.unit)) {{ byUnit.set(s.unit, []); units.push(s.unit); }}
    byUnit.get(s.unit).push(s);
  }}
  const rows = units.map(u => ({{ title: u, sigs: byUnit.get(u), weight: 3 }}));
  for (const s of chosen.filter(s => s.kind === 'categorical')) {{
    rows.push({{ title: s.label, sigs: [s], weight: 1, cat: true }});
  }}
  return rows;
}}

function render() {{
  const rows = layoutRows();
  const surface = ink('--surface'), grid = ink('--grid');
  const axisInk = ink('--muted'), textInk = ink('--ink-2');

  const traces = [], layout = {{
    // Height goes in the LAYOUT, not on the div. Plotly records its own height
    // on first draw and keeps using it, so growing the container alone makes
    // the page taller while the plot inside stays exactly as it was.
    height: plotHeight(rows),
    paper_bgcolor: surface, plot_bgcolor: surface,
    font: {{ color: textInk, family: 'system-ui, -apple-system, sans-serif', size: 12 }},
    margin: {{ l: 68, r: 18, t: 26, b: 44 }},
    showlegend: true,
    legend: {{ orientation: 'h', y: 1.04, x: 0, font: {{ size: 11.5 }} }},
    hovermode: 'x unified',
    hoverlabel: {{ bgcolor: surface, bordercolor: ink('--border'),
                  font: {{ color: ink('--ink'), size: 12 }} }},
    xaxis: {{
      // The crosshair. `across` is what carries the line through EVERY panel
      // rather than stopping at the one under the pointer -- the whole point
      // is reading one instant in all of them at once.
      showspikes: true, spikemode: 'across', spikesnap: 'cursor',
      spikecolor: axisInk, spikethickness: 1, spikedash: 'solid',
      gridcolor: grid, linecolor: ink('--baseline'), zeroline: false,
      tickfont: {{ color: axisInk, size: 11 }}, anchor: 'y'
    }},
    spikedistance: -1
  }};

  if (!rows.length) {{
    gd.style.height = plotHeight(rows) + 'px';
    Plotly.react(gd, [], {{ ...layout, annotations: [{{
      text: 'Tick a signal to draw it.', showarrow: false,
      xref: 'paper', yref: 'paper', x: 0.5, y: 0.5,
      font: {{ color: axisInk, size: 13 }} }}],
      xaxis: {{ visible: false }}, yaxis: {{ visible: false }} }},
      {{ displaylogo: false, responsive: true }});
    buildTable([]);
    return;
  }}

  // Stack top-down. The gap is a FRACTION of the plot, so it is computed from
  // a pixel target -- a fixed fraction would grow into a canyon as the panels
  // get taller and crush the strips when they get short.
  const gap = Math.min(0.09, 34 / plotHeight(rows));
  const total = rows.reduce((a, r) => a + r.weight, 0);
  const free = 1 - gap * (rows.length - 1);
  let top = 1;
  rows.forEach((row, i) => {{
    const h = free * row.weight / total;
    const key = i === 0 ? 'yaxis' : 'yaxis' + (i + 1);
    const id = i === 0 ? 'y' : 'y' + (i + 1);
    layout[key] = {{
      domain: [Math.max(0, top - h), top],
      title: {{ text: row.title, font: {{ size: 11.5, color: axisInk }} }},
      gridcolor: grid, zeroline: !row.cat, zerolinecolor: ink('--baseline'),
      linecolor: ink('--baseline'), tickfont: {{ color: axisInk, size: 11 }}
    }};
    if (row.cat) {{
      const cats = row.sigs[0].categories;
      layout[key].tickvals = cats.map((_, n) => n);
      layout[key].ticktext = cats;
      layout[key].range = [-0.5, cats.length - 0.5];
      layout[key].title.text = '';
    }}
    for (const sig of row.sigs) {{
      traces.push({{
        type: 'scattergl', mode: 'lines', name: sig.label,
        x: DATA.x, y: sig.values, yaxis: id,
        line: {{ color: colourOf(sig), width: 2, dash: sig.dash,
                shape: row.cat ? 'hv' : 'linear' }},
        hovertemplate: row.cat
          ? '%{{customdata}}<extra>' + sig.label + '</extra>'
          : '%{{y}} ' + sig.unit + '<extra>' + sig.label + '</extra>',
        customdata: row.cat ? sig.values.map(v => sig.categories[v]) : undefined,
        connectgaps: false
      }});
    }}
    top -= h + gap;
  }});

  gd.style.height = plotHeight(rows) + 'px';
  const keep = gd.layout && gd.layout.xaxis && gd.layout.xaxis.range;
  if (keep && !(gd.layout.xaxis.autorange)) layout.xaxis.range = keep.slice();
  Plotly.react(gd, traces, layout, {{ displaylogo: false, responsive: true }});
  buildTable(rows.flatMap(r => r.sigs));
}}

// ---- table view. A value must be reachable without hovering, so the ticked
// signals are also printed for whatever window is on screen.
const MAX_TABLE_ROWS = 300;
function buildTable(sigs) {{
  const host = document.getElementById('tablewrap');
  host.textContent = '';
  if (!sigs.length) return;

  let lo = 0, hi = DATA.x.length - 1;
  const r = gd.layout && gd.layout.xaxis && gd.layout.xaxis.range;
  if (r && !gd.layout.xaxis.autorange) {{
    const a = new Date(r[0]).getTime(), b = new Date(r[1]).getTime();
    lo = DATA.x.findIndex(t => new Date(t).getTime() >= a);
    hi = DATA.x.length - 1;
    for (let i = DATA.x.length - 1; i >= 0; i--) {{
      if (new Date(DATA.x[i]).getTime() <= b) {{ hi = i; break; }}
    }}
    if (lo < 0) lo = 0;
  }}
  const n = Math.max(0, hi - lo + 1);
  const step = Math.max(1, Math.ceil(n / MAX_TABLE_ROWS));

  const table = document.createElement('table');
  const thead = table.createTHead().insertRow();
  for (const h of ['timestamp', ...sigs.map(s => s.label)]) {{
    const th = document.createElement('th');
    th.textContent = h;
    thead.appendChild(th);
  }}
  const body = table.createTBody();
  for (let i = lo; i <= hi; i += step) {{
    const tr = body.insertRow();
    tr.insertCell().textContent = DATA.x[i];
    for (const s of sigs) {{
      const v = s.values[i];
      tr.insertCell().textContent =
        s.kind === 'categorical' ? s.categories[v] : (v === null ? '' : v);
    }}
  }}
  host.appendChild(table);
  if (step > 1) {{
    const p = document.createElement('div');
    p.className = 'hint';
    p.textContent = 'Showing every ' + step + 'th of ' + n +
                    ' intervals. Zoom in for more; the CSV has all of them.';
    host.appendChild(p);
  }}
}}

// ---- panel height. Re-renders rather than only resizing the div, because
// the inter-panel gap is derived from the pixel height and has to be recomputed.
const slider = document.getElementById('rowh');
const readout = document.getElementById('rowhval');
function showHeight() {{
  slider.value = String(rowH);
  readout.textContent = rowH + ' px';
}}
slider.addEventListener('input', () => {{
  rowH = Number(slider.value);
  readout.textContent = rowH + ' px';
  render();
}});
slider.addEventListener('change', () => storeHeight(rowH));
document.getElementById('rowhreset').addEventListener('click', () => {{
  rowH = DEFAULT_ROW_H; storeHeight(rowH); showHeight(); render();
}});

document.getElementById('none').addEventListener('click', () => {{
  checked.clear(); buildPicker(); render();
}});
document.getElementById('theme').addEventListener('click', () => {{
  document.documentElement.setAttribute('data-theme', dark() ? 'light' : 'dark');
  buildPicker(); render();
}});
showHeight();
buildPicker();
render();
// Registered AFTER the first render -- Plotly attaches .on to the div when it
// draws, so binding earlier binds to nothing.
gd.on('plotly_relayout', () => buildTable(layoutRows().flatMap(r => r.sigs)));
</script>
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", default="latest", help="Run directory under runs/ (default: latest).")
    p.add_argument(
        "--overlay",
        default=None,
        help="A second run directory to draw over this one, as dashed lines. "
        "Commanded traces are dropped while overlaying (the dash carries the run).",
    )
    p.add_argument("--out", default=None, help=f"Output HTML (default: runs/<run>/{REPORT_NAME}).")
    args = p.parse_args()

    run_dir = _REPO_ROOT / "runs" / args.run
    run_path = run_dir / LOG_NAME
    manifest_path = run_dir / MANIFEST_NAME
    if not run_path.exists():
        # A pre-directory run log sits at runs/<name>.csv. Say so, because
        # "no run log" while the file is plainly there is a maddening message.
        legacy = _REPO_ROOT / "runs" / f"{args.run}.csv"
        if legacy.exists():
            print(
                f"{legacy} is the old flat layout. A run is a directory now "
                f"({run_dir}{os.sep}) -- re-run `python runner.py` to write one.",
                file=sys.stderr,
            )
        else:
            print(f"no run log at {run_path}. Run `python runner.py` first.", file=sys.stderr)
        return 1

    run_rows = _read_csv(run_path)
    if not run_rows:
        print(f"{run_path} has no intervals.", file=sys.stderr)
        return 1

    manifest: dict[str, Any] = {"name": args.run}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    # The scenario is read from where the run said it was, NOT copied at run
    # time. Nothing detects that the file changed since -- except the interval
    # count, which is checked below because a silent misalignment would draw
    # every input against the wrong interval.
    scenario_rows: list[dict[str, str]] | None = None
    scenario_path: Path | None = None
    declared = manifest.get("scenario")
    if declared is None:
        note = (
            "No scenario file — this run used the built-in synthetic day, so it has no "
            "exogenous inputs to draw. Pass <code>--scenario</code> to runner.py for those."
        )
    else:
        scenario_path = Path(declared)
        if not scenario_path.is_absolute():
            scenario_path = _REPO_ROOT / scenario_path
        if not scenario_path.exists():
            note = f"<span class='warn'>Scenario {declared} is gone — inputs not drawn.</span>"
            scenario_path = None
        else:
            scenario_rows = _read_csv(scenario_path)
            if len(scenario_rows) != len(run_rows):
                note = (
                    f"<span class='warn'>{declared} has {len(scenario_rows):,} intervals but "
                    f"the run has {len(run_rows):,} — it changed since. Inputs not drawn.</span>"
                )
                scenario_rows, scenario_path = None, None
            else:
                note = f"Scenario <code>{declared}</code>"

    overlay_rows: list[dict[str, str]] | None = None
    overlay_manifest: dict[str, Any] | None = None
    if args.overlay:
        overlay_dir = _REPO_ROOT / "runs" / args.overlay
        overlay_log = overlay_dir / LOG_NAME
        if not overlay_log.exists():
            print(f"no run log at {overlay_log}.", file=sys.stderr)
            return 1
        overlay_rows = _read_csv(overlay_log)
        if len(overlay_rows) != len(run_rows):
            # Two runs on different timelines share an x-axis only by accident,
            # and every trace after the shorter one ends would be drawn against
            # the wrong interval. Refused rather than truncated.
            print(
                f"{args.overlay} has {len(overlay_rows):,} intervals and {args.run} has "
                f"{len(run_rows):,}. They are not the same timeline, so overlaying them "
                f"would draw one against the other's clock.",
                file=sys.stderr,
            )
            return 1
        overlay_manifest = {"name": args.overlay}
        overlay_manifest_path = overlay_dir / MANIFEST_NAME
        if overlay_manifest_path.exists():
            overlay_manifest = json.loads(overlay_manifest_path.read_text())

    signals = build_signals(run_rows, scenario_rows, overlay_rows)
    timestamps = [r["timestamp"] for r in run_rows]
    html = render_html(
        signals, timestamps, manifest, run_path, scenario_path, note, overlay_manifest
    )

    out = Path(args.out) if args.out else run_dir / REPORT_NAME
    out.write_text(html)
    drawn = sum(1 for s in signals if s.kind == "numeric")
    print(f"Wrote {out} — {len(signals)} signals ({drawn} numeric), {len(run_rows):,} intervals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
