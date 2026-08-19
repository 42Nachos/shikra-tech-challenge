"""Open a CSV in an interactive window.

    python tools/plot_csv.py data/plant_ts_total_load.csv

Every numeric column is drawn as a line on one shared axes. Pan and zoom with
the toolbar; click a legend entry to hide or show that series.

The x-axis is worked out from the file: a `Date` and a `Time` column are
combined into a timestamp, a single timestamp-ish column is parsed on its own,
and a file with neither is plotted against the row number. Nothing else about
the file is assumed -- the header is whatever the first row says, and a column
counts as numeric only if every one of its values parses as a float.

This is for eyeballing a CSV from the terminal, which is why it is matplotlib
and not the plotly that `tools/report.py` uses: plotly would give an HTML page.
The window blocks the terminal it was launched from and dies with it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

# Chosen before pyplot is imported, so the window works over a plain X/Wayland
# session with no Qt in the environment (`matplotlib-base` ships no Qt).
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Tried in order against the whole column; the first that parses every value
# wins. The ETAP reports print MM-DD-YYYY, which is first only because ISO is
# unambiguous and will not be mistaken for it.
DATE_FORMATS = ("%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y")
TIME_FORMATS = ("%H:%M:%S", "%H:%M")

# Names a lone x-axis column may go by, lowercased.
TIMESTAMP_NAMES = ("timestamp", "datetime", "date_time", "time", "date")


class PlotError(Exception):
    """The CSV had nothing this tool could draw."""


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise PlotError(f"{path}: empty file")
    header, data = rows[0], [r for r in rows[1:] if any(field.strip() for field in r)]
    if not data:
        raise PlotError(f"{path}: header only, no data rows")
    width = len(header)
    for i, row in enumerate(data, start=2):
        if len(row) != width:
            raise PlotError(f"{path}: line {i} has {len(row)} fields, header has {width}")
    return header, data


def column(header: list[str], data: list[list[str]], name: str) -> list[str]:
    return [row[header.index(name)] for row in data]


def numeric_columns(header: list[str], data: list[list[str]]) -> dict[str, list[float]]:
    """The columns whose every value is a float or empty, in file order.

    An EMPTY cell becomes NaN, which matplotlib draws as a break in the line.
    That is deliberate: in a run log an empty cell means NOT COMMANDED, and
    plotting it as 0.0 would draw a command that was never issued. A column
    holding any other unparseable value is not numeric and is skipped.
    """
    found: dict[str, list[float]] = {}
    for index, name in enumerate(header):
        values: list[float] = []
        for row in data:
            cell = row[index].strip()
            if not cell:
                values.append(float("nan"))
                continue
            try:
                values.append(float(cell))
            except ValueError:
                break
        else:
            if any(value == value for value in values):  # not all-NaN
                found[name] = values
    return found


def parse_all(values: list[str], formats: tuple[str, ...]) -> tuple[str, list[datetime]] | None:
    """The first format that parses every value, with the parsed result."""
    for fmt in formats:
        try:
            return fmt, [datetime.strptime(value.strip(), fmt) for value in values]
        except ValueError:
            continue
    return None


def find_column(header: list[str], name: str) -> str | None:
    for candidate in header:
        if candidate.strip().lower() == name:
            return candidate
    return None


def x_axis(header: list[str], data: list[list[str]]) -> tuple[list[Any], str, str]:
    """The x values, their axis label, and a note on how they were read."""
    date_col, time_col = find_column(header, "date"), find_column(header, "time")
    if date_col is not None and time_col is not None:
        dates = parse_all(column(header, data, date_col), DATE_FORMATS)
        times = parse_all(column(header, data, time_col), TIME_FORMATS)
        if dates is not None and times is not None:
            combined = [
                datetime.combine(day.date(), clock.time())
                for day, clock in zip(dates[1], times[1], strict=True)
            ]
            note = f"{date_col} + {time_col} as {dates[0]} {times[0]}"
            return list(combined), "time", note

    for name in TIMESTAMP_NAMES:
        found = find_column(header, name)
        if found is None:
            continue
        values = column(header, data, found)
        try:
            return [datetime.fromisoformat(v.strip()) for v in values], "time", f"{found} as ISO"
        except ValueError:
            parsed = parse_all(values, DATE_FORMATS + TIME_FORMATS)
            if parsed is not None:
                return list(parsed[1]), "time", f"{found} as {parsed[0]}"

    return list(range(len(data))), "row", "row number (no timestamp column found)"


def on_legend_pick(figure: Any, mapping: dict[Any, Line2D]) -> Any:
    """Click a legend entry to toggle its series."""

    def handler(event: Any) -> None:
        line = mapping.get(event.artist)
        if line is None:
            return
        visible = not line.get_visible()
        line.set_visible(visible)
        event.artist.set_alpha(1.0 if visible else 0.25)
        figure.canvas.draw_idle()

    return handler


def plot(path: Path) -> None:
    header, data = read_csv(path)
    series = numeric_columns(header, data)
    x, x_label, note = x_axis(header, data)

    # Whatever became the x-axis is not also a line on the y-axis.
    for name in list(series):
        if name.strip().lower() in TIMESTAMP_NAMES:
            del series[name]
    if not series:
        raise PlotError(f"{path}: no numeric columns to plot")

    figure, axes = plt.subplots(figsize=(14, 7))
    lines: list[Line2D] = []
    for name, values in series.items():
        lines.append(axes.plot(x, values, linewidth=1.0, label=name)[0])

    axes.set_xlabel(x_label)
    axes.set_title(f"{path.name}  --  {len(data)} rows, x = {note}")
    axes.grid(True, alpha=0.3)
    axes.margins(x=0.01)
    if x_label == "time":
        figure.autofmt_xdate()

    legend = axes.legend(loc="upper right", framealpha=0.9)
    mapping: dict[Any, Line2D] = {}
    for entry, line in zip(legend.get_lines(), lines, strict=True):
        entry.set_picker(8)  # points of slack around the legend line
        mapping[entry] = line
    figure.canvas.mpl_connect("pick_event", on_legend_pick(figure, mapping))

    figure.tight_layout()
    print(f"{path}: {len(data)} rows, {len(series)} numeric columns, x = {note}")
    print("  toolbar pans and zooms; click a legend entry to toggle a series")
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="Path to the CSV to plot.")
    args = parser.parse_args()
    try:
        plot(args.csv)
    except (PlotError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
