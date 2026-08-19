"""Overlay columns from several CSVs on one axis, each normalised by its own maximum.

    python tools/plot_overlay.py --csv data/rooftop_175259_Microgridroom_20220601_20240208.csv \
                                --csv data/irradiance_iitm_20220601_20240208.csv \
                                --columns "Total PV,PV to battery (filtered),Total Load,ghi_w_m2" \
                                --start 2023-04-10 --end 2023-04-14

Sibling of tools/plot_csv.py, which draws ONE file. This one exists to compare
series that live in different files and have wildly different units -- kWh per
interval against W/m2 against kW -- which is impossible on a shared axis until
everything is made dimensionless.

Each column is divided by its own maximum OVER THE PLOTTED WINDOW, so every
trace spans 0..1 and only SHAPE is compared. The divisor is printed in the
legend, because a normalised plot with the scale thrown away is a picture that
cannot be checked against a number. Note the divisor is window-dependent by
design: narrowing --start/--end renormalises, which is what makes a single day
readable, and is also why the legend has to state it.

Read this file top to bottom; it runs in that order and there are no functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

# A native window if there is a display, a PNG if there is not. `matplotlib-base`
# ships no Qt, and TkAgg import fails outright over a bare SSH session, so the
# fallback is chosen here rather than discovered as a traceback.
try:
    matplotlib.use("TkAgg")
    _INTERACTIVE = True
except Exception:  # noqa: BLE001 - any backend failure means "no display"
    matplotlib.use("Agg")
    _INTERACTIVE = False

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TIMEZONE = "Asia/Kolkata"
# Column names tried, in order, as the x axis. The repo's own scenario CSVs use
# `timestamp`; `k` is the interval index a scenario also carries.
TIME_COLUMN_CANDIDATES = ("timestamp", "time", "Timestamp", "datetime", "k")

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--csv", action="append", required=True, type=Path, help="Repeatable.")
parser.add_argument(
    "--columns",
    default=None,
    help="Comma-separated column names to draw. Omit to draw every numeric column. "
    "A name is looked for in every file, so two files may each contribute some.",
)
parser.add_argument("--start", default=None, help="Inclusive, e.g. 2023-04-10.")
parser.add_argument("--end", default=None, help="Inclusive, e.g. 2023-04-14.")
parser.add_argument("--out", default=None, help="Write a PNG here as well as showing it.")
parser.add_argument("--raw", action="store_true", help="Do NOT normalise; plot native units.")
args = parser.parse_args()

wanted = [c.strip() for c in args.columns.split(",")] if args.columns else None

figure, axes = plt.subplots(figsize=(14, 7))
drawn = 0
missing: list[str] = []

# ---------------------------------------------------------------------------
# One pass per file: load, find the time axis, slice the window, draw each
# requested column normalised by its own maximum.
# ---------------------------------------------------------------------------
for path in args.csv:
    if not path.exists():
        print(f"no such file: {path}")
        sys.exit(2)

    frame = pd.read_csv(path)

    time_column = None
    for candidate in TIME_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            time_column = candidate
            break
    if time_column is None:
        print(f"{path}: none of {TIME_COLUMN_CANDIDATES} present; cannot place an x axis")
        sys.exit(2)

    if time_column == "k":
        frame = frame.set_index("k")
    else:
        index = pd.to_datetime(frame[time_column])
        # Sources here disagree about whether they carry an offset: the Victron
        # export is naive local, the irradiance CSV round-trips through +05:30.
        # Both are localised to the SAME named zone so that two files plotted
        # together line up on the clock rather than being silently 5.5 hours
        # apart -- which looks like a phase lag in the data, not a bug in the plot.
        index = (
            index.dt.tz_localize(TIMEZONE) if index.dt.tz is None else index.dt.tz_convert(TIMEZONE)
        )
        frame = frame.set_index(index).drop(columns=[time_column])

    if args.start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(args.start, tz=TIMEZONE)]
    if args.end is not None:
        # Inclusive of the whole end DAY, which is what a reader typing a date means.
        frame = frame.loc[frame.index <= pd.Timestamp(args.end, tz=TIMEZONE) + pd.Timedelta(days=1)]

    if len(frame) == 0:
        print(f"{path}: no rows inside {args.start} .. {args.end}")
        continue

    numeric = frame.select_dtypes("number")
    columns = [c for c in (wanted if wanted is not None else numeric.columns) if c in numeric]

    for name in columns:
        series = numeric[name].astype(float)
        peak = float(series.abs().max())
        if peak <= 0.0:
            # All-zero over this window: normalising would divide by zero, and
            # drawing it flat at 0 with a "/0" legend is a lie about the scale.
            print(f"  skipped {name!r} from {path.name}: identically zero over the window")
            continue
        if args.raw:
            axes.plot(series.index, series, linewidth=1.2, label=f"{name}  [{path.name}]")
        else:
            axes.plot(
                series.index,
                series / peak,
                linewidth=1.2,
                label=f"{name}  (/{peak:g})  [{path.name}]",
            )
        drawn += 1

    if wanted is not None:
        missing.extend(c for c in wanted if c not in numeric)

# A name absent from EVERY file is a typo the reader needs to hear about; absent
# from one file but present in another is normal and is not reported.
for name in dict.fromkeys(missing):
    if sum(1 for p in args.csv if name in pd.read_csv(p, nrows=0).columns) == 0:
        print(f"  column not found in any file: {name!r}")

if drawn == 0:
    print("nothing to draw")
    sys.exit(1)

window = f"{args.start or 'start'} .. {args.end or 'end'}"
axes.set_title(
    (
        "Normalised overlay -- each series divided by its own maximum over the window"
        if not args.raw
        else "Overlay, native units"
    )
    + f"\n{window}"
)
axes.set_ylabel("fraction of that series' maximum" if not args.raw else "native units")
axes.grid(alpha=0.3)
axes.legend(loc="upper left", fontsize=8, ncols=2)
figure.autofmt_xdate()
figure.tight_layout()

if args.out is not None:
    figure.savefig(args.out, dpi=140)
    print(f"  -> {args.out}")
if _INTERACTIVE:
    plt.show()
elif args.out is None:
    print("no display available and no --out given; nothing written")
