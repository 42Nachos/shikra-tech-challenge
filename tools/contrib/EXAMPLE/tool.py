"""Print one line per cost line from a run directory. The dumbest tool that runs.

REFERENCE SUBMISSION, never shipped in the pack. It exists so the track 4 smoke
runner has something that produces an artifact. `tools/report.py` already does
this properly and in HTML; this is the floor.

    python tool.py --run runs/latest --out ./out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", type=Path, required=True, help="A directory holding manifest.json.")
parser.add_argument("--out", type=Path, required=True, help="A DIRECTORY to write into.")
args = parser.parse_args()

manifest = json.loads((args.run / "manifest.json").read_text())
lines = manifest["cost_breakdown_inr"]

report = [f"run: {args.run}", f"intervals: {manifest.get('intervals', '?')}", ""]
report += [f"{key:28} {value:14,.2f}" for key, value in lines.items()]
report += ["", f"{'TOTAL':28} {manifest['total_inr']:14,.2f}"]

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "summary.txt").write_text("\n".join(report) + "\n")
print("\n".join(report))
