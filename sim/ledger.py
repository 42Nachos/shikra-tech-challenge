"""The tally: one cost object per member, driven over the run. ADR-0004 D7/D8.

Cost is NOT a sum -- see CLAUDE.md. This module owns the LOOP and the CALENDAR
and no arithmetic:

    build       one cost model per member, asked of the member itself
    update      hand every member its own keyed delivered record
    rollover    detect the billing month change and close every term
    finalize    one CostBreakdown line per (member, term), never a scalar

Every rupee of arithmetic lives on the model that generates the quantity being
priced (ADR-0016 D3). Adding a cost term is a class beside its model plus a
`cost_terms:` line in config, with no edit here.

STILL MISSING (D8): genuinely cross-asset terms -- a renewable mandate, or any
portfolio-level charge -- which cannot attach to one member and would be computed
here. There are none today, so `finalize()` emits member lines only.

An INDEPENDENT implementation of the cost model: it must never share code with
dispatch/'s 5.2 cost function (D7), whose disagreement with it is a measured
output.
"""

from __future__ import annotations

from config.schema import SiteConfig
from contracts import CostBreakdown, billing_month, parse_timestamp
from sim.asset import AssetCost
from sim.assets.grid_models.tnerc_ht import DemandChargeParams
from sim.cost_terms import IMPLEMENTED_COST_TERMS
from sim.load import LoadCost
from sim.scenario import Interval
from sim.fleet import AssembledFleet
from sim.plant import FleetDelivered
from sim.scenario import ScenarioRow

__all__ = ["IMPLEMENTED_COST_TERMS", "Ledger"]


def _check_demand_windows(assembled: AssembledFleet, config: SiteConfig) -> None:
    """A maximum-demand window must be a whole number of plant steps.

      * SHORTER than the plant step is UNREPRESENTABLE -- a 15-minute peak
        cannot be recovered from hourly averages.
      * not a whole multiple has no defined block.

    Here because it is the one seam that sees both the term's params and the
    site's step. `DemandChargeParams` separately refuses a window that does not
    divide the DAY, which keeps a block off a month boundary.
    """
    for view in assembled.view.assets:
        for term in view.cost_terms:
            if term.kind != "demand_charge":
                continue
            window = DemandChargeParams.model_validate(dict(term.params)).demand_window_minutes
            if window < config.dt_minutes or window % config.dt_minutes != 0:
                raise ValueError(
                    f"{view.id}: demand_window_minutes={window} is not a whole number of "
                    f"{config.dt_minutes}-minute plant steps. A maximum demand is the average "
                    f"over its window, and this plant produces no reading it could average."
                )


class Ledger:
    """Prices one run. Constructed from the assembled fleet, so what it can
    charge for is exactly what the fleet declared."""

    def __init__(self, assembled: AssembledFleet, config: SiteConfig) -> None:
        self._dt_minutes = config.dt_minutes
        _check_demand_windows(assembled, config)

        # DECLARATION ORDER throughout (D10): these drive the update loop and
        # the order of the CostBreakdown's lines, and float addition is not
        # associative.
        assets = {a.id: a for a in assembled.assets}
        loads = {ln.id: ln for ln in assembled.loads}

        # ONE cost object per member, asked of the member itself. The ledger
        # implements no tariff.
        self._asset_costs: dict[str, AssetCost] = {}
        self._load_costs: dict[str, LoadCost] = {}

        for view in assembled.view.assets:
            self._asset_costs[view.id] = assets[view.id].build_cost()
        for load_view in assembled.view.loads:
            self._load_costs[load_view.id] = loads[load_view.id].build_cost()

        # The open billing month. None between a close and the next update, so
        # close_billing_period() and finalize() are idempotent together.
        self._month: tuple[int, int] | None = None

    def update(self, delivered: FleetDelivered, row: ScenarioRow) -> None:
        """Price one interval, closing the previous billing month first if this
        row crossed into a new one."""
        ts = parse_timestamp(row.timestamp)
        month = billing_month(row.timestamp)
        if self._month is None:
            self._month = month
        elif month != self._month:
            self._close_period()  # settle the month that just ended
            self._month = month

        interval = Interval(row=row, ts=ts, dt_minutes=self._dt_minutes)
        for member_id, cost in self._asset_costs.items():
            cost.update(delivered.assets[member_id], interval)
        for load_id, load_cost in self._load_costs.items():
            load_cost.update(delivered.loads[load_id], interval)

    def _close_period(self) -> None:
        """Settle every path-dependent term for the open billing period."""
        if self._month is None:
            return
        for cost in self._asset_costs.values():
            cost.close_period()
        for load_cost in self._load_costs.values():
            load_cost.close_period()
        self._month = None

    def close_billing_period(self) -> None:
        """Flush the open billing month. Idempotent with finalize()."""
        self._close_period()

    def finalize(self) -> CostBreakdown:
        """One line per (member, term), in fleet declaration order.

        Each line is rounded to the paisa and `total_inr` adds the ROUNDED lines,
        so the printed items always add up to the printed total.
        """
        self._close_period()  # no-op if close_billing_period already ran
        lines: dict[str, float] = {}
        for member_id, cost in self._asset_costs.items():
            for kind, amount_inr in cost.lines().items():
                lines[f"{member_id}.{kind}"] = round(amount_inr, 2)
        for load_id, load_cost in self._load_costs.items():
            for kind, amount_inr in load_cost.lines().items():
                lines[f"{load_id}.{kind}"] = round(amount_inr, 2)
        return CostBreakdown(lines=lines)
