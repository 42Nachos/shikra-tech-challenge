"""The asset and load type registry. ADR-0004 D1/D9.

A new asset type is a new file in `sim/assets/` plus one registration here; a new
customer is a config file.

A REGISTRY ENTRY states what a TYPE is capable of -- a fact about the equipment,
not about one site's choices. A PV array can never be slack-bearing (D5); a
rotating genset has no controllable reactive channel (D3). Config makes CHOICES
within that ceiling, and `sim/fleet.py` checks `chosen <= capable`.

There is no separate "builder" callable: the registry knows WHICH class, and the
class knows how to build itself through `from_config`.

`spec_cls`, `believed_cls` and `noise_cls` do the same for a fleet entry's
`spec:`, `believed:` and `noise:` mappings, which `config/schema.py` receives as
raw mappings because config/ cannot import sim/.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from contracts import (
    AssetCapabilities,
    BessPlantModel,
    EmptyPlantModel,
    GensetPlantModel,
    PlantModel,
    PvPlantModel,
)
from sim.asset import Asset, AssetNoise
from sim.assets.bess import BessAsset, BessNoise, BessSpec
from sim.assets.genset import GensetAsset, GensetSpec
from sim.assets.grid import GridAsset, GridSpec
from sim.assets.pv import PvAsset, PvSpec
from sim.load import FixedLoad, Load, LoadNoise


@dataclass(frozen=True, kw_only=True)
class AssetTypeSpec:
    """One registered asset type: what it can do, and what builds it.

    The capabilities live in contracts.py because the dispatcher needs to read
    them (see AssetCapabilities). The class does not: it carries physics and
    names AssetState, so it stays on this side of the boundary.
    """

    capabilities: AssetCapabilities
    # Each class carries its own `from_config` classmethod; see the module
    # docstring.
    asset_cls: type[Asset] | None = None
    # Validates this type's `spec:` mapping. None means the type's model takes
    # no parameters, and an entry carrying a spec anyway is refused.
    spec_cls: type[BaseModel] | None = None
    # Which parameters the DISPATCHER is assumed to believe about this type
    # (ADR-0009 D2/D3), validating the entry's `believed:` mapping.
    #
    # REQUIRED, with no None case: a type with nothing to estimate registers
    # `EmptyPlantModel`, so "nothing" is an answer that has to be given.
    believed_cls: type[PlantModel]
    # What this type's INSTRUMENTS do, validating the entry's `noise:` mapping
    # (ADR-0012 D2). REQUIRED, with no None case, like `believed_cls`.
    #
    # A pydantic model rather than a PlantModel: noise is TRUTH-SIDE and has no
    # business in contracts.py. Annotated with the concrete base, since every
    # asset's noise class derives from AssetNoise.
    noise_cls: type[AssetNoise]

    @property
    def name(self) -> str:
        return self.capabilities.name


@dataclass(frozen=True, kw_only=True)
class LoadTypeSpec:
    """One registered load type. Thin: loads have far fewer axes than assets."""

    name: str
    load_cls: type[Load] | None = None
    # As on AssetTypeSpec, and required for the same reason. Every load type
    # today registers EmptyPlantModel, but it has to say so.
    believed_cls: type[PlantModel]
    # As on AssetTypeSpec. Every load type today registers LoadNoise.
    noise_cls: type[LoadNoise]
    # The one channel a load has: its contactor (D6a). A statement about the
    # TYPE, like AssetCapabilities.
    #
    # No `p_controllable`: a continuous served level is what a contactor cannot
    # honour. It returns only with a load type that can actually modulate.
    on_off_controllable: bool = True


_ASSET_TYPES: dict[str, AssetTypeSpec] = {}
_LOAD_TYPES: dict[str, LoadTypeSpec] = {}


def register_asset_type(spec: AssetTypeSpec) -> None:
    if spec.name in _ASSET_TYPES:
        raise ValueError(f"asset type {spec.name!r} is already registered")
    _ASSET_TYPES[spec.name] = spec


def register_load_type(spec: LoadTypeSpec) -> None:
    if spec.name in _LOAD_TYPES:
        raise ValueError(f"load type {spec.name!r} is already registered")
    _LOAD_TYPES[spec.name] = spec


def asset_type(name: str) -> AssetTypeSpec:
    """Look up a type, or refuse to start. A typo in a fleet's `type:` field is
    a config error and must not fall through to a default."""
    try:
        return _ASSET_TYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown asset type {name!r}. Known types: {', '.join(known_asset_types())}"
        ) from None


def load_type(name: str) -> LoadTypeSpec:
    try:
        return _LOAD_TYPES[name]
    except KeyError:
        raise ValueError(
            f"unknown load type {name!r}. Known types: {', '.join(known_load_types())}"
        ) from None


def known_asset_types() -> tuple[str, ...]:
    """Sorted, so error messages and any hash over them are deterministic (D10)
    regardless of registration order."""
    return tuple(sorted(_ASSET_TYPES))


def known_load_types() -> tuple[str, ...]:
    return tuple(sorted(_LOAD_TYPES))


# ---------------------------------------------------------------------------
# Today's types. The physics lives in sim/assets/ and sim/load.py; the classes
# here make each type constructible, and the plant steps exactly what
# `assemble()` builds from this table.
# ---------------------------------------------------------------------------

register_asset_type(
    AssetTypeSpec(
        capabilities=AssetCapabilities(
            name="pv",
            # Curtailment is a P setpoint below available (D4). Never
            # slack-bearing: an array cannot hold a bus.
            p_controllable=True,
            # A modern PV inverter is four-quadrant, bounded by pv.s_max_kva.
            # That rating equals p_ac_max_kw at this site, so the D3 coupling is
            # live: at full sun there is no reactive headroom.
            q_controllable=True,
            # Output follows the sun, so a setpoint is a CEILING: above available
            # runs free, below it curtails.
            resource_limited=True,
        ),
        asset_cls=PvAsset,
        spec_cls=PvSpec,
        # Soiling, module degradation and the rear-face gain: all move over a
        # site's life and none is on a nameplate, so all are beliefs rather than
        # ratings (ADR-0009 D4).
        believed_cls=PvPlantModel,
        # An array publishes one thing a policy can use: what its meter read
        # last interval. No SOC, and no forecast (ADR-0012 D2).
        noise_cls=AssetNoise,
    )
)

register_asset_type(
    AssetTypeSpec(
        capabilities=AssetCapabilities(
            name="genset",
            # q_controllable stays False: a rotating machine's reactive output
            # is a measurement, not a channel the dispatcher may command (D3).
            on_off_controllable=True,
            slack_bearing_capable=True,
        ),
        asset_cls=GensetAsset,
        spec_cls=GensetSpec,
        # The fuel curve drifts with injector fouling and ring wear. The minimum
        # stable load does not, and is here anyway: it is the only channel that
        # can carry a FORBIDDEN BAND to a dispatcher (ADR-0006).
        believed_cls=GensetPlantModel,
        # A set's `running` flag is TRUTH and does not cross, so what a policy
        # sees of a genset is its meter and nothing else.
        noise_cls=AssetNoise,
    )
)

register_asset_type(
    AssetTypeSpec(
        capabilities=AssetCapabilities(
            name="bess",
            # A grid-tie battery inverter is four-quadrant, bounded by
            # bess.s_max_kva. The clearest case of the ceiling/choice split:
            # setting the ceiling False to keep Q off would be a lie about the
            # hardware.
            q_controllable=True,
            slack_bearing_capable=True,
        ),
        asset_cls=BessAsset,
        spec_cls=BessSpec,
        # Capacity falls with cycling and round-trip efficiency falls as cell
        # impedance grows. NOT here: the rate limits and nameplate energy, which
        # are stamped on the pack and cross as AssetRatings.
        believed_cls=BessPlantModel,
        # The only type with a third channel: `soc_observed_kwh` is an ESTIMATE
        # the BMS publishes, not a reading. See BessNoise.
        noise_cls=BessNoise,
    )
)

register_asset_type(
    AssetTypeSpec(
        capabilities=AssetCapabilities(
            name="grid",
            # The grid is "just an asset" (D7), and IS P-controllable even
            # though the balance determines its output while it holds the bus:
            # D5 makes the setpoint ADVISORY, not forbidden, and the
            # commanded-versus-delivered delta is a primary diagnostic.
            p_controllable=True,
            slack_bearing_capable=True,
        ),
        asset_cls=GridAsset,
        # The grid's one parameter: the PCC transformer's apparent-power
        # rating, which GridAsset turns into its envelope like any other
        # asset's rating (D3). The tariff is NOT spec -- it is cost data, and
        # lives on the grid's own fleet entry as `cost_terms:` (D7).
        spec_cls=GridSpec,
        # Nothing to estimate: the transformer rating is stamped on it and the
        # tariff is declared cost data. Empty rather than omitted (ADR-0009 D2).
        believed_cls=EmptyPlantModel,
        # The PCC meter -- a revenue meter, in practice the best-calibrated
        # instrument on site.
        noise_cls=AssetNoise,
    )
)

# Priority and shed price are DECLARED, so a fixed load has nothing a dispatcher
# could be wrong about. Its FEEDER still has a meter.
register_load_type(
    LoadTypeSpec(
        name="fixed", load_cls=FixedLoad, believed_cls=EmptyPlantModel, noise_cls=LoadNoise
    )
)
