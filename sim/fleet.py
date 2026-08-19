"""Resolve a FleetSpec against the registry, and refuse bad ones. ADR-0004 D9.

Where the three sources meet: config declares CHOICES, `sim/registry.py` declares
what each type is CAPABLE of, and `sim/cost_terms.py` declares which cost terms
can be priced. Nothing downstream re-checks any of it.

The ADR's five rejections, plus a sixth:

    1. duplicate id                      config/schema.py, FleetSpec
    2. Q command to a P-only asset       Fleet.validate_commands()
    3. slack-bearing mark on an incapable
       asset                             Fleet.validate_slack_bearing()
    4. cost term the ledger cannot price build_fleet()
    5. command targeting a missing asset Fleet.validate_commands()
    6. more than one grid asset          _check_at_most_one_grid()

Rejection 5's converse -- a declared member that received no command -- is
checked at the end of validate_commands(). Rejection 4 checks the KIND only;
params, member support and sibling terms are checked when the models build.

Also where an asset's declared PARAMETERS are typed: `config/` cannot import
`sim/` and so cannot know that `pv` means `PvSpec`, but the registry can.

`assemble()` turns a validated Fleet into the constructed objects the plant
steps. `runner.simulate()` calls both.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel, ValidationError

from config.schema import CostTermSpec, FleetAssetEntry, FleetLoadEntry, FleetSpec, SiteConfig
from contracts import (
    AssetCommand,
    AssetView,
    CostParam,
    CostTerm,
    FleetView,
    LoadCommand,
    LoadView,
    PlantModel,
)

# ResolvedAsset is defined in sim/asset.py and ResolvedLoad in sim/load.py,
# not here, so that sim/assets/* and the load classes can name them in
# from_config without closing the cycle fleet -> registry -> assets/pv ->
# fleet. Re-exported for callers that reasonably expect the fleet module to
# hand out fleet members. See their docstrings for the full reasoning.
from sim.asset import Asset, KvaPriority, ResolvedAsset
from sim.cost_terms import IMPLEMENTED_COST_TERMS
from sim.load import Load, ResolvedLoad
from sim.registry import AssetTypeSpec, asset_type, load_type
from sim.scenario import ScenarioRow, availability_field

__all__ = ["AssembledFleet", "Fleet", "ResolvedAsset", "ResolvedLoad", "assemble", "build_fleet"]

# Assets and loads register different noise classes (AssetNoise and LoadNoise have
# no common base but pydantic's), and _resolve_noise must hand each builder back
# the concrete type it declared rather than a widened one.
_NoiseT = TypeVar("_NoiseT", bound=BaseModel)


def _resolve_choice(chosen: bool | None, capable: bool, asset_id: str, channel: str) -> bool:
    """A choice within a capability ceiling. None means "take the type's
    default"; True on an incapable type is a config error, never clamped."""
    if chosen is None:
        return capable
    if chosen and not capable:
        raise ValueError(
            f"asset {asset_id!r} declares {channel}=True but its type is not capable of it. "
            f"Capability is a fact about the equipment, not a per-site setting."
        )
    return chosen


def _resolve_kva_priority(chosen: str | None, q_controllable: bool, asset_id: str) -> KvaPriority:
    """Which of P and Q gives way outside the kVA circle (ADR-0004 D3).

    None takes the default. Declaring one on an asset with no commandable
    reactive channel is refused: there is no circle to prioritise within, so the
    declaration would be silently inert.
    """
    if chosen is None:
        return KvaPriority.P
    if not q_controllable:
        raise ValueError(
            f"asset {asset_id!r} declares kva_priority={chosen!r} but is not q_controllable. "
            f"An asset with no commandable reactive channel has no kVA circle to resolve, "
            f"so the setting would never apply."
        )
    return KvaPriority(chosen)


def _resolve_cost_terms(
    terms: tuple[CostTerm, ...], member_id: str, implemented: frozenset[str]
) -> tuple[CostTerm, ...]:
    """Rejection 4. A declared term with no implementation would cost zero
    forever and never say so."""
    for term in terms:
        if term.kind not in implemented:
            raise ValueError(
                f"{member_id!r} declares cost term {term.kind!r}, which the ledger cannot "
                f"price. Implemented: {', '.join(sorted(implemented))}. A term with no "
                f"implementation silently costs nothing."
            )
    return terms


def _freeze_param(value: object) -> CostParam:
    """One declared param value, deep-frozen into the CostParam shape.

    Lists become tuples and mappings read-only views, recursively, so no
    CostTerm aliases a live config object. An unrecognised type is REFUSED.
    """
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_param(v) for k, v in value.items()})
    if isinstance(value, Sequence):
        return tuple(_freeze_param(v) for v in value)
    raise ValueError(
        f"cost term param of type {type(value).__name__} is not a CostParam. "
        f"Params are scalars, sequences of params, or mappings of params."
    )


def _declared_terms(entries: tuple[CostTermSpec, ...]) -> tuple[CostTerm, ...]:
    """CostTermSpec (config) -> CostTerm (contracts), params carried whole and
    deep-frozen so no contract instance aliases a config object."""
    return tuple(
        CostTerm(kind=t.kind, params={k: _freeze_param(v) for k, v in t.params.items()})
        for t in entries
    )


@dataclass(frozen=True, kw_only=True)
class Fleet:
    """A validated fleet, in DECLARATION ORDER. ADR-0004 D10.

    Every iteration that sums must use these tuples rather than a dict or set:
    floating-point addition is not associative.
    """

    assets: tuple[ResolvedAsset, ...]
    loads: tuple[ResolvedLoad, ...]
    # The validated `believed:` blocks, keyed by member id and TOTAL over
    # `assets + loads`. ADR-0009 D2/D3.
    #
    # HERE RATHER THAN ON ResolvedAsset: a belief carried there would be readable
    # by every asset class, and from there by the plant, the ledger and the
    # oracle. `assemble()` copies this onto the FleetView and nowhere else.
    believed: Mapping[str, PlantModel]

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.assets)

    @property
    def load_ids(self) -> tuple[str, ...]:
        return tuple(ln.id for ln in self.loads)

    def asset(self, asset_id: str) -> ResolvedAsset:
        for a in self.assets:
            if a.id == asset_id:
                return a
        raise ValueError(
            f"no asset {asset_id!r} in the fleet. Declared: {', '.join(self.asset_ids)}"
        )

    def load(self, load_id: str) -> ResolvedLoad:
        for ln in self.loads:
            if ln.id == load_id:
                return ln
        raise ValueError(f"no load {load_id!r} in the fleet. Declared: {', '.join(self.load_ids)}")

    def validate_slack_bearing(self, asset_id: str) -> None:
        """Rejection 3. Only grid, BESS, genset or fuel cell may bear the slack.

        CAPABILITY only; D5's "exactly one per interval" holds by construction
        under a priority ORDER. Every entry is checked, not just the head.
        """
        asset = self.asset(asset_id)
        if not asset.slack_bearing_capable:
            raise ValueError(
                f"asset {asset_id!r} is marked slack-bearing but its type "
                f"{asset.capabilities.name!r} cannot form a bus (D5)."
            )

    def validate_commands(
        self,
        asset_commands: Mapping[str, AssetCommand],
        load_commands: Mapping[str, LoadCommand] | None = None,
    ) -> None:
        """Rejections 2 and 5, plus the other half of each: a channel commanded
        on an asset that does not accept it, a channel it accepts left
        uncommanded, a command naming a missing member, and a member never
        commanded. Absence is never a default (D4)."""
        for asset_id, cmd in asset_commands.items():
            asset = self.asset(asset_id)  # rejection 5

            # Every channel, both directions, one shape.
            for channel, commanded, declared in (
                ("p_setpoint_kw", cmd.p_setpoint_kw is not None, asset.p_controllable),
                ("q_setpoint_kvar", cmd.q_setpoint_kvar is not None, asset.q_controllable),
                ("on", cmd.on is not None, asset.on_off_controllable),
            ):
                if commanded and not declared:
                    raise ValueError(
                        f"command to {asset_id!r} carries {channel} but the asset does not "
                        f"declare that channel. Rejected, never ignored at run time (D3)."
                    )
                if declared and not commanded:
                    raise ValueError(
                        f"asset {asset_id!r} declares {channel} but its command carries none. "
                        f"An absent command is an error, not a default."
                    )

            if not any((asset.p_controllable, asset.q_controllable, asset.on_off_controllable)):
                raise ValueError(
                    f"asset {asset_id!r} declares no controllable channel and accepts no "
                    f"command, but one was issued to it."
                )

        # Loads, both directions, exactly as assets above (D6: loads are
        # first-class and symmetric to assets).
        for load_id, load_cmd in (load_commands or {}).items():
            load = self.load(load_id)  # rejection 5, load side
            commanded = load_cmd.on is not None
            if commanded and not load.on_off_controllable:
                raise ValueError(
                    f"command to load {load_id!r} carries on but the load does not declare "
                    f"that channel. A load with no contactor cannot be shed voluntarily; "
                    f"only physics may drop it (D6a)."
                )
            if load.on_off_controllable and not commanded:
                raise ValueError(
                    f"load {load_id!r} declares on but its command carries none. "
                    f"An absent command is an error, not a default."
                )

        # The MEMBERSHIP direction, the converse of rejection 5: everything above
        # iterates the COMMANDS and says nothing about a member never commanded
        # at all, which would otherwise die as a KeyError inside plant.step().
        #
        # Checked LAST, so a command carrying a bad channel still reports that
        # channel. Declaration order (D10) on the way out, since this names
        # members in an error string.
        uncommanded = [a.id for a in self.assets if a.id not in asset_commands]
        uncommanded += [ln.id for ln in self.loads if ln.id not in (load_commands or {})]
        if uncommanded:
            raise ValueError(
                f"{len(uncommanded)} fleet member(s) received no command at all: "
                f"{', '.join(uncommanded)}. Every declared member is commanded every "
                f"interval; an absent command is an error, not a default."
            )


def _resolve_spec(entry: FleetAssetEntry, type_spec: AssetTypeSpec) -> BaseModel | None:
    """Type this asset's declared parameters, or refuse. Both directions are
    errors: a spec on a type that takes none, and a spec missing fields."""
    if type_spec.spec_cls is None:
        if entry.spec:
            raise ValueError(
                f"asset {entry.id!r} declares a spec, but its type {entry.type!r} takes no "
                f"model parameters. Declared: {sorted(entry.spec)}. Refused rather than "
                f"ignored -- a silently discarded spec is a rating nobody knows was dropped."
            )
        return None

    try:
        return type_spec.spec_cls.model_validate(entry.spec)
    except ValidationError as exc:
        raise ValueError(
            f"asset {entry.id!r} (type {entry.type!r}) has an invalid spec: {exc}"
        ) from None


def _resolve_noise(
    member_kind: str, member_id: str, declared: Mapping[str, Any], noise_cls: type[_NoiseT]
) -> _NoiseT:
    """Type one member's `noise:` block, or refuse. ADR-0012 D2.

    No hand-rolled walk is needed, unlike `believed:`: these are pydantic models
    with `extra="forbid"` and no defaults, so `model_validate` refuses in both
    directions and range-checks the sigmas.
    """
    try:
        return noise_cls.model_validate(dict(declared))
    except ValidationError as exc:
        raise ValueError(f"{member_kind} {member_id!r} has an invalid noise block: {exc}") from None


def _build_plant_model(cls: type[Any], declared: Mapping[str, Any], path: str) -> Any:
    """One `believed:` block, typed against the parameters its type registers.

    Refuses in both directions; partial beliefs are not a thing. State all of the
    type's parameters, or `{}` for a type with none.

    RECURSIVE, since a believed fuel curve is a nested block (ADR-0009 D3).
    `path` accumulates so a refusal names the leaf, not the outermost block.

    VALUES are not checked here -- each contract type range-checks itself in
    `__post_init__`. This decides only which keys may be present.
    """
    hints = get_type_hints(cls)
    fields = dataclasses.fields(cls)
    expected = {f.name for f in fields}

    missing = sorted(expected - set(declared))
    if missing:
        raise ValueError(
            f"{path}: the believed block is missing {', '.join(missing)}. State every parameter "
            f"{cls.__name__} registers, or `{{}}` for a type that registers none -- a partial "
            f"belief is a plant model nobody wrote down."
        )
    alien = sorted(set(declared) - expected)
    if alien:
        raise ValueError(
            f"{path}: the believed block declares {', '.join(alien)}, which {cls.__name__} does not "
            f"register. Known: {', '.join(sorted(expected)) or '(none)'}. Refused rather than "
            f"ignored -- a silently dropped belief is a parameter somebody wrote down and "
            f"nobody knows was lost."
        )

    kwargs: dict[str, Any] = {}
    for f in fields:
        value = declared[f.name]
        field_type = hints[f.name]
        leaf = f"{path}.{f.name}"
        if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
            if not isinstance(value, Mapping):
                raise ValueError(f"{leaf}: expected a block of parameters, got {value!r}")
            kwargs[f.name] = _build_plant_model(field_type, value, leaf)
            continue
        if field_type is not float:
            # Every believed parameter is a float or a nested block. Any other
            # type is a deliberate extension of this walk, never a coercion.
            raise ValueError(
                f"{leaf}: {cls.__name__} declares it as {field_type}, which this builder has no "
                f"rule for. Extend _build_plant_model rather than coercing it."
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{leaf} is {value!r}, which is not a number.")
        kwargs[f.name] = float(value)
    return cls(**kwargs)


def _resolve_believed(
    member_kind: str, member_id: str, declared: Mapping[str, Any], believed_cls: type[PlantModel]
) -> PlantModel:
    """Type one member's `believed:` block, or refuse. ADR-0009 D2.

    The wrapper names the member on RANGE failures, which fire inside a contract
    type that has never heard of a fleet.
    """
    path = f"{member_kind} {member_id!r}"
    try:
        # Annotated rather than returned straight through: `_build_plant_model`
        # can only promise Any.
        model: PlantModel = _build_plant_model(believed_cls, declared, path)
    except ValueError as exc:
        message = str(exc)
        raise ValueError(message if message.startswith(path) else f"{path}: {message}") from None
    return model


def _build_asset(entry: FleetAssetEntry, implemented: frozenset[str]) -> ResolvedAsset:
    type_spec = asset_type(entry.type)  # unknown type refuses here
    caps = type_spec.capabilities
    return ResolvedAsset(
        id=entry.id,
        capabilities=caps,
        spec=_resolve_spec(entry, type_spec),
        p_controllable=caps.p_controllable,
        q_controllable=_resolve_choice(
            entry.q_controllable, caps.q_controllable, entry.id, "q_controllable"
        ),
        on_off_controllable=_resolve_choice(
            entry.on_off_controllable, caps.on_off_controllable, entry.id, "on_off_controllable"
        ),
        kva_priority=_resolve_kva_priority(
            entry.kva_priority,
            _resolve_choice(entry.q_controllable, caps.q_controllable, entry.id, "q_controllable"),
            entry.id,
        ),
        carry_in=dict(entry.carry_in),
        cost_terms=_resolve_cost_terms(_declared_terms(entry.cost_terms), entry.id, implemented),
        noise=_resolve_noise("asset", entry.id, entry.noise, type_spec.noise_cls),
    )


def _build_load(entry: FleetLoadEntry, implemented: frozenset[str]) -> ResolvedLoad:
    type_spec = load_type(entry.type)  # unknown type refuses here
    return ResolvedLoad(
        id=entry.id,
        type_name=entry.type,
        priority=entry.priority,
        carry_in=dict(entry.carry_in),
        cost_terms=_resolve_cost_terms(_declared_terms(entry.cost_terms), entry.id, implemented),
        on_off_controllable=type_spec.on_off_controllable,
        noise=_resolve_noise("load", entry.id, entry.noise, type_spec.noise_cls),
    )


@dataclass(frozen=True, kw_only=True)
class AssembledFleet:
    """A CONSTRUCTED fleet: the frozen asset/load objects the plant will step,
    plus the one FleetView the dispatcher is handed at construction.

    Declaration order throughout (D10). The assets and the view are built in
    one pass so they cannot disagree about ids, order, or resolved channels --
    the view is derived from the constructed objects, not re-read from config.
    """

    assets: tuple[Asset, ...]
    loads: tuple[Load, ...]
    view: FleetView

    def required_scenario_fields(self) -> Mapping[str, tuple[str, ...]]:
        """Every exogenous field this fleet needs -> the members that need it.

        Keyed this way round so a refusal can name WHO wanted a missing column.
        Sorted within each field, so the message is deterministic.
        """
        needed: dict[str, list[str]] = {}
        # Walked separately: Asset and Load share these members by convention,
        # not through a common base (D1/D6 keep them symmetric but distinct).
        for asset in self.assets:
            for field_name in asset.required_scenario_fields():
                needed.setdefault(field_name, []).append(asset.id)
        for load in self.loads:
            for field_name in load.required_scenario_fields():
                needed.setdefault(field_name, []).append(load.id)
        return {name: tuple(sorted(ids)) for name, ids in sorted(needed.items())}

    def validate_scenario_row(self, row: ScenarioRow) -> None:
        """Refuse a scenario that cannot feed this fleet. ADR-0004 D1.

        Checked against a real ROW, not `ScenarioRow`'s type: what a row PROVIDES
        is its field names plus the ids in its `available` mapping.

        Runs once, on the first interval; rows are homogeneous by construction.
        Extra columns are ignored, so one CSV can feed more than one fleet.
        """
        provided = {f.name for f in dataclasses.fields(row) if f.name != "available"}
        provided |= {availability_field(member_id) for member_id in row.available}

        missing = {
            name: members
            for name, members in self.required_scenario_fields().items()
            if name not in provided
        }
        if missing:
            detail = "; ".join(
                f"{name!r} (needed by {', '.join(members)})" for name, members in missing.items()
            )
            raise ValueError(
                f"the scenario does not provide {len(missing)} field(s) this fleet requires: "
                f"{detail}. It provides: {', '.join(sorted(provided))}."
            )


def assemble(fleet: Fleet, config: SiteConfig) -> AssembledFleet:
    """Construct every fleet member through its registered class, or refuse.

    Where construction-time validation completes: build_fleet() typed the specs,
    from_config here narrows them and builds the frozen objects.

    Also the CAPABILITY half of the D5 slack check (rejection 3), over the WHOLE
    priority list rather than just the head.
    """
    for slack_candidate in config.slack_priority:
        fleet.validate_slack_bearing(slack_candidate)
    assets: list[Asset] = []
    asset_views: list[AssetView] = []
    for resolved in fleet.assets:
        type_name = resolved.capabilities.name
        type_spec = asset_type(type_name)
        if type_spec.asset_cls is None:
            raise ValueError(
                f"asset type {type_name!r} is registered without a class; it cannot be "
                f"constructed. Every shipped type has one since Stage D."
            )
        asset = type_spec.asset_cls.from_config(resolved, config.dt_minutes)
        assets.append(asset)
        asset_views.append(
            AssetView(
                id=asset.id,
                type_name=type_name,
                # The RESOLVED choices off the constructed object, not the
                # type ceiling: what the dispatcher may actually command.
                p_controllable=asset.p_controllable,
                q_controllable=asset.q_controllable,
                on_off_controllable=asset.on_off_controllable,
                slack_bearing_capable=asset.slack_bearing_capable,
                resource_limited=asset.resource_limited,
                ratings=asset.ratings,
                cost_terms=resolved.cost_terms,
            )
        )

    loads: list[Load] = []
    load_views: list[LoadView] = []
    for resolved_load in fleet.loads:
        load_spec = load_type(resolved_load.type_name)
        if load_spec.load_cls is None:
            raise ValueError(
                f"load type {resolved_load.type_name!r} is registered without a class; "
                f"it cannot be constructed."
            )
        load = load_spec.load_cls.from_config(resolved_load, config.dt_minutes)
        loads.append(load)
        load_views.append(
            LoadView(
                id=load.id,
                type_name=resolved_load.type_name,
                priority=load.priority,
                on_off_controllable=load.on_off_controllable,
                cost_terms=resolved_load.cost_terms,
            )
        )

    return AssembledFleet(
        assets=tuple(assets),
        loads=tuple(loads),
        view=FleetView(
            assets=tuple(asset_views),
            loads=tuple(load_views),
            # The ONLY consumer of `Fleet.believed` (ADR-0009 D3), which is what
            # keeps the belief off every constructed asset, the plant and the
            # ledger.
            believed=fleet.believed,
            # The dispatcher's copy of the resolution order, so a policy can work
            # out who holds the bus without a per-interval channel.
            slack_priority=tuple(config.slack_priority),
            # The control period, declared once rather than assumed by every
            # policy (ADR-0013, ADR-0015).
            dt_minutes=config.dt_minutes,
            dispatch_interval_multiple=config.dispatch_interval_multiple,
        ),
    )


def _check_at_most_one_grid(assets: tuple[ResolvedAsset, ...]) -> None:
    """Rejection 6. Exactly one grid connection is supportable.

    Three things are missing behind it: only one asset bears the slack at a time,
    the scenario price pair is site-level, and each entry declaring its own
    tariff bills the demand-charge floor twice.

    A guard standing in for a model. Restricts the GRID only -- two batteries or
    two PV arrays run correctly.
    """
    grids = [a.id for a in assets if a.capabilities.name == "grid"]
    if len(grids) > 1:
        raise ValueError(
            f"fleet declares {len(grids)} grid assets ({', '.join(grids)}); exactly one is "
            f"supportable. They would share one price pair from the scenario, only one could "
            f"bear the slack, and each would bill its own demand charge. Refused rather than "
            f"run, because it produces a plausible WRONG bill rather than an error."
        )


def build_fleet(spec: FleetSpec, implemented: frozenset[str] | None = None) -> Fleet:
    """Resolve and validate a fleet, or refuse to start.

    `implemented` is injectable so a test can pin the term library. Declaration
    order is preserved exactly (D10).
    """
    terms = IMPLEMENTED_COST_TERMS if implemented is None else implemented
    assets = tuple(_build_asset(a, terms) for a in spec.assets)
    _check_at_most_one_grid(assets)
    # The believed blocks, in declaration order and TOTAL over the fleet (D10,
    # ADR-0009 D3). Built from the raw entries rather than from the resolved
    # members on purpose: this is the only place the two ever meet, and it keeps
    # the belief off everything downstream constructs.
    believed = {
        **{
            a.id: _resolve_believed("asset", a.id, a.believed, asset_type(a.type).believed_cls)
            for a in spec.assets
        },
        **{
            ln.id: _resolve_believed("load", ln.id, ln.believed, load_type(ln.type).believed_cls)
            for ln in spec.loads
        },
    }
    return Fleet(
        assets=assets,
        loads=tuple(_build_load(ln, terms) for ln in spec.loads),
        believed=MappingProxyType(believed),
    )
