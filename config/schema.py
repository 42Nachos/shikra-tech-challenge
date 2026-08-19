"""The site config schema: what a fleet DECLARES, and the loader that reads it.
Validate hard; refuse to start on an invalid config. See Module 2.

STRUCTURE HERE, PHYSICS IN `sim/`. This module validates the SHAPE of a site
YAML -- that every member has an id, a type, and the four blocks it must
declare. What those blocks MEAN is the registry's knowledge, so `spec:`,
`believed:`, `noise:` and cost-term `params:` are raw mappings here and are
typed by `sim/fleet.py` against the classes `sim/registry.py` names.

That split is enforced: `config/` may import nothing internal
(`tools/check_boundary.py`). It keeps a new asset type to one file in
`sim/assets/` plus one registration, rather than an edit here as well
(ADR-0004 D1).

The cost, stated so it is not discovered: a config is validated in TWO calls,
`load_site_config()` then `build_fleet()`. A malformed spec or tariff is caught
by the second -- still startup, still before any interval runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CostTermSpec(BaseModel):
    """One declared cost term. ADR-0004 D7: data, never a function.

    `kind` selects an implementation from the ledger's library and, separately,
    from the dispatcher's own. Both sides read the same declaration and neither
    reads the other's code -- their disagreement is the measured cost-model
    error (CLAUDE.md). A `kind` the ledger cannot price is refused by
    `sim/fleet.py`, where config meets the registry.

    `params` is raw: what a `fuel` term's params must contain is knowledge about
    pricing fuel, and typing it here would put the LEDGER's reading of a
    parameter into the one module `dispatch/`'s independent library also reads.
    `sim/ledger.py` validates each term's params against that term's own model.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class FleetAssetEntry(BaseModel):
    """One asset in the fleet, declaring four blocks that answer four questions:

        spec:      what the equipment IS          (ADR-0003)
        carry_in:  its state at k=0
        believed:  what a POLICY thinks it is     (ADR-0009 D1)
        noise:     how well anyone can SEE it     (ADR-0012 D1)

    `believed:`, `noise:` and `cost_terms:` are MANDATORY with no default.
    Absence is refused rather than taken as empty, because a member whose block
    was forgotten would otherwise be indistinguishable from one that correctly
    has none, and a member added later would inherit silence. A member with
    nothing to declare writes `{}` or `[]`, which says the same thing out loud.

    `cost_terms:` earns that for a reason of its own: an omitted one made the
    member FREE, and free is the most flattering thing a cost model can say
    about an asset. `pv_1` genuinely costs nothing to run and now declares `[]`
    to say so.

    `spec:` and `carry_in:` default to empty instead, because a type whose model
    takes no parameters and holds no state is legitimate -- and that is no
    licence to omit, since a spec class with required fields rejects `{}` naming
    them.

    Per-entry rather than per-type, so two PV arrays in one fleet can have
    different nameplates and two batteries different starting SOCs.

    Capability comes from the registry; the two overrides below are the site's
    CHOICE within it, defaulting to None meaning "take the type's default"
    rather than to a value that could silently disagree.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    spec: dict[str, Any] = Field(default_factory=dict)
    # A run's starting SOC changes its cost, so it lives inside what the config
    # hash covers rather than as a literal in runner.py (CLAUDE.md).
    carry_in: dict[str, float] = Field(default_factory=dict)
    believed: dict[str, Any]
    noise: dict[str, Any]
    cost_terms: tuple[CostTermSpec, ...]
    q_controllable: bool | None = None
    on_off_controllable: bool | None = None
    # How a command outside the inverter's kVA circle is resolved (ADR-0004 D3):
    # `p` holds real power, `q` holds reactive, `scale` shrinks both at constant
    # power factor. None means "take the type's default", like the two above.
    #
    # A string rather than sim/'s `KvaPriority`, because config/ imports nothing
    # internal (ADR-0014 D2); sim/fleet.py converts it.
    kva_priority: Literal["p", "q", "scale"] | None = None


class FleetLoadEntry(BaseModel):
    """One load. Same declaration contract as `FleetAssetEntry`, because ADR-0004
    D6's "loads are first-class and symmetric to assets" is a rule about
    declaration and not only about commands -- including the mandatory
    `cost_terms:`, where an omitted `unserved` term made shedding free.

    `priority` and the shed price answer different questions and are deliberately
    not collapsed (D6): one is an ORDERING, the other a PRICE, and a load that is
    expensive to restart should be shed later than an equally-priced one that is
    not.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    # LOWER = more critical = shed LAST. Asserted, never inferred: a flipped
    # convention sheds the most critical load first, costs the maximum every
    # interval, and reads as a bad scenario rather than a sign bug (D6a).
    priority: int = Field(ge=0)
    carry_in: dict[str, float] = Field(default_factory=dict)
    believed: dict[str, Any]
    noise: dict[str, Any]
    cost_terms: tuple[CostTermSpec, ...]


def _refuse_missing_block(data: Any, block: str, why: str) -> Any:
    """Refuse a fleet member that omits a mandatory block, NAMING THE MEMBER.

    Runs before field validation, which is the whole point. The field is
    required, so a missing one already fails -- but it fails as
    `fleet.assets.1.believed`, an INDEX. Every other rejection in this file names
    the member, and the id is what somebody edits. A missing required field never
    reaches a `mode="after"` validator, so naming it means reading the raw
    mapping here.
    """
    if not isinstance(data, dict):
        return data
    for group in ("assets", "loads"):
        for entry in data.get(group) or ():
            if isinstance(entry, dict) and block not in entry:
                member = entry.get("id", "<unnamed>")
                raise ValueError(f"fleet member {member!r} declares no `{block}:` block. {why}")
    return data


class FleetSpec(BaseModel):
    """The fleet, as an ORDERED list. ADR-0004 D10.

    Order is not cosmetic. Once the plant iterates and sums, iteration order is
    part of "same config + same seed = bit-identical output", because floating
    point addition is not associative. These are tuples, populated from YAML
    sequence order, and never dicts or sets -- a reordered YAML file is a
    behaviour change and should be reviewed as one.
    """

    model_config = ConfigDict(frozen=True)

    assets: tuple[FleetAssetEntry, ...] = Field(min_length=1)
    loads: tuple[FleetLoadEntry, ...] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _check_beliefs_are_declared(cls, data: Any) -> Any:
        """ADR-0009 D2."""
        return _refuse_missing_block(
            data,
            "believed",
            "Every member states what the dispatcher is assumed to know about it, and a "
            "type with nothing to estimate writes `believed: {}`. Absence is refused "
            "rather than taken as empty -- forgotten and correctly-empty must not look "
            "the same (ADR-0009 D2).",
        )

    @model_validator(mode="before")
    @classmethod
    def _check_noise_is_declared(cls, data: Any) -> Any:
        """ADR-0012 D1. Its own validator rather than folded into the one above:
        they enforce two decisions of two ADRs, and the day one is relaxed the
        other must not have to be disentangled from it first."""
        return _refuse_missing_block(
            data,
            "noise",
            "Every member states what its instruments do to the truth on the way to "
            "dispatch/, and a perfect sensor is declared by writing its sigmas as 0.0. "
            "Absence is refused rather than taken as noiseless -- forgotten and "
            "deliberately-perfect must not look the same (ADR-0012 D1).",
        )

    @model_validator(mode="before")
    @classmethod
    def _check_cost_terms_are_declared(cls, data: Any) -> Any:
        """Its own validator for the same reason the two above are separate."""
        return _refuse_missing_block(
            data,
            "cost_terms",
            "Every member states what it is charged for, and one charged for nothing "
            "writes `cost_terms: []`. Absence is refused rather than taken as empty -- an "
            "omitted list makes the member FREE, which is the most flattering thing a "
            "cost model can say about an asset.",
        )

    @model_validator(mode="after")
    def _check_ids_are_unique(self) -> Self:
        """Across assets AND loads: ids key every keyed contract under D9, so a
        collision would silently merge two fleet members' commands and costs."""
        seen: set[str] = set()
        for member_id in [a.id for a in self.assets] + [ln.id for ln in self.loads]:
            if member_id in seen:
                raise ValueError(f"duplicate fleet id {member_id!r}: ids must be unique")
            seen.add(member_id)
        return self

    @model_validator(mode="after")
    def _check_priorities_are_unique(self) -> Self:
        """D6a. "Two loads, same priority, only one can be dropped to close the
        balance" is a real interval; without a deterministic ordering the shed
        is nondeterministic and bit-identical reproducibility is gone (D10)."""
        priorities = [ln.priority for ln in self.loads]
        if len(set(priorities)) != len(priorities):
            dupes = sorted({p for p in priorities if priorities.count(p) > 1})
            raise ValueError(
                f"load priorities must be unique, got repeats at {dupes}. Priority is the "
                f"involuntary-shed ORDER; ties make the shed nondeterministic (D6a/D10)."
            )
        return self


class SiteConfig(BaseModel):
    """A whole site. Only what is genuinely site-wide lives here -- which site,
    how long an interval is, and who holds the bus. Everything about a MEMBER is
    declared on its fleet entry (ADR-0004)."""

    model_config = ConfigDict(frozen=True)

    site: str
    # THE PLANT STEP: how wide an interval the plant integrates over, and how
    # often an observation reaches the dispatcher.
    dt_minutes: int = Field(gt=0)
    # THE CONTROL RATE, in plant steps. The dispatcher re-optimises every this
    # many intervals and holds its commands in between (a zero-order hold), so
    # its control period is `dt_minutes * dispatch_interval_multiple`.
    #
    # 1 means the policy decides on every interval, which is what every run
    # before this existed did. A real EMS samples fast and re-optimises slowly,
    # and the two rates were previously the same number with nothing able to say
    # otherwise.
    #
    # NO DEFAULT. A run's control rate changes what a policy can do and what its
    # commands mean, so it is a property of the experiment that has to be
    # declared rather than inherited -- the same rule as `believed:` and
    # `noise:`, one level up.
    dispatch_interval_multiple: int = Field(ge=1)
    # The assets that may have their real power determined by the balance
    # (ADR-0004 D5), best first. Each interval the plant takes the first entry
    # that can actually hold the bus -- available in the scenario, and not
    # commanded off -- so a grid outage hands the bus down the list rather than
    # leaving the site with no primary slack. The plant SELECTS one, it never counts marks, so
    # zero and two are unrepresentable.
    #
    # ORDER IS BEHAVIOUR (D10), like the fleet's: reordering these lines changes
    # which asset carries the site through an outage, and is a reviewed change
    # rather than a tidy-up.
    slack_priority: tuple[str, ...] = Field(min_length=1)
    fleet: FleetSpec

    @model_validator(mode="after")
    def _check_slack_priority(self) -> Self:
        """The id half of the D5 check, over the whole list. Whether each TYPE
        may bear the slack needs the registry, so that half lives in
        sim/fleet.py and runs at assembly; ids pointing at nothing, and repeats,
        are pure config and refuse here.

        A repeat is refused rather than ignored: the second occurrence can never
        be reached, so it is either a typo for a different asset or a belief
        about failover that the list does not actually express.
        """
        declared = [a.id for a in self.fleet.assets]
        unknown = [i for i in self.slack_priority if i not in declared]
        if unknown:
            raise ValueError(
                f"slack_priority names {len(unknown)} asset(s) that are not declared: "
                f"{', '.join(unknown)}. Declared: {declared}"
            )
        if len(set(self.slack_priority)) != len(self.slack_priority):
            raise ValueError(
                f"slack_priority repeats an asset: {list(self.slack_priority)}. A repeat is "
                f"unreachable, so it is a typo or a failover belief the list does not express."
            )
        return self


def load_site_config(path: str | Path, carry_in_overlay: str | Path | None = None) -> SiteConfig:
    """Load and hard-validate a site YAML. Refuses to start on an invalid config.

    A `carry_in_overlay` may set the INITIAL STATE of fleet members and nothing
    else. The site config stays authoritative for every model parameter, price
    and capability; only `carry_in:` blocks are replaced.
    """
    with Path(path).open() as f:
        data = yaml.safe_load(f)
    config = SiteConfig.model_validate(data)
    if carry_in_overlay is None:
        return config
    return apply_carry_in_overlay(config, load_carry_in_overlay(carry_in_overlay))


# An overlay sets initial state and nothing else. Any other key is REFUSED
# rather than ignored: one that quietly failed to change a spec would be a
# parameter somebody wrote down and nobody knows was dropped.
_OVERLAY_KEYS = frozenset({"carry_in", "synth"})


def load_carry_in_overlay(path: str | Path) -> Mapping[str, Mapping[str, float]]:
    """Read a carry-in overlay YAML: `carry_in: {<member_id>: {<key>: value}}`.

    Structure only. Whether a member exists is checked in
    apply_carry_in_overlay(), and whether its keys are the right ones for its
    type is checked where it always was -- the asset's own initial_state().
    """
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a carry-in overlay must be a mapping, got {type(raw).__name__}")

    unknown = sorted(set(raw) - _OVERLAY_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: a carry-in overlay may only set {sorted(_OVERLAY_KEYS)}, but it declares "
            f"{unknown}. Model parameters, prices and capabilities belong to the site config; "
            f"an overlay that could change them would be a second site config."
        )

    blocks = raw.get("carry_in") or {}
    if not isinstance(blocks, dict):
        raise ValueError(f"{path}: `carry_in:` must be a mapping of member id -> block")

    overlay: dict[str, dict[str, float]] = {}
    for member_id, block in blocks.items():
        if not isinstance(block, dict):
            raise ValueError(f"{path}: carry_in for {member_id!r} must be a mapping, got {block!r}")
        for key, value in block.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{path}: carry_in {member_id!r}.{key} is {value!r}, which is not a number."
                )
        overlay[str(member_id)] = {str(k): float(v) for k, v in block.items()}
    return overlay


def apply_carry_in_overlay(
    config: SiteConfig, overlay: Mapping[str, Mapping[str, float]]
) -> SiteConfig:
    """Return a new SiteConfig with the named members' carry_in REPLACED.

    Replaced, not merged: a half-overridden carry-in block is a starting state
    nobody wrote down. The result is re-validated from scratch, so an overlay
    cannot slip a config past the validators the base file had to pass -- and it
    moves `config_hash`, which is correct, since a different starting SOC is a
    different run and must invalidate a golden captured without it.
    """
    if not overlay:
        return config

    declared = {a.id for a in config.fleet.assets} | {ln.id for ln in config.fleet.loads}
    unknown = sorted(set(overlay) - declared)
    if unknown:
        raise ValueError(
            f"carry-in overlay names {len(unknown)} member(s) that the fleet does not declare: "
            f"{', '.join(unknown)}. Declared: {', '.join(sorted(declared))}."
        )

    data = config.model_dump()
    for entry in data["fleet"]["assets"] + data["fleet"]["loads"]:
        if entry["id"] in overlay:
            entry["carry_in"] = dict(overlay[entry["id"]])
    return SiteConfig.model_validate(data)
