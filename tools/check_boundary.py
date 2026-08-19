"""Fail if sim/ imports dispatch/ or vice versa, or if contracts.py imports
anything internal. The core architectural rule.

The second check is what makes the ADR-0004 split load-bearing. contracts.py is
the only module dispatch/ may import, so membership of it is a permission:
AssetState, AssetEnvelope, AssetDelivered, LoadState and LoadDemand are kept in
sim/ precisely so a dispatcher cannot name them. If contracts.py could import
sim/, it could re-export any of them and the boundary would quietly become a
matter of who is paying attention. It cannot, and this is the check that says
so out loud.

`synth/` joined the rules at ADR-0005. It generates exogenous TRUTH -- the
irradiance and load a scenario is made of, and the event windows that make a
case adversarial -- so a dispatcher importing it could read the answer sheet
directly, and read it BEFORE the interval rather than one interval late. That is
the same class of constraint as not reading PlantState, and ADR-0005 D2 makes it
absolute.

WHERE THE ORACLE SITS, since ADR-0005 D2 guessed wrong and this docstring
repeated the guess. The oracle is NOT a dispatcher and receives no `Observation`:
it lives in metrics/, reads PlantState and the whole future scenario, and is
scanned by none of the rules below -- metrics/ is in neither RULES nor any
package this walks. That is the wall working rather than a hole in it. The wall
protects DISPATCHERS; it says nothing about analysing a finished run, and the
oracle is the yardstick rather than a contestant (ADR-0008).

What crosses BACK is frozen data only: a precomputed `Commands` schedule replayed
by dispatch/playback.py, which imports contracts and nothing else and is policed
here like any other policy.

`sim/` may not import `synth/` either, and that direction is not in the ADR. It
is here because the testbed is built once and frozen: a generator is a tool that
FEEDS the plant, so a plant reaching back into it would make the frozen half
depend on the churning one. synth/ -> sim/ stays open, which is what lets a
generated scenario be validated by the loader's own invariants (ADR-0005 D4).

`fitting/` joined at ADR-0010 D1, on both counts. It holds the true asset models,
the true measured record and a SEARCH over the true parameters, so a dispatcher
reaching into it would read the answer sheet -- the same constraint as synth/,
for the same reason, and absolute. It is also a tool, so `sim/` may not reach
back into it either; fitting/ -> sim/ stays open, which is the whole point (D2:
a fit drives the real asset rather than a paraphrase of it).

`fitting/ -> dispatch/` is forbidden as well. Nothing would obviously go wrong
today, but a fitter that could import a policy is one step from fitting a policy's
parameters against the truth, which is the experiment eating itself.

WHY `studies/` HAS NO RULE OF ITS OWN, and it is a decision rather than an
oversight. An earlier draft of ADR-0011 asked for one -- `"studies":
("dispatch",)` -- and it CANNOT BE WRITTEN. `studies/regression/golden.py`
imports `dispatch.base`, `dispatch.null` and `dispatch.scripted` structurally,
because `GoldenCase.dispatcher` is typed `Callable[[FleetView], Dispatcher]`;
`test_oracle_bound.py` imports `dispatch.catalog` to enumerate the policies it
bounds. A golden harness whose entire job is RUNNING dispatchers must be able to
name them.

BUT NOT BEING A KEY IS A DIFFERENT THING FROM NOT BEING FORBIDDEN, and this file
conflated them three times. `metrics` was missing from `dispatch`'s forbidden
TUPLE until ADR-0013 and `config` until ADR-0014; `studies` is the third and is
added here. `from studies.regression.golden import ...` inside a policy passed
this check, and `studies/` holds the expected outputs of every golden case --
and, since ADR-0018, the scoring machinery and the held-out case beside them.
The two directions are independent and both are now stated: what may import a
dispatcher, and what a dispatcher may import.

RELATIVE IMPORTS ARE REFUSED OUTRIGHT inside every package above. Every other
rule here resolves an import to its TOP-LEVEL package name, and a relative
import has no top-level name in its source text: to know that `from ..sim
import plant` reaches `sim`, you have to know where the file sits and
re-implement the interpreter's resolution. A checker that gets that subtly
wrong is a hole with a passing test over it -- so the rule is that the name
must be absolute, rather than that this script must be cleverer. Nothing in the
repo uses one today, so the ban costs nothing now; the day something wants one,
it is a reviewed change here rather than a silent bypass of every rule above.

TWO PROFILES SINCE ADR-0018, and the second is never the default.

    --profile default      every rule above. What pre-commit and CI run.
    --profile submission   the above, PLUS a ban inside dispatch/ on the modules
                           by which a policy reaches the filesystem, the network
                           or the import machinery.

The submission profile is layer ONE of two, and it is worth being clear about
what it cannot do. It reads source, so a module name assembled at run time is
invisible to it; the audit hook the scorer installs before `simulate()` sees the
resolution and nothing else. Neither layer subsumes the other. The attack the
pair closes is specific: `check_boundary.py` constrains IMPORTS, and the
held-out scenario CSV is a file on disk at scoring time -- without the hook, a
submission reads the future with `open()` and violates no import rule.

WHY IT MUST NOT BECOME THE DEFAULT: ordinary development in `dispatch/` would
lose `sys`, and a rule that fires during everyday work gets switched off.

Runs in pre-commit and CI. See CLAUDE.md, ADR-0002, ADR-0005 and ADR-0018.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# package -> packages it may not import. A TUPLE per package, not one string:
# `dispatch` is now walled off from several sides, and a single-value mapping
# could only ever express one of them.
RULES: dict[str, tuple[str, ...]] = {
    "sim": ("dispatch", "synth", "fitting"),
    # `metrics` joined this tuple at ADR-0013, and it should have been here since
    # ADR-0008. `metrics/oracle_model.py` imports `sim/` and builds the MILP over
    # the TRUE specs; a dispatcher importing `metrics` would reach every asset
    # class, every spec and the oracle's own solved plan in one line. Nothing did
    # -- but nothing stopped it either, and ADR-0013 hands `dispatch/` a solver
    # on the argument that this check is what keeps the true models out of reach.
    # That argument needed this entry to be true.
    #
    # `config` is the SAME OMISSION one more time, and it was the worst of the
    # three. ADR-0009 D3 says in as many words that "both directions of the
    # separation hold structurally and neither needs a lint rule: dispatch/
    # cannot import config.schema, so it can only ever see what FleetView chose
    # to publish". It could. `config/schema.py` imports nothing but yaml and
    # pydantic, so it was freely importable, and two lines would have handed a
    # policy every true `spec:` (ADR-0009's own answer sheet), every true
    # `noise:` sigma (ADR-0012 D5's), and the true carry-in SOC. Nothing in
    # dispatch/ ever did -- and, again, nothing stopped it.
    #
    # `studies` is the omission a THIRD time, added at ADR-0018. It holds every
    # golden's expected output, and now the scorers and the held-out case as
    # well; a policy that could import it could read what it is about to be
    # measured against.
    "dispatch": ("sim", "synth", "fitting", "metrics", "config", "studies"),
    "synth": ("dispatch",),
    "fitting": ("dispatch",),
    # `config/` imports NOTHING internal, and until now that was a docstring
    # claim rather than a check. What enforced it in practice was a cycle --
    # `sim/` imports `config/` in ten modules, so `config -> sim` would not have
    # loaded -- which is a mechanism, not a rule, and it stopped being one when
    # the spec and noise classes moved into `sim/` beside their assets. The
    # design reason survives the cycle and is what this entry pins: a site YAML
    # is validated for STRUCTURE here and for PHYSICS by the registry, so a new
    # asset type stays one file in `sim/assets/` plus one registration and never
    # an edit to `config/` (ADR-0004 D1).
    "config": ("sim", "dispatch", "synth", "fitting", "metrics"),
}

# contracts.py imports nothing internal (CLAUDE.md). Everything the repo owns.
INTERNAL = frozenset(
    {"sim", "dispatch", "synth", "config", "studies", "tools", "runner", "metrics", "fitting"}
)

# ADR-0018, layer one. Banned INSIDE dispatch/ under `--profile submission`
# only: the filesystem, the network, and the import machinery. A dispatcher is a
# pure function from Observation to Commands (ADR-0002), so none of this is
# reachable by anything a policy legitimately does -- which is what makes the
# ban free to impose rather than a restriction to work around.
SUBMISSION_BANNED_MODULES = frozenset(
    {
        "ctypes",
        "importlib",
        "io",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "socket",
        "subprocess",
        "sys",
        "urllib",
    }
)

# `builtins` is NOT in ADR-0018's list and is here deliberately. Without it the
# three names below are one `import builtins` away from being reachable again,
# statically and in plain sight -- which is a hole this layer can close, unlike
# the runtime-assembled names that are the hook's job.
SUBMISSION_BANNED_MODULES_EXTRA = frozenset({"builtins"})

# Detected as NAME LOADS rather than as calls, so `run = eval` is caught as well
# as `eval(...)`. Shadowing a builtin with a local of the same name would false
# positive; ruff's A001 already refuses that, so the stricter reading is free.
SUBMISSION_BANNED_BUILTINS = frozenset({"__import__", "eval", "exec"})


def imported_roots(path: Path) -> list[str]:
    """Top-level package of every ABSOLUTE import in a module, in source order.

    `node.level == 0` is what makes it absolute. Relative imports are not
    ignored by that guard -- they carry no top-level name to return here, and
    `relative_imports` below refuses them by their form instead.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.append(node.module.split(".")[0])
    return roots


def relative_imports(path: Path) -> list[str]:
    """Every relative import in a module, quoted back with its line number.

    `node.module` is None for a bare `from . import x`, so it is rendered as the
    dots alone rather than as the string "None".
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"line {node.lineno}: `from {'.' * node.level}{node.module or ''} import ...`"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]


def banned_builtin_uses(path: Path) -> list[str]:
    """Every mention of a banned builtin, quoted back with its line number."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"line {node.lineno}: `{node.id}`"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in SUBMISSION_BANNED_BUILTINS
    ]


def offending_imports(path: Path, forbidden: str) -> list[str]:
    return [r for r in imported_roots(path) if r == forbidden]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        choices=("default", "submission"),
        default="default",
        help=(
            "'submission' adds ADR-0018's filesystem/network/import bans inside "
            "dispatch/. Never make it the default: ordinary development there loses sys."
        ),
    )
    args = parser.parse_args()

    failures = []
    for pkg, forbidden_pkgs in RULES.items():
        for py in Path(pkg).rglob("*.py"):
            # Once per FILE, and OUTSIDE the forbidden loop below. This rule is
            # about the import's FORM rather than about what it reaches, so
            # pairing it with each forbidden package would report one line
            # three times for sim/ and once for synth/ -- the same finding,
            # printed a number of times that depends on an unrelated table.
            for rel in relative_imports(py):
                failures.append(
                    f"{py} {rel} -- relative imports are refused inside {pkg}/; "
                    f"name the package absolutely so this checker can resolve it"
                )
            for forbidden in forbidden_pkgs:
                for imp in offending_imports(py, forbidden):
                    failures.append(f"{py} imports '{imp}' (forbidden: {pkg} -> {forbidden})")

    for root in imported_roots(Path("contracts.py")):
        if root in INTERNAL:
            failures.append(
                f"contracts.py imports '{root}' -- it must import nothing internal, "
                f"or truth-side types could be re-exported to dispatch/"
            )

    # ADR-0018 layer one, and ONLY under the submission profile. The whole of
    # dispatch/ is scanned rather than one submission path: a submission lands in
    # dispatch/submissions/, the rest of dispatch/ is already clean, and a
    # checker aimed at one file would miss a helper module beside it.
    banned_modules = SUBMISSION_BANNED_MODULES | SUBMISSION_BANNED_MODULES_EXTRA
    if args.profile == "submission":
        for py in Path("dispatch").rglob("*.py"):
            for imp in sorted(set(imported_roots(py)) & banned_modules):
                failures.append(
                    f"{py} imports '{imp}' -- forbidden under --profile submission. A "
                    f"dispatcher is a pure function from Observation to Commands; it needs "
                    f"no filesystem, no network and no import machinery."
                )
            for use in banned_builtin_uses(py):
                failures.append(
                    f"{py} {use} -- forbidden under --profile submission: it reaches the "
                    f"interpreter directly and defeats every import rule above."
                )

    if failures:
        print(f"Import-boundary violations (--profile {args.profile}):")
        for f in failures:
            print("  " + f)
        return 1
    # ONLY THE PACKAGES THAT EXIST. Every rule above still applies -- a ban on
    # importing `metrics` matters most where `metrics/` is absent and a
    # submission might vendor one -- but a walked package that is not on disk was
    # checked vacuously, and saying it "respects the rules" reads as though it
    # had been read. The candidate pack is exactly that tree (ADR-0018 D3).
    walked = [pkg for pkg in RULES if Path(pkg).is_dir()]
    print(
        f"Import boundary OK: {', '.join(f'{pkg}/' for pkg in walked)} respect the rules "
        f"and name every import absolutely; contracts.py is standalone."
    )
    if args.profile == "submission":
        print(
            f"Submission profile OK: dispatch/ reaches none of "
            f"{', '.join(sorted(banned_modules))}, nor "
            f"{', '.join(sorted(SUBMISSION_BANNED_BUILTINS))}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
