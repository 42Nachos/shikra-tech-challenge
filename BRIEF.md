# The brief

PICK WHATEVER TRACK YOU LIKE. Read the common rules, then the section for the
track you want. You may attempt more than one; each is collected and scored
on its own.

## Common rules

**Write only inside your slot.** Your track names one directory (or one file).
When your submission is scored, that slot is copied into a fresh checkout of our
repository and *everything else you changed is discarded unread*.

**To add new dependencies.** You can edit `environment.yml`. Your file is collected
with your submission and your environment is built from it, so the scored run
happens in the stack you asked for. Don't change the pinned versions; they affect
our scoring, and moving `numpy` or `scipy` can move results that are compared
against expected outputs.

Your file must still solve. The build runs `conda env create` once, with no
network afterwards. If it doesn't resolve, you are scored on the shipped
environment instead and the error is recorded against you.

Before adding anything, check what is already installed: `pvlib` (single-diode,
DeSoto, CEC, thermal models), `scipy`, `numpy`, `pandas`, `highspy`, `hypothesis`,
`plotly` and `matplotlib`.

**A write-up is required, as a PDF on the submission form.** A page or two:
what you built and why, what you assumed, what you tried that did not work, and
what you would do with another week. It is read beside your code and counts
towards your assessment. If you attempted more than one track, cover each.

**These must pass**, and they are run against your submission:

```bash
python tools/check_boundary.py     # --profile submission, for track 1
mypy --strict .
ruff check . && ruff format --check .
```

**Timeouts are per track and published.** Wall clock, inside a network-less
container:

| Track | Budget | For scale |
|---|---|---|
| 1 Dispatch | 30 min | the trivial policy runs in 1s; a rolling day-horizon LP solving every interval takes ~35s |
| 2 Modelling | 20 min | a conformance run is a handful of plant runs |
| 3 Test | 45 min | your suite, once per build, against about sixteen builds |
| 4 Tooling | 15 min | one run of your tool against a prepared run directory |

---

## Track 1 — Dispatch

**Slot:** `dispatch/submissions/<handle>/`

Write a dispatch policy. It receives an `Observation` and returns `Commands`,
once per interval. Read `dispatch/null.py` for the smallest complete example and
`dispatch/base.py` for the contract.

**Your entry point is `policy.py` in your slot, exporting `DISPATCHER`.** It is
loaded by path, so the name is the whole interface -- your policy is never
registered in `dispatch/catalog.py` and you should not edit that file.

```python
# dispatch/submissions/<handle>/policy.py
class MyDispatcher:
    def __init__(self, view: FleetView) -> None: ...
    def step(self, obs: Observation) -> Commands: ...

DISPATCHER = MyDispatcher      # Callable[[FleetView], Dispatcher]
```

A class taking the `FleetView` is already the factory shape required; anything
else callable with a view and returning a dispatcher works too. Your slot may
hold as many other modules as you like and they may import each other relatively; 
only `policy.py` and `DISPATCHER` are fixed. **A submission without them does
not load, and does not score.**

**What you may import:** `contracts` and the standard library, minus the
filesystem, the network and the import machinery. `check_boundary.py --profile
submission` lists exactly what is refused and why. A dispatcher is a pure
function from observation to commands; it needs none of them.

**What you cannot see, deliberately.** You never receive `PlantState`, so the
true state of charge is not available to you; only a noisy estimate on the
observation. You never receive `spec:`. What you get instead is
`FleetView.believed`, a separate declaration that is *allowed to be wrong*. How a
policy copes with being wrong is part of what is being measured, so do not assume
the two agree.

**The plant you are scored on is not the plant here.** Same members, same ids,
same tariff; slightly different equipment values, and a different month with different
outages. A policy that hard-codes the numbers in `config/sites/hack_public.yaml`,
or the shape of the trace in the shipped scenario, is fitting the wrong machine.

**Scored on** normalised regret against a do-nothing baseline and a
perfect-information optimum, reported per cost line so it is visible *which* line
you moved. Determinism is checked by running you twice with the same seed and
comparing the logs cell for cell.

---

## Track 2 — Modelling

**Slots:** `sim/assets/bess_models/<handle>.py` **or**
`sim/assets/pv_models/<handle>.py`. Pick one.

Write a real asset model and fit it into the repository so that it runs against
the scenarios. Read `sim/assets/bess_models/tank.py` or
`sim/assets/pv_models/pvwatts.py` for the shape, `sim/asset.py`'s `AssetModel`
for the contract every model satisfies, and `sim/assets/bess_models/base.py` for
what storage adds on top.

### If you take the battery

Build a battery pack, in a 3S2P arrangement, and show that it runs. 

Build a **physics-based pack model**. It could be an equivalent circuit with an open-circuit
voltage curve, internal resistance, and RC branches if you want them, or a electrochemical model, and
**abstract it into the 15-minute power model the port requires.**

**The abstraction is the deliverable, and this brief licenses it explicitly.**
For our problem statement, we care about a 15-minute step; this is energy
bookkeeping and not ODE solving. Your job is to implement better models, while still
collapsing it to the simplicity we need. 
How you justify the collapse matters more than which circuit you choose.

### If you take the array

Model an array. The physics is your choice — a performance-ratio term, soiling
that decays over weeks, a thermal model that takes more than ambient. Reuse the
PVWatts chain where it makes sense.

**Reuse means copying it into your module, not importing it.** Models here are
standalone: none imports another and none inherits from a shared base, so that a
new model cannot change what an existing one computes. That is a convention we do
not lint for, and we will read your code.

### Both halves

**Declare your model in a `slot.yaml` at the repository root**, three lines:

```yaml
family: bess          # bess | pv
model: mymodel        # sim/assets/<family>_models/mymodel.py, and the name config selects
spec_class: MyModelSpec
```

Registering it on the port's params class is an edit to a shared file, so **we
make it, not you** — that keeps your slot the unit of collection and keeps every
candidate out of one file they would all have to touch. A `slot.yaml` naming a
model you did not write is refused.

**Your model carries its own parameter values.** hardcode your params into your script. 
Every field of your spec class takes a default, so the plant selects your model with an empty params block and
you never edit a YAML:

```python
class MyModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    energy_nom_kwh: float = Field(default=2000.0, gt=0)
```


**Scored on conformance, not accuracy.** No validation data ships, so "better" is
unmeasurable here and inviting you to fit against nothing would be a trap. What
is checked: energy balance holds every interval, your envelope is respected,
clip-and-report semantics survive (the plant clips an infeasible command and
*records* it — it never silently fixes one), the property tests pass, your model
declares a spec block, and **the existing golden files do not move**. A golden
that moves means shared physics changed, which is a boundary violation wearing a
feature's clothes.

---

## Track 3 — Test

**Slot:** `tests/submissions/<handle>/`

Write a test suite that enforces this architecture.

**Scored by whether your tests notice a wrong answer.** We keep a set of builds,
each being this repository with one deliberate semantic change. Your suite runs
against each. A test that fails kills that build; an all-green run means it
survived. Your score is (builds killed/builds total).

**Your suite is also run against unmodified code, as a gate.** A suite that fails
on correct code has asserted something untrue, and is reported as failing however
many builds it killed. Without this the winning move would be `assert False`
everywhere.

**These are the bugs we have built in our builds** :

1. A policy can be handed information about an interval the plant has not stepped yet, so a dispatcher that merely echoes what it was given looks prescient.
2. A dispatcher can read the plant's actual state instead of estimating it, so being wrong stops costing anything.
3. A dispatcher can be handed the real equipment parameters in place of the declared ones, making its plant model right by construction.
4. Load can be shed by a fraction that no contactor could actually honour.
5. A cost that was declared can vanish from the bill entirely, because of a name that matches nothing -- and the run reports a total with no sign anything was dropped.
6. A cost the tariff says must be charged can be made silently free.
7. A configuration the loader is supposed to reject can be accepted instead, and quietly given a value nobody declared.
8. The plant's own books can fail to add up, and the run finishes anyway.
9. An impossible command can be quietly made possible: the plant delivers something achievable and nothing on the record says the command was changed.
10. Storage can be driven past the ends of the band it declares.
11. A battery can hand back more energy than was ever put into it.
12. An asset the scenario says is unavailable can still be dispatched.
13. When the grid is out, the bus can be handed to a member the priority order does not name -- or to nobody.
14. An asset's recorded state can move as though it did something the plant then prevented it from doing.
15. An instruction that was never given can become indistinguishable from an instruction to do nothing.

Make a test suite that will catch all of them. 

---

## Track 4 — Tooling

**Slot:** `tools/contrib/<handle>/`

Build a developer interface for a controls or digital-twin engineer to ideate and
analyse with. The audience is the other three tracks: someone who has just run a
policy and needs to understand what the plant did and why the bill came out as it
did.

Read `tools/report.py`, which turns one run into a single self-contained HTML
page. Extending that is the natural shape, but it is not the only one: you can
add to `environment.yml`, so a small server or a richer plotting stack is open to
you if you can justify it. Whatever you build has to run offline once installed.

**Your entry point is `tool.py` in your slot**, and it is called exactly like
this:

```bash
python tools/contrib/<handle>/tool.py --run <a run directory> --out <a directory>
```

`--run` holds `log.csv` and `manifest.json`, prepared the way `runner.py` writes
one. `--out` is a directory you write into — **write at least one file there.**
A tool that prints a beautiful report to stdout and leaves nothing behind has not
produced an artifact anyone can open, and does not pass the gate. Your slot may
hold as many other modules as you like; only `tool.py` and those two flags are
fixed.

**How this is scored, stated plainly: by a person.** Your tool is run against a
prepared run directory and must produce its artifact — that is a gate, not a
score. The ranking is two reviewers reading it independently. There is no
automated metric for an interface, and inventing one would measure whatever the
metric happened to be rather than whether the thing is useful.
