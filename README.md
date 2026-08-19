# The EMS exercise

A 15-minute-resolution simulation testbed for an industrial microgrid: PV, a
battery, a diesel genset, a grid connection and a factory load. It runs, it
enforces physics, and it adds up money.

## Getting started

**1. Make your own copy.** Click **Use this template** on the repository page.
Do *not* fork it — a fork shares its object store with every other fork, so
forking makes your work reachable by other candidates. Use the template, and you
get a fresh private repository of your own.

**2. Your handle is your institute roll number**, exactly as you entered it on
the application form. Everywhere the docs write `<handle>`, put that. It is how
your work is found:

```
dispatch/submissions/CH25B039/policy.py      # if your roll number is CH25B039
```

A directory under any other name is not collected, and a submission that is not
collected cannot be scored. 

**3. Set up the environment.**

```bash
conda env create -f environment.yml
conda activate ems-testbed
```

**4. Read `BRIEF.md`** and pick a track. It names what you write, where it goes,
how long it may take, and how it is scored.

**5. Commit and push before the deadline.** Whatever is on your default branch at
that moment is your submission. We record the commit ourselves.

**6. Submit the form, the write-up, and read access.** Three things, all
required:

- **Fill in the Google form shared with you**, giving your repository link and
  your roll number.
- **Upload a short write-up as a PDF** on the same form. See below.
- **Add `42Nachos` as a collaborator** on your repository:
  *Settings -> Collaborators -> Add people*.

Your repository is yours and private, so without the form and the grant we cannot
see your work and you cannot be scored. There is no other way we find out your
repository exists. Do both when you start rather than at the deadline; it does
not have to be finished for us to have access to it.

## The write-up

**A PDF, submitted on the form, explaining what you did.** It is read alongside
your code and it is part of how you are assessed, not paperwork attached to the
end. A page or two is plenty.

What we are looking for:

- What you built, and why you chose that approach over the alternatives.
- What you assumed, and where you know the assumption is weak.
- What you tried that did not work, and what that told you.
- What you would do with another week.

Write it for an engineer who will read your code straight afterwards. If you
attempted more than one track, cover each. **Track 2 in particular is judged on
your reasoning as much as on the model** — how you justify collapsing your
physics into the 15-minute port is the deliverable, and this is where you say it.

## The four tracks

Pick whichever you like. You may attempt more than one; each is collected and
scored on its own. **One track done well beats two done subpar.**

| Track | You write | It goes in | Judged on |
|---|---|---|---|
| **1 · Dispatch** | A control policy: it sees an `Observation` each interval and returns `Commands`. | `dispatch/submissions/<handle>/policy.py` | Realized cost against a do-nothing baseline and a perfect-information optimum, per cost line. Plus determinism. |
| **2 · Modelling** | A real asset model (an equivalent-circuit battery pack, or a PV array) abstracted into the 15-minute port. | `sim/assets/bess_models/<handle>.py` **or** `sim/assets/pv_models/<handle>.py` | Contract conformance, Model design/thought process: energy balance, envelope, clip-and-report, and that existing behaviour does not move. |
| **3 · Test** | A test suite that holds this architecture still. We are not providing any of our own tests. | `tests/submissions/<handle>/` | Whether your tests notice a wrong answer. Your suite runs against builds each carrying one deliberate defect; the score is how many it catches. |
| **4 · Tooling** | A developer interface for the other three roles to analyse a run with. | `tools/contrib/<handle>/tool.py` | Read by us. There is no automated score for an interface. |

Each track's full brief, including what you may import, what you deliberately
cannot see, and the exact entry point, is in `BRIEF.md`.

## Run it

```bash
python runner.py --scenario studies/scenarios/synthetic/hack_public_2026_03/hack_public_2026_03.csv \
                 --config config/sites/hack_public.yaml
python tools/report.py            # the last run -> one self-contained HTML page
```

## What must stay true

```bash
python tools/check_boundary.py --profile submission
mypy --strict .
ruff check . && ruff format --check .
```

These run against your submission when it is scored. A submission that fails one
of them is ranked below one that runs.

**`pytest` runs the shipped example suite and nothing else.** Our own tests are
deliberately not shipped, so a green run here says the example passes, not that
your work is correct.

## The rules that are not negotiable

- **`sim/` must never import `dispatch/`, and `dispatch/` must never import
  `sim/`.** Both import `contracts.py`, which imports nothing internal. This is
  the architecture, and `tools/check_boundary.py` enforces it.
- **`spec:` is the truth and `believed:` is not.** Your dispatcher never sees
  `spec:`. It receives `believed:`, and those numbers are allowed to be wrong.
- **The plant you are scored on is not the plant shipped here.** Same members,
  same ids, same tariff, different equipment values. Hard-coding the numbers in
  `config/sites/hack_public.yaml` is fitting the wrong machine.
- **Write only inside your slot.** Work that only functions because you also
  edited a shared file will not survive collection.

