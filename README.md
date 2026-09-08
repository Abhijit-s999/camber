# Camber

Closed-loop airframe optimiser: place payloads and thrusters under constraints,
state what the aircraft is *for*, and get back a Pareto front of real designs
with the components chosen, the layout solved, and manufacturable CAD out the
other end.

Design plan and full rationale:
<https://claude.ai/code/artifact/c48a6764-85d8-4645-b309-0f73e68d70c8>

> **All phases delivered.** Vehicle model, physics validated against 115
> wind-tunnel propellers, multi-objective optimiser, CFD calibration loop,
> generated structure, and a 3D viewport.

```console
$ camber inspect examples/quad5.toml     # mass, control, hover, constraints
$ camber optimise examples/quad5.toml --prop gwsdd_5x4.3
$ camber endurance examples/quad5.toml   # predicted flight time
$ camber cad examples/quad5.toml out.step
$ camber ui examples/quad5.toml          # drag payloads and rotors in 3D
$ camber validate                        # model vs UIUC wind-tunnel data
$ camber catalogue fetch                 # get the source datasets
```

The whole chain, spec to manufacturable output, is exercised by
`tests/test_end_to_end.py`: optimise a real vehicle, generate an airframe for
the winner, confirm it still flies, export STEP and STL, and load the result
back as input.

## What it does, and how far to trust it

| tier | what it is | validated against |
|---|---|---|
| geometry | exact analytic mass properties, CAD | closed forms, the triangle inequality |
| control | allocation matrix, hover trim | hand arithmetic; a flat quad is correctly rank 4/6 |
| physics | blade-element propellers, motors, packs | **115 UIUC propellers, CT median +7.1%** |
| optimiser | NSGA-II with constrained domination | Pareto fronts, not a weighted score |
| CFD | actuator discs, RANS | momentum theory to 1.6% |
| calibration | interference GP | **25 CFD cases, LOO 1.34%, R2 0.952** |
| structure | cantilever sizing, 3-axis CNC rules | closed forms; stock section ladder |

Known limits, stated rather than buried: the interference model has seen 25
cases of one airframe family in hover only; static thrust (J = 0) is not
claimed; blade count is assumed to be two; the generated frame is a structural
lower bound with no fasteners or crash margin.

## Data, and what is not shipped

**No third-party measurements are redistributed.** None of the four sources --
UIUC, APC, Tyto Robotics, ThrustCurve -- grants a redistribution right, so
Camber ships the *builder*: `camber catalogue fetch` downloads from the
originals, `camber catalogue sources` prints the terms, and `NOTICE` carries the
citations. `tests/test_licensing.py` fails the build if any source data ever
enters the tree. See `DECISIONS.md` LIC-01.

## The two halves, and the wall between them

| | `camber/` | `bench/` |
|---|---|---|
| what it is | the product | development tooling |
| network | never | NVIDIA NIM, or a local server |
| ships in the wheel | yes | **no** |
| may import the other | **no** | yes |

`bench/` orchestrates three models to help *build* Camber. It is not part of
Camber, the same way a compiler is not part of what it compiles. `camber/` runs
with no network, no API key and no model loaded — component selection is a
numerical search over a catalogue that ships with the release, not a model call.

That boundary is enforced, not merely intended: `tests/test_isolation.py` fails
the build if `camber/` ever imports `bench/`, if a network client appears in the
runtime dependencies, or if `bench` is added to the packaged modules.

## The bench

Three models, routed by what they are measurably good at:

| role | model | why |
|---|---|---|
| architect | Opus 5, via Claude Code | decomposition, physics review, `DECISIONS.md` |
| `implementer` | `deepseek-v4-pro-0813` | 80.6% SWE-bench, #1 LiveCodeBench (93.5%) |
| `investigator` | `kimi-k3` | 89.5 agentic vs 59.1, and **native vision** |
| `janitor` | local, ternary | unmetered, always on, escalates when stuck |

Kimi is the only route that can look at a residual plot, a mesh screenshot or a
ParaView render — which is most of what diagnosing a failed CFD case involves.

### Setup

```fish
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

Then the key. Either an environment variable:

```fish
set -Ux NVIDIA_API_KEY nvapi-...   # free tier, 40 req/min account-wide
```

...or, preferably, a key file — because Claude Code launches the MCP server as
a subprocess with the environment *it* started with, so a variable exported in
a live shell will not reach the server until Claude Code restarts:

```fish
mkdir -p ~/.config/camber
echo 'NVIDIA_API_KEY=nvapi-...' > ~/.config/camber/env
chmod 600 ~/.config/camber/env
bench status
```

The file is read at call time. It is never logged, never cached, and never
committed.

### Use

Normally you do not — Claude Code calls it over MCP:

```fish
claude mcp add bench -- (pwd)/.venv/bin/python -m bench.mcp_server
```

which exposes `implement`, `investigate`, `review` and `budget` as tools.
Directly, for a quick check:

```fish
bench status                       # budget, routing, call totals
bench ask janitor "reply with OK"  # smallest possible live test
bench serve                        # MCP server on stdio
```

## Rules that are not negotiable

1. **The bench never ships.** See `docs/DECISIONS.md` AI-04.
2. **Models populate and explain; the optimiser decides.** No model ever picks
   a motor — it fills catalogue rows with sources, and NSGA-II searches them.
3. **Tests before implementation, and not written by the implementer.** For
   physics that means analytic limits and published wind-tunnel data, not
   invented unit tests.
4. **CFD never enters the optimiser's inner loop.** Its job is refitting the
   mid-fidelity model's correction factors.
5. **Other projects on this machine are off-limits.** Copy
   and vendor; never extract, never refactor. `~/.local/opt/OpenFOAM-v2606` is
   shared and is strictly read-only, pinned at v2606.

## Layout

| path | what it is |
|---|---|
| `docs/DECISIONS.md` | every non-obvious choice, with reasoning and source |
| `bench/` | model orchestration — build-time only |
| `camber/` | the product |
| `tests/` | including `test_isolation.py`, which guards the wall |

## Licence

GPL-3.0-or-later. Note that the component catalogue is a **separate** question:
UIUC, APC and Tyto Robotics each set their own terms on their measurements, and
those must be verified before any compiled catalogue is redistributed. See
`DECISIONS.md` LIC-01.
