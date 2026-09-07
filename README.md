# Camber

Closed-loop airframe optimiser: place payloads and thrusters under constraints,
state what the aircraft is *for*, and get back a Pareto front of real designs
with the components chosen, the layout solved, and manufacturable CAD out the
other end.

Design plan and full rationale:
<https://claude.ai/code/artifact/c48a6764-85d8-4645-b309-0f73e68d70c8>

> **Status: P2.** Vehicle schema, geometry, mass properties, control
> allocation, CAD export — and the physics tier: blade-element propeller
> aerodynamics, motor and pack models, validated against 115 propellers in the
> UIUC wind-tunnel database.

```console
$ camber endurance examples/quad5.toml --prop gwsdd_5x4.3
quad5: 460 g, 4.51 N on 4 rotors -> 1.13 N each

  hover rpm            9233
  per-rotor current    4.29 A     motor eta 63.2%
  thrust per watt    0.0507 N/W
  endurance            4.20 min   capacity-limited
  mission needs        8.00 min   -> MISSES

$ camber validate
section e63: 115 propellers, 9322 points
  CT  median   +7.1%  mean|err|  14.3%  within10%  51%
  CP  median   +6.3%  mean|err|  14.4%  within10%  49%
```

The validation numbers are the point. Until a model reproduces measurements it
has no standing, and `camber validate` is the gate — errors are scaled by each
run's peak coefficient, never by the local value, for reasons in
`DECISIONS.md` PHYS-03.

```console
$ camber inspect examples/quad5.toml
quad5   mission: max_endurance

mass properties
  mass             460.0 g
  cg            (-0.0025, +0.0000, +0.0035) m
  inertia diag  (1.4097e-03, 1.7655e-03, 2.9757e-03) kg m^2

control
  rank          4/6
  commandable   Fz, Mx, My, Mz
  NOT           Fx, Fy   (underactuated: it must tilt to translate)

hover
  weight           4.511 N
    front_right     1.099 N  10.7% of rated
    rear_right      1.156 N  11.2% of rated     <- CG is aft, so the rear works harder
  thrust:weight     9.13 : 1

constraints
  ok   CGBox / ViewCone / MinDistance
```

```fish
camber cad examples/quad5.toml quad5.step   # STEP for CNC
camber cad examples/quad5.toml quad5.stl    # STL for printing
```

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
