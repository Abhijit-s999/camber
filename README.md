# Camber

[![tests](https://github.com/Abhijit-s999/camber/actions/workflows/tests.yml/badge.svg)](https://github.com/Abhijit-s999/camber/actions/workflows/tests.yml)

**A drone design optimiser. You give it a mission, and it designs the drone for
it.**

You place the payloads and rotors, say what the aircraft is *for* (stay up as
long as possible, or go as fast as possible) and set the constraints. Camber
searches the layout and sizing for the best trade-offs, grows a load-bearing
frame by topology optimisation, checks the aerodynamics with CFD, and flies the
result in a physics simulator. It exports the frame for CNC or 3D printing.

It is a desktop app and a command line, and it works offline.

| step in the app | what happens | under the hood |
|---|---|---|
| 1 · Design | place parts; symmetry, balance and clearance are measured, not eyeballed | analytic mass properties, control allocation, hover trim |
| 2 · Optimise | a Pareto front of designs for the mission; pick one | NSGA-II with constrained domination |
| 3 · Structure | the frame is *grown*: material goes only where the loads run | SIMP topology optimisation with hex8 FEA in the loop |
| 4 · Aerodynamics | CFD across the flight envelope the vehicle will actually fly | OpenFOAM, actuator-disc rotors (optional) |
| 5 · Fly | autopilots, your own Python pilot, or the sticks yourself | MuJoCo, with the vehicle's own aerodynamics |

## Install

Linux, Python 3.11 or newer. Developed and tested on Arch (CachyOS) with
Python 3.14.

**Arch and derivatives**, as a desktop app with a launcher entry:

```sh
git clone https://github.com/Abhijit-s999/camber
cd camber
packaging/arch/install.sh        # asks for one pacman line if Qt is missing
```

Then open *Camber* from the app launcher, or run `camber-app`. The install
lives in `~/.local/share/camber` and runs from the checkout, so `git pull`
updates it. `packaging/arch/uninstall.sh` removes it and leaves your projects
alone.

**Any Linux**, from the release wheel:

```sh
uv tool install "camber[app,qt] @ https://github.com/Abhijit-s999/camber/releases/download/v0.1.1/camber-0.1.1-py3-none-any.whl"
camber app
```

(`pipx install` takes the same argument.) Leave out `qt` and the app opens in
your browser instead of its own window.

**Propeller data.** The physics tier needs the UIUC propeller database, which
Camber is not allowed to redistribute (see below). Fetch it once, about 87 MB:

```sh
camber catalogue fetch
```

**Optional:** OpenFOAM for the Aerodynamics step (v2606 is what it was built
against; set `FOAM_INSTALL`, or have `simpleFoam` on your `PATH`), and ROS 2
Jazzy for the bridge in `ros/`, which brings its own environment
(`ros/pixi.toml`). Everything else works without either.

## Command line

Every app step is also a command:

```console
$ camber inspect examples/quad5.toml              # mass, control, hover, constraints
$ camber optimise examples/quad5.toml --prop gwsdd_5x4.3
$ camber endurance examples/quad5.toml            # predicted flight time
$ camber generate examples/racer5.toml --mass 40 --out frame.stl   # ~30 min
$ camber cad examples/quad5.toml out.step         # STEP for CNC, or .stl / .3mf
$ camber sim examples/racer5.toml --viewer        # fly it (docs/FLYING.md)
$ camber aero examples/racer5.toml                # CFD across its flight envelope
$ camber validate                                 # the model against wind-tunnel data
$ camber app                                      # the desktop app
```

A vehicle is a TOML file: bodies, rotors, frame, constraints and a
`[mission]`. `examples/quad5.toml` (an endurance quad) and
`examples/racer5.toml` (a 5-inch racer) are commented, and they are the two
templates the app starts from. Installed from the wheel, the same commands work
from any folder: `examples/quad5.toml`, or just `quad5`, finds the copy that
ships inside Camber, and the CLI prints where it is so you can copy and edit it.

`generate` at its default 3 mm elements takes about half an hour on a laptop.
`--element 0.006 --iterations 15` gives a rough frame in under a minute.

The whole chain is exercised by `tests/test_end_to_end.py`: optimise a real
vehicle, grow a frame for the winner, confirm it still flies, export STEP and
STL, and load the result back in.

## How far to trust it

| tier | what it is | validated against |
|---|---|---|
| geometry | exact analytic mass properties, CAD | closed forms, the triangle inequality |
| control | allocation matrix, hover trim | hand arithmetic; a flat quad is correctly rank 4/6 |
| physics | blade-element propellers, motors, packs | **115 UIUC propellers, CT median +7.1%** |
| optimiser | NSGA-II with constrained domination | Pareto fronts, not a weighted score |
| CFD | actuator discs, RANS | momentum theory to 1.6% |
| calibration | interference model fitted to CFD | **25 CFD cases, leave-one-out 1.34%, R² 0.952** |
| structure | cantilever sizing, 3-axis CNC rules | closed forms; stock section ladder |
| generation | SIMP topology optimisation, hex8 FEA | the element has exactly 6 rigid-body modes |

**Known limits, stated rather than buried:**

- **Components are not chosen yet.** The optimiser searches part positions,
  arm length, one rotor size shared by every rotor, and battery capacity. The
  motor is set by flags (`--kv --rm --i0`), and the propeller is one UIUC map
  scaled across sizes. Picking motors, props and cell counts from a catalogue
  is the next major piece of work.
- **Endurance uses one motor and one scaled propeller.** Once the propeller
  data is fetched, the app and `camber optimise --prop` rank designs on the
  validated blade-element model. Without it they fall back to a momentum-theory
  estimate that reads 10-20 times too long, and say so beside the chart.
- **Multirotors only, four rotors or more.** Wings and tilting rotors are built
  in the simulator but not yet in the app, so this release leaves them out
  (DECISIONS SIM-06, SIM-07).
- The interference model has seen 25 cases of one airframe family, in hover.
  Generated frames can leave that envelope, and the CLI says when they do.
- Static thrust (J = 0) is not claimed.
- The grown frame is a stiffness design with drilled bolt circles. It carries
  no crash or landing load case yet.

Every one of these, and every non-obvious choice behind the rest, is in
[`docs/DECISIONS.md`](https://github.com/Abhijit-s999/camber/blob/main/docs/DECISIONS.md) with its reasoning.

## Data, and what is not shipped

**No third-party measurements are redistributed.** The UIUC Propeller Database
grants no redistribution right, so Camber ships the *builder*:
`camber catalogue fetch` downloads it from the original site and checks its
checksum, `camber catalogue sources` prints the terms, and `NOTICE` carries the
citation. `tests/test_licensing.py` fails the build if any source data ever
enters the tree. See DECISIONS LIC-01.

What does ship is Camber's own: `data/interference.json` (fitted from Camber's
own OpenFOAM runs) and `data/validation_summary.json` (aggregate error
statistics).

## How it was built

Camber is built with AI assistance: Claude (Anthropic), in Claude Code, wrote
and reviewed most of the code, and `bench/` delegated some implementation to
open models over NVIDIA NIM. The bench is development tooling. It is not part
of Camber, in the same way a compiler is not part of what it compiles:

| | `camber/` | `bench/` |
|---|---|---|
| what it is | the product | development tooling |
| network | never, except `catalogue fetch` | NVIDIA NIM, or a local server |
| ships in the wheel | yes | **no** |
| may import the other | **no** | yes |

`tests/test_isolation.py` fails the build if `camber/` ever imports `bench/` or
a network client enters the runtime dependencies. No model picks a component or
evaluates a design; the optimiser decides, offline and deterministically. Tests
come before implementation, and for physics they are analytic limits and
published wind-tunnel data. DECISIONS section 0 has the bench's own record.

## Development

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/camber catalogue fetch                  # without it, 8 data tests skip
.venv/bin/python -m pytest -q                     # about 2.5 minutes
```

Contributions are welcome; [`CONTRIBUTING.md`](https://github.com/Abhijit-s999/camber/blob/main/CONTRIBUTING.md) has the few
rules that keep the project honest.

| path | what it is |
|---|---|
| `camber/` | the product |
| `camber/app/` | the desktop app: a standard-library server and a three.js front end |
| `docs/DECISIONS.md` | every non-obvious choice, with reasoning and source |
| `docs/FLYING.md` | flying a vehicle by hand, by script, or over ROS 2 |
| `tests/` | including the isolation and licensing guards |
| `ros/` | the ROS 2 bridge, in its own pixi environment |
| `bench/` | model orchestration, build-time only (`python -m bench.cli`) |

## Licence

GPL-3.0-or-later; see `LICENSE`. The UIUC data you fetch keeps its own terms;
see `NOTICE` and cite it if you publish results.
