# Contributing to Camber

Bug reports, fixes, validation data and new features are all welcome. Camber is
small enough that one person can read all of it, and a few rules keep it that
way. Each one exists because breaking it once cost something.

## Getting set up

```sh
git clone https://github.com/Abhijit-s999/camber
cd camber
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/camber catalogue fetch      # UIUC propeller data, ~87 MB, once
.venv/bin/python -m pytest -q         # about 2.5 minutes
```

OpenFOAM and ROS 2 are optional. Nothing in the test suite needs them.

## Reporting a bug

Open an issue with `camber --version`, your Linux distribution and Python
version, the exact command, and what it printed. If a vehicle file is involved,
attach it. A spec that reproduces the problem is worth more than any
description of it.

## The rules

1. **Tests come first.** For physics, a test is an analytic limit or a published
   measurement, not a number the code happened to print. A regression gate
   (such as the UIUC error bound in `tests/test_physics.py`) is loosened only
   with a reason written down, never to make a failing build pass.

2. **Non-obvious choices go in [`docs/DECISIONS.md`](docs/DECISIONS.md).** Give
   the entry the next free ID in its section, a status (`FIXED`, `PROVISIONAL`,
   `OPEN`...), the value chosen and the reasoning. Code comments refer to entries
   by ID. If you change something an entry describes, update the entry in the
   same commit.

3. **The optimiser decides, offline and deterministically.** No language model
   picks a component or evaluates a design, and `camber/` makes no network
   calls except `camber catalogue fetch`. `camber/` never imports `bench/`,
   which is development tooling and is not shipped. `tests/test_isolation.py`
   enforces this.

4. **No third-party data in the repository.** Most component and wind-tunnel
   data grants no redistribution right. A new source arrives as a fetcher that
   downloads from the origin, plus its terms in `camber catalogue sources`.
   `tests/test_licensing.py` fails if source data is ever committed.

5. **The core stays light.** `camber inspect` and `camber optimise` must run
   with numpy alone. Anything heavier (SciPy, MuJoCo, OpenFOAM, CAD) goes behind
   an optional extra in `pyproject.toml` and is imported where it is used.

6. **CFD never runs inside the optimiser loop.** The ladder is analytic
   (microseconds), then blade element (milliseconds), then OpenFOAM (hours). CFD
   refits the cheap model's correction factors; it never scores a candidate.

7. **State the limits.** If a feature is approximate, the CLI and the app say
   so where the number appears. The README's "Known limits" list is part of the
   product.

## Pull requests

Keep a pull request to one change, with its tests and any DECISIONS entry it
needs. CI runs the full suite on the oldest and newest supported Python.

By contributing you agree that your work is licensed under GPL-3.0-or-later,
the same as the rest of Camber.
