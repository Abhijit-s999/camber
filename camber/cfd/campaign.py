"""Run a batch of CFD cases and fit the interference model to what they say.

This is the loop from ARCH-01 actually turning: sample layouts across the
feature space, measure the download each one suffers, fit the residual, and
hand the correction back to the cheap tier the optimiser searches.

Resumable in the same way the earlier CFD sweeps were, and for the same reason: a
campaign runs unattended for hours and must survive a closed lid, a power cut
or a Ctrl-C without losing completed cases. State is written to disk after
every case and a case whose result already exists is skipped.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..calibrate import InterferenceModel, design_features, loss_fraction
from ..control import solve_hover
from ..optimise.problem import arm_length_variable, rotor_radius_variable
from ..spec.vehicle import Vehicle
from .case import CaseBuilder, CaseConfig
from .screen import SKIP_THRESHOLD, screen
from .run import FoamError, cell_count, mesh, read_body_force, solve


@dataclass
class CaseResult:
    name: str
    arm: float
    radius: float
    features: list[float]
    isolated_thrust_n: float
    body_force_z_n: float | None
    loss: float | None
    cells: int | None
    seconds: float
    ok: bool
    message: str = ""
    #: True when geometry ruled the layout out and no CFD was run.
    screened: bool = False
    overlap: float = 0.0


@dataclass
class Campaign:
    """A set of layouts to measure, and the results so far."""

    template: Vehicle
    root: Path
    results: list[CaseResult] = field(default_factory=list)

    @property
    def state_path(self) -> Path:
        return self.root / "campaign.json"

    def load(self) -> "Campaign":
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            self.results = [CaseResult(**r) for r in data]
        return self

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps([asdict(r) for r in self.results], indent=2))

    def done(self, name: str) -> bool:
        return any(r.name == name and r.ok for r in self.results)


def sample_interesting(template: Vehicle, n: int,
                       arm_range: tuple[float, float],
                       radius_range: tuple[float, float],
                       seed: int = 0, oversample: int = 40,
                       threshold: float = SKIP_THRESHOLD) -> list[tuple[float, float]]:
    """Sample layouts the geometric screen thinks are worth solving.

    The first campaign sampled the whole box uniformly and spent 22 of 30 runs
    in the flat region, where the answer is "about zero" and the model learns
    nothing (CAL-02). Screening candidates first costs microseconds each and
    puts the entire CFD budget in the band where download actually varies.

    Oversamples the parameter box and keeps the survivors, rather than trying
    to invert the geometry -- the feasible region is an awkward shape and
    rejection sampling against a 7 ms test is far simpler than describing it.
    """
    rng = np.random.default_rng(seed)
    keep: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(n * oversample):
        if len(keep) >= n:
            break
        arm = rng.uniform(*arm_range)
        radius = rng.uniform(*radius_range)
        if 2.0 * radius > 1.90 * arm:
            continue
        key = (int(arm * 2000), int(radius * 2000))   # 0.5 mm buckets
        if key in seen:
            continue
        variant = build_variant(template, arm, radius)
        if screen(variant, threshold, samples=4000).worth_solving:
            seen.add(key)
            keep.append((arm, radius))
    return keep


def sample_layouts(n: int, arm_range: tuple[float, float],
                   radius_range: tuple[float, float],
                   seed: int = 0) -> list[tuple[float, float]]:
    """Latin hypercube over arm length and rotor radius.

    A grid would waste runs: with CFD cases costing minutes each, two points
    that share an arm length teach less than two that share nothing. Latin
    hypercube spreads both coordinates evenly with no coincidences.

    Layouts where the discs would overlap are rejected -- an actuator disc
    intersecting another is not a design, and the momentum sources would
    double-count in the shared cells.
    """
    rng = np.random.default_rng(seed)
    picks: list[tuple[float, float]] = []
    tries = 0
    while len(picks) < n and tries < n * 40:
        tries += 1
        i = len(picks)
        arm = arm_range[0] + (arm_range[1] - arm_range[0]) * (
            (i + rng.random()) / n)
        radius = radius_range[0] + (radius_range[1] - radius_range[0]) * rng.random()
        # Adjacent rotors on a square layout are arm*sqrt(2) apart... but the
        # nearer pair share an axis at 2*arm/sqrt(2). Keep a 5% gap.
        if 2.0 * radius > 1.90 * arm:
            continue
        picks.append((arm, radius))
    return picks


def build_variant(template: Vehicle, arm: float, radius: float) -> Vehicle:
    v = copy.deepcopy(template)
    arm_length_variable(template, 0.0, 1.0).apply(v, arm)
    rotor_radius_variable(template, 0.0, 1.0).apply(v, radius)
    return v


def run_campaign(template: Vehicle, root: Path, n: int = 12,
                 arm_range: tuple[float, float] = (0.09, 0.20),
                 radius_range: tuple[float, float] = (0.050, 0.085),
                 iterations: int = 400, seed: int = 0,
                 stl_dir: Path | None = None,
                 verbose: bool = True,
                 focus: bool = True) -> Campaign:
    """Measure `n` layouts. Safe to interrupt and re-run.

    With `focus` (the default) the layouts are drawn from the region the
    geometric screen says can actually interfere, so the CFD budget is not
    spent confirming that widely spaced rotors do not.
    """
    from ..geometry import cad

    root = Path(root)
    stl_dir = Path(stl_dir or root / "stl")
    stl_dir.mkdir(parents=True, exist_ok=True)
    campaign = Campaign(template, root).load()

    layouts = (sample_interesting(template, n, arm_range, radius_range, seed)
               if focus else sample_layouts(n, arm_range, radius_range, seed))
    if verbose and focus:
        print(f"{len(layouts)} layouts selected by the geometric screen "
              f"(the flat region is skipped, not solved)", flush=True)

    for index, (arm, radius) in enumerate(layouts):
        name = f"case{index:02d}"
        if campaign.done(name):
            if verbose:
                print(f"{name}: already done, skipping", flush=True)
            continue

        started = time.monotonic()
        vehicle = build_variant(template, arm, radius)

        verdict = screen(vehicle)
        if not verdict.worth_solving:
            campaign.results.append(CaseResult(
                name, arm, radius, list(design_features(vehicle)), 0.0,
                0.0, 0.0, None, time.monotonic() - started, True,
                f"screened: {verdict.reason}", screened=True,
                overlap=verdict.overlap))
            campaign.save()
            if verbose:
                print(f"{name}: screened out ({verdict.overlap:.1%} overlap), "
                      f"no CFD run", flush=True)
            continue

        hover = solve_hover(vehicle)
        if not hover.feasible:
            campaign.results.append(CaseResult(
                name, arm, radius, list(design_features(vehicle)), 0.0,
                None, None, None, 0.0, False, f"cannot hover: {hover.reason}"))
            campaign.save()
            continue

        thrusts = {t.name: float(u) for t, u in zip(vehicle.thrusters, hover.thrusts)}
        isolated = float(sum(thrusts.values()))

        stl = stl_dir / f"{name}.stl"
        cad.export(vehicle, stl, metres=True)   # metres: see CFD-02

        builder = CaseBuilder(vehicle, thrusts,
                              CaseConfig(v_inf=0.0, n_iterations=iterations))
        case_dir = root / name
        try:
            builder.build(case_dir, stl)
            mesh(case_dir)
            solve(case_dir, parallel=True)
            force = read_body_force(case_dir)
            if force is None:
                raise FoamError("no force output")
            loss = loss_fraction(isolated, force["fz"])
            result = CaseResult(
                name, arm, radius, list(design_features(vehicle)), isolated,
                force["fz"], loss, cell_count(case_dir),
                time.monotonic() - started, True, overlap=verdict.overlap)
        except Exception as exc:  # noqa: BLE001
            result = CaseResult(
                name, arm, radius, list(design_features(vehicle)), isolated,
                None, None, cell_count(case_dir), time.monotonic() - started,
                False, str(exc)[:200])

        campaign.results.append(result)
        campaign.save()
        if verbose:
            if result.ok:
                print(f"{name}: arm {arm*1000:5.1f} mm  R {radius*1000:5.1f} mm  "
                      f"cells {result.cells:>7}  loss {result.loss:+7.3%}  "
                      f"{result.seconds:5.0f}s", flush=True)
            else:
                print(f"{name}: FAILED {result.message[:80]}", flush=True)
    return campaign


def usable(campaign: Campaign, min_cell_ratio: float = 0.35) -> list[CaseResult]:
    """Successful cases whose mesh is not obviously under-resolved.

    A case that converged is not necessarily a case that resolved anything.
    In the first campaign one layout meshed to 37,000 cells while every other
    reached 141,000 to 366,000 -- four to ten times finer -- and that same
    case reported by far the most extreme download. Fitting the model to it
    would teach the optimiser an artefact.

    The test is relative rather than absolute because the right cell count
    depends on the layout: a case far below its cohort's median is suspicious
    whatever the number.
    """
    # Screened cases carry no mesh and are excluded from the resolution test;
    # they are legitimate data points, just not CFD ones.
    good = [r for r in campaign.results
            if r.ok and r.loss is not None and (r.cells or r.screened)]
    if len(good) < 3:
        return good
    solved = [r for r in good if r.cells]
    if len(solved) < 3:
        return good
    median = float(np.median([r.cells for r in solved]))
    return ([r for r in solved if r.cells >= min_cell_ratio * median]
            + [r for r in good if r.screened])


def fit(campaign: Campaign, transform: str = "sqrt") -> InterferenceModel | None:
    """Fit the interference model to the usable cases.

    `transform` defaults to sqrt because the response spans three orders of
    magnitude -- 0.04% when the discs clear the airframe, 40% when they sit on
    it -- and a GP fitted to raw values spends its length scale on the large
    end. Measured by leave-one-out over the first campaign: 6.9% mean error in
    sqrt space against 8.3% raw. Both are still too large to trust; see
    DECISIONS.md CAL-02.
    """
    good = usable(campaign)
    if len(good) < 3:
        return None
    x = np.array([r.features for r in good])
    y = np.array([r.loss for r in good])
    if transform == "sqrt":
        model = InterferenceModel()
        model.gp.tune(x, np.sqrt(np.maximum(y, 0.0)))
        model.fitted, model.n_observations = True, len(good)
        model._sqrt_space = True
        return model
    return InterferenceModel().fit(x, y)
