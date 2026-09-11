"""`camber` -- inspect a vehicle specification.

Deliberately dependency-free at runtime beyond numpy: this must work with no
network, no key and no model loaded (AI-07).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .control import AXES, G, allocation_matrix, controllability, solve_hover
from .spec import io


def _fmt_vec(v) -> str:
    return f"({v.x:+.4f}, {v.y:+.4f}, {v.z:+.4f})"


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    mp = v.mass_properties()
    print(f"{v.name}   mission: {v.mission.objective.value}")
    print(f"  bodies    {len(v.all_bodies())}    thrusters {len(v.thrusters)}")
    print()
    print("mass properties")
    print(f"  mass          {mp.mass * 1000:8.1f} g")
    print(f"  cg            {_fmt_vec(mp.cg)} m")
    ix, iy, iz = np.diag(mp.inertia)
    print(f"  inertia diag  ({ix:.4e}, {iy:.4e}, {iz:.4e}) kg m^2")
    print(f"  physical      {'yes' if mp.is_physical() else 'NO - check signs'}")
    if v.total_energy_wh:
        print(f"  energy        {v.total_energy_wh:8.2f} Wh")

    print()
    c = controllability(v)
    print("control")
    print(f"  rank          {c.rank}/6")
    print(f"  commandable   {', '.join(c.controllable_axes) or 'none'}")
    if c.uncontrollable_axes:
        print(f"  NOT           {', '.join(c.uncontrollable_axes)}"
              f"   (underactuated: it must tilt to translate)")
    print(f"  margin        {c.margin:.4g}")

    print()
    h = solve_hover(v)
    print("hover")
    print(f"  weight        {h.weight_n:8.3f} N")
    if h.feasible:
        print(f"  thrust needed {h.total_thrust_n:8.3f} N")
        for t, u, util in zip(v.thrusters, h.thrusts, h.utilisation):
            bar = "" if not np.isfinite(util) else f"  {util:5.1%} of rated"
            print(f"    {t.name:<14} {u:6.3f} N{bar}")
        installed = sum(t.max_thrust() for t in v.thrusters
                        if getattr(t, "static_thrust_n", None))
        if installed:
            print(f"  thrust:weight {installed / h.weight_n:8.2f} : 1")
    else:
        print(f"  INFEASIBLE    {h.reason}")

    breaches = 0
    if v.constraints:
        print()
        print("constraints")
        pos = v.part_positions()
        for con in v.constraints:
            try:
                breach = con.violation(pos, v)
                ok = breach <= 1e-9
                mark = "ok  " if ok else "FAIL"
                extra = "" if ok else f"   breach {breach:.4g}"
                breaches += 0 if ok else 1
            except Exception as exc:  # noqa: BLE001
                mark, extra = "ERR ", f"   {exc}"
                breaches += 1
            print(f"  {mark} {type(con).__name__}{extra}")

    return 0 if (h.feasible and not breaches) else 1


def cmd_convert(args: argparse.Namespace) -> int:
    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    io.save_json(v, args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_cad(args: argparse.Namespace) -> int:
    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2
    from .geometry import cad
    try:
        written = cad.export(v, args.out)
    except cad.CadUnavailable as exc:
        print(f"{exc}", file=sys.stderr)
        return 3
    print(f"wrote {written}")
    return 0


def cmd_endurance(args: argparse.Namespace) -> int:
    """Predicted hover endurance, end to end through the physics tier."""
    from .catalogue import uiuc
    from .physics.battery import Pack
    from .physics.bemt import BladeDefinition
    from .physics.motor import Motor
    from .physics.propulsion import PropellerMap, thrust_for

    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    try:
        db = uiuc.load(args.propdb)
    except uiuc.UiucError as exc:
        print(f"{exc}", file=sys.stderr)
        return 3
    if args.prop not in db.geometry:
        print(f"no geometry for {args.prop!r}; "
              f"{len(db.validation_keys)} available, e.g. "
              f"{', '.join(db.validation_keys[:4])}", file=sys.stderr)
        return 2

    geom = db.geometry[args.prop]
    blade = BladeDefinition.from_uiuc(geom, args.blades).resample(25)
    prop = PropellerMap.from_blade(blade, name=args.prop)

    motor = Motor("motor", kv_rpm_v=args.kv, resistance_ohm=args.rm,
                  no_load_current_a=args.i0)
    pack = Pack("pack", cells=args.cells, capacity_ah=args.capacity,
                resistance_ohm=args.r_int, max_c_rate=args.c_rate)

    mp = v.mass_properties()
    weight = mp.mass * G
    per_rotor = weight / len(v.thrusters)

    print(f"{v.name}: {mp.mass*1000:.0f} g, {weight:.2f} N on {len(v.thrusters)} rotors "
          f"-> {per_rotor:.2f} N each")
    print(f"prop {args.prop} (D={geom.diameter_m*1000:.0f} mm), "
          f"motor {args.kv:.0f} Kv, pack {args.cells}S {args.capacity:.1f} Ah")

    try:
        op = thrust_for(motor, prop, pack, per_rotor, v_inf=0.0)
    except ValueError as exc:
        print(f"\ncannot hover: {exc}", file=sys.stderr)
        return 1

    total_current = op.current_a * len(v.thrusters)
    endurance = pack.endurance_s(total_current)
    print()
    print(f"  hover rpm        {op.rpm:8.0f}")
    print(f"  per-rotor current{op.current_a:8.2f} A     motor eta {op.motor_efficiency:.1%}")
    print(f"  pack current     {total_current:8.2f} A     "
          f"({total_current/pack.max_current_a:.0%} of the pack's rating)")
    print(f"  thrust per watt  {op.thrust_per_watt:8.4f} N/W")
    print(f"  endurance        {endurance/60:8.2f} min   "
          f"{'SAG-LIMITED' if pack.sag_limited(total_current) else 'capacity-limited'}")
    if v.mission.min_endurance:
        need = v.mission.min_endurance
        verdict = "MEETS" if endurance >= need else "MISSES"
        print(f"  mission needs    {need/60:8.2f} min   -> {verdict}")
        return 0 if endurance >= need else 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the model against the UIUC wind-tunnel database."""
    from .physics.validate import validate
    rep = validate(root=args.propdb, airfoil=args.airfoil,
                   stations=args.stations, max_props=args.max_props)
    s = rep.summary()
    if not s.get("points"):
        print("no usable points", file=sys.stderr)
        return 1
    print(f"section {s['airfoil']}: {s['propellers']} propellers, {s['points']} points")
    for label, key in (("CT", "ct_scaled"), ("CP", "cp_scaled")):
        d = s[key]
        print(f"  {label}  median {d['median']:+7.1%}  mean|err| {d['mean_abs']:6.1%}  "
              f"p90 {d['p90_abs']:6.1%}  within10% {d['within_10pct']:4.0%}  "
              f"within20% {d['within_20pct']:4.0%}")
    print("  (errors scaled by each run's peak coefficient -- DECISIONS PHYS-03)")
    return 0


#: How to render each objective on the front. Keyed by the names
#: `DesignProblem.objective_names` returns, because hardcoding "endurance" and
#: "mass" here meant that changing the MAX_SPEED objective pair (OPT-07) left
#: the table printing thrust margin under a "mass" heading, in grams, negative.
_OBJECTIVE_COLUMNS = {
    "-endurance_s":   ("endurance", 10, lambda v: f"{-v / 60:.2f}m"),
    "mass_kg":        ("mass", 9, lambda v: f"{v * 1000:.1f}g"),
    "-thrust_margin": ("T:W", 8, lambda v: f"{-v:.2f}"),
    "time_to_target_s": ("time", 9, lambda v: f"{v:.1f}s"),
    "min_stable_speed": ("min speed", 10, lambda v: f"{v:.2f}m/s"),
}


def cmd_optimise(args: argparse.Namespace) -> int:
    """Search the design space and print the Pareto front."""
    import time

    from .optimise.nsga2 import minimise
    from .optimise.problem import (
        DesignProblem, arm_length_variable, battery_capacity_variable,
        physics_endurance_model, position_variables, rotor_radius_variable,
    )
    from .physics.motor import Motor

    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    variables = position_variables(v, span=args.span)
    variables.append(arm_length_variable(v, args.arm_min, args.arm_max))
    variables.append(rotor_radius_variable(v, args.radius_min, args.radius_max))
    packs = [b.name for b in v.bodies if hasattr(b, "capacity_ah")]
    if packs:
        variables.append(battery_capacity_variable(
            v, packs[0], args.capacity_min, args.capacity_max))

    endurance = None
    if args.prop:
        motor = Motor("motor", kv_rpm_v=args.kv, resistance_ohm=args.rm,
                      no_load_current_a=args.i0)
        try:
            endurance = physics_endurance_model(args.prop, motor, str(args.propdb))
        except (KeyError, Exception) as exc:  # noqa: BLE001
            print(f"could not build the physics model: {exc}", file=sys.stderr)
            return 3
    else:
        print("warning: no --prop given, so endurance uses the momentum-theory\n"
              "         placeholder, which overstates it by more than 10x. Use it\n"
              "         to compare layouts, never as a number.", file=sys.stderr)

    problem = DesignProblem(v, variables, endurance_model=endurance)
    print(f"{v.name}: {problem.n_var} variables, objectives "
          f"{problem.objective_names}")

    started = time.time()
    result = minimise(problem, population=args.population,
                      generations=args.generations, seed=args.seed,
                      verbose=args.verbose)
    print(f"{time.time()-started:.1f}s, feasible "
          f"{result.feasible_count}/{len(result.evaluations)}, "
          f"front {len(result.front_indices)}\n")

    names = [d.name for d in problem.variables]
    interesting = [n for n in names
                   if n.endswith(("capacity_ah", "radius", "length"))]
    cols = [_OBJECTIVE_COLUMNS.get(n, (n, 10, lambda v: f"{v:.4g}"))
            for n in problem.objective_names]
    header = "  ".join(f"{c[0]:>{c[1]}}" for c in cols) + "  " \
        + "  ".join(f"{n:>16}" for n in interesting)
    print(header)
    print("-" * len(header))
    seen = set()
    for obj, x in sorted(zip(result.front_objectives, result.front_x),
                         key=lambda t: tuple(t[0])):
        cells = [c[2](v) for c, v in zip(cols, obj)]
        key = tuple(cells)
        if key in seen:
            continue
        seen.add(key)
        d = dict(zip(names, x))
        print("  ".join(f"{t:>{c[1]}}" for c, t in zip(cols, cells)) + "  "
              + "  ".join(f"{d[n]:16.4f}" for n in interesting))

    if args.out:
        # A front is not an answer until one point on it can be handed to the
        # next stage. Without this the chain breaks at the CLI: the optimiser
        # prints a table and `camber generate` needs a spec file.
        best = min(range(len(result.front_objectives)),
                   key=lambda i: result.front_objectives[i][0])
        chosen = problem.build(result.front_x[best])
        chosen.name = f"{v.name}_optimised"
        io.save_json(chosen, args.out)
        obj = result.front_objectives[best]
        print(f"\nwrote {args.out}: best on {problem.objective_names[0]}, "
              + ", ".join(f"{c[0]} {c[2](o)}" for c, o in zip(cols, obj)))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Serve the drag-to-place viewport on loopback."""
    from .web.server import serve
    try:
        serve(args.spec, port=args.port, open_browser=not args.no_browser)
    except OSError as exc:
        print(f"could not start the server: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_catalogue(args: argparse.Namespace) -> int:
    """Fetch source datasets, or print their terms."""
    from .catalogue.fetch import FetchError, fetch_all
    from .catalogue.sources import SOURCES, notice, redistributable

    if args.action == "sources":
        print(notice())
        print(f"{len(redistributable())} of {len(SOURCES)} may be redistributed "
              f"with Camber.")
        return 0

    print("Fetching source datasets. None of these is redistributed with")
    print("Camber; each is downloaded from its origin (see `catalogue sources`).\n")
    failed = 0
    for r in fetch_all(args.root, force=args.force):
        name = SOURCES[r.source].name
        if r.path is None:
            print(f"  --  {name}\n      {r.message}")
            failed += 1
        else:
            mark = ("verified" if r.verified else
                    "UNVERIFIED" if r.verified is False else "no checksum")
            print(f"  ok  {name}  {r.bytes/2**20:.1f} MiB  "
                  f"{'cached' if r.cached else 'downloaded'}, {mark}")
    return 1 if failed else 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Grow an airframe by topology optimisation, and mesh it.

    This is the generative path: nothing here decides that a frame is a plate
    and four tubes. It states where load enters, where the structure is held,
    and how much carbon there is to spend, and the FEA decides the rest.
    """
    import time

    from .generate import topology

    try:
        v = io.load(args.spec)
    except io.SpecError as exc:
        print(f"spec error: {exc}", file=sys.stderr)
        return 2

    problem = topology.frame_problem(
        v, element=args.element, thickness=args.thickness,
        target_mass=None if args.price else args.mass / 1000.0,
        material_density=args.density, material_price=args.price,
        max_deflection=args.deflection / 1000.0)
    d = problem.domain
    print(f"domain {d.nx}x{d.ny}x{d.nz} = {d.n_elements:,} elements "
          f"at {d.element * 1000:.1f} mm")
    if args.price:
        print(f"material priced at {args.price:g} against "
              f"{problem.mass_scale * 1000:.0f} g of aircraft and "
              f"{args.deflection:g} mm of allowed deflection "
              f"({problem.compliance_scale * 1000:.1f} mJ); "
              f"mass is an outcome")
    else:
        print(f"mass budget {args.mass:.1f} g, volume fraction "
              f"{problem.volume_fraction:.4f}")
    print(f"{len(problem.fixed):,} fixed dof, "
          f"{int(np.count_nonzero(problem.load)):,} loaded dof")

    if args.min_feature:
        # The filter radius is the length scale: a density filter of radius R
        # cannot produce a member much thinner than R, which is the whole
        # mechanism. Quoting it in millimetres rather than elements means the
        # number stays meaningful when the grid changes.
        args.filter = max(1.5, args.min_feature / 1000.0 / args.element)
        print(f"minimum feature {args.min_feature:g} mm "
              f"-> filter radius {args.filter:.2f} elements")

    if not args.robust and not args.min_feature:
        pass
    elif d.nz < 2 * args.filter + 1:
        # Air on both faces means a member needs a filter radius of clearance
        # from each; a domain thinner than that has no interior, every element
        # is "near a face", and the void boundary greys the entire field --
        # measured at 57.5% grey on a two-element-thick test. Say so rather
        # than return mush that looks like a convergence problem.
        print(f"  WARNING: the domain is {d.nz} elements thick and the length "
              f"scale needs {2 * args.filter + 1:.0f}. Use a finer --element "
              f"or a smaller --min-feature, or expect a grey result",
              file=sys.stderr)

    t0 = time.time()
    result = topology.optimise(
        problem, iterations=args.iterations, filter_radius=args.filter,
        penalty_start=args.penalty if args.no_continuation else 1.0,
        penalty=args.penalty, E0=args.modulus, Emin=args.modulus * 1e-9,
        beta=args.beta, robust=args.robust, solver=args.solver,
        verbose=args.verbose)
    seconds = time.time() - t0

    c0, c1 = result.compliance_history[0], result.compliance_history[-1]
    islands, connected = result.connectivity()
    hinges = result.hinges()
    grey = float(((result.density > 0.05) & (result.density < 0.95)).mean())
    print(f"\n{result.iterations} iterations in {seconds:.0f} s, "
          f"converged={result.converged}")
    scale = problem.compliance_scale
    print(f"  compliance   {c0:.3e} -> {c1:.3e} J  ({c0 / c1:.1f}x stiffer)"
          + ("  [eroded design: the worst case]" if args.robust else ""))
    # c = u.f, so c / F is the load-weighted mean deflection of the loaded
    # nodes -- not the peak, which is higher. Stated as what it is.
    load_n = float(abs(problem.load).sum()) / 2.0
    print(f"  deflection   {c1 / load_n * 1000:.3f} mm mean at the mounts "
          f"under {load_n:.1f} N")
    print(f"  grey         {grey:.1%} of elements")
    print(f"  islands      {islands}, largest holds {connected:.1%} "
          f"of the material")
    print(f"  hinges       {hinges} joins that hold only at an edge")
    if args.min_feature:
        # Diameter in cells of a ball the minimum feature must contain.
        thin = result.thin_fraction(args.min_feature / 1000.0 / args.element)
        print(f"  thin         {thin:.1%} of the material is narrower than "
              f"{args.min_feature:g} mm")
    if connected < 0.95:
        print("  WARNING: material is stranded in islands that carry no load; "
              "the meshed part will weigh less than the target", file=sys.stderr)
    if hinges:
        # Different defect, different fix: islands mean wasted budget, hinges
        # mean the element formulation is carrying load through a corner that
        # carbon cannot. More material or a coarser grid, not more iterations.
        print(f"  WARNING: {hinges} edge-only joins; members are about one "
              f"element wide, so the discrete model is stiffer here than the "
              f"part would be", file=sys.stderr)

    if args.price:
        chosen = topology.structural_mass(result, d, args.density)
        print(f"  it chose     {chosen * 1000:.1f} g of material "
              f"({result.volume_fraction:.4f} of the box)")

    if args.density_out:
        np.save(args.density_out, result.density)
        print(f"  wrote {args.density_out}")

    try:
        mesh = topology.to_mesh(result, d)
    except ImportError as exc:
        print(f"meshing needs scikit-image and trimesh: {exc}", file=sys.stderr)
        return 3
    if problem.fasteners and not args.no_drill:
        before = mesh.volume
        try:
            mesh = topology.drill(mesh, problem.fasteners)
            print(f"  drilled      {len(problem.fasteners)} bolt holes, "
                  f"{(before - mesh.volume) * 1e6:.3f} cm^3 removed")
        except Exception as exc:                             # noqa: BLE001
            print(f"  WARNING: could not drill holes: {exc}", file=sys.stderr)

    mass = mesh.volume * args.density        # while still in metres
    print(f"  mesh         {len(mesh.faces):,} facets, "
          f"watertight={mesh.is_watertight}, {mesh.body_count} "
          f"{'body' if mesh.body_count == 1 else 'bodies'}, "
          f"{mesh.volume * 1e6:.2f} cm^3 = {mass * 1000:.1f} g")
    if mesh.body_count > 1:
        print(f"  WARNING: the mesh is {mesh.body_count} separate shells, not "
              f"one part; a slicer will treat each as its own object",
              file=sys.stderr)
    # GEOM-04: millimetres for anything headed to a slicer or a mill, metres
    # only when asked. An STL carries no unit, so every consumer guesses, and
    # they all guess millimetres -- a metre-scale file makes this 202 mm frame
    # 0.2 mm across and a slicer accepts it without a word. CFD-02 is the same
    # mistake in the other direction, and cost a whole CFD campaign.
    if not args.metres:
        # Manufacturing output: millimetres and Z-up. `--metres` is the CFD
        # path, which stays in NED because the CFD case is built in NED.
        mesh.apply_scale(1000.0)
        topology.ned_to_z_up(mesh)
    unit = "m (NED)" if args.metres else "mm, Z-up"
    lo, hi = mesh.bounds
    print(f"  extents      {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x "
          f"{hi[2]-lo[2]:.1f} {unit}")
    mesh.export(args.out)
    print(f"  wrote {args.out} in {unit}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # `argv` is for tests: a CLI that can only be driven through sys.argv
    # can only be tested by poking at its internals.
    p = argparse.ArgumentParser(prog="camber", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser("inspect", help="mass properties, control, hover")
    ins.add_argument("spec", type=Path)
    ins.set_defaults(func=cmd_inspect)

    conv = sub.add_parser("convert", help="normalise a spec to JSON")
    conv.add_argument("spec", type=Path)
    conv.add_argument("out", type=Path)
    conv.set_defaults(func=cmd_convert)

    cad = sub.add_parser("cad", help="export STEP (CNC) or 3MF/STL (print)")
    cad.add_argument("spec", type=Path)
    cad.add_argument("out", type=Path)
    cad.set_defaults(func=cmd_cad)

    end = sub.add_parser("endurance", help="predicted hover endurance")
    end.add_argument("spec", type=Path)
    end.add_argument("--prop", default="gwsdd_5x4.3", help="UIUC propeller key")
    end.add_argument("--propdb", type=Path, default=Path("data/raw/UIUC-propDB"))
    end.add_argument("--blades", type=int, default=2)
    end.add_argument("--kv", type=float, default=1900.0)
    end.add_argument("--rm", type=float, default=0.075, help="winding ohms")
    end.add_argument("--i0", type=float, default=1.4, help="no-load amps")
    end.add_argument("--cells", type=int, default=4)
    end.add_argument("--capacity", type=float, default=1.5, help="Ah")
    end.add_argument("--r-int", type=float, default=0.018, help="pack ohms")
    end.add_argument("--c-rate", type=float, default=75.0)
    end.set_defaults(func=cmd_endurance)

    val = sub.add_parser("validate", help="check the model against UIUC data")
    val.add_argument("--propdb", type=Path, default=Path("data/raw/UIUC-propDB"))
    val.add_argument("--airfoil", default=None)
    val.add_argument("--stations", type=int, default=25)
    val.add_argument("--max-props", type=int, default=None)
    val.set_defaults(func=cmd_validate)

    opt = sub.add_parser("optimise", help="search the design space")
    opt.add_argument("spec", type=Path)
    opt.add_argument("--prop", default=None,
                     help="UIUC propeller key; without it endurance uses a "
                          "placeholder that overstates by >10x")
    opt.add_argument("--propdb", type=Path, default=Path("data/raw/UIUC-propDB"))
    opt.add_argument("--kv", type=float, default=1900.0)
    opt.add_argument("--rm", type=float, default=0.075)
    opt.add_argument("--i0", type=float, default=1.4)
    opt.add_argument("--population", type=int, default=32)
    opt.add_argument("--generations", type=int, default=20)
    opt.add_argument("--seed", type=int, default=0)
    opt.add_argument("--span", type=float, default=0.04, help="payload travel, m")
    opt.add_argument("--arm-min", type=float, default=0.08)
    opt.add_argument("--arm-max", type=float, default=0.16)
    opt.add_argument("--radius-min", type=float, default=0.050)
    opt.add_argument("--radius-max", type=float, default=0.090)
    opt.add_argument("--capacity-min", type=float, default=0.8)
    opt.add_argument("--capacity-max", type=float, default=4.0)
    opt.add_argument("--verbose", action="store_true")
    opt.add_argument("--out", type=Path,
                     help="write the front's best design on the primary "
                          "objective, as a spec `camber generate` can read")
    opt.set_defaults(func=cmd_optimise)

    ui = sub.add_parser("ui", help="3D viewport: drag payloads and rotors")
    ui.add_argument("spec", type=Path)
    ui.add_argument("--port", type=int, default=5173)
    ui.add_argument("--no-browser", action="store_true")
    ui.set_defaults(func=cmd_ui)

    gen = sub.add_parser("generate",
                         help="grow a frame by topology optimisation")
    gen.add_argument("spec", type=Path)
    gen.add_argument("--out", type=Path, default=Path("frame.stl"))
    budget = gen.add_mutually_exclusive_group()
    budget.add_argument("--mass", type=float, default=40.0,
                        help="fix the structural mass, grams (default 40); the "
                             "optimiser must hit it exactly")
    budget.add_argument("--price", type=float,
                        help="instead of a fixed mass, charge for material at "
                             "this rate and let the optimiser choose. 1.0 "
                             "weights stiffness and mass equally; below 1.0 "
                             "says material is cheaper than compliance, so it "
                             "spends more and comes out stiffer")
    gen.add_argument("--element", type=float, default=0.003,
                     help="element edge, metres (default 0.003)")
    gen.add_argument("--thickness", type=float, default=0.012,
                     help="design domain thickness, metres")
    gen.add_argument("--density", type=float, default=1550.0,
                     help="material density, kg/m^3 (default: carbon)")
    gen.add_argument("--modulus", type=float, default=70e9,
                     help="Young's modulus, Pa (default: 70e9, CFRP). Running "
                          "the solve at the real modulus is what puts "
                          "compliance in joules and deflection in metres")
    gen.add_argument("--deflection", type=float, default=1.0,
                     help="largest acceptable deflection under the manoeuvre "
                          "load, mm (default 1.0). Only used by --price, as "
                          "the scale stiffness is charged against")
    gen.add_argument("--iterations", type=int, default=80)
    gen.add_argument("--min-feature", type=float, dest="min_feature",
                     default=3.0,
                     help="smallest structural member the design may contain, "
                          "mm (default 3). Sets the density filter radius and, "
                          "with --robust, is enforced rather than encouraged")
    gen.add_argument("--robust", type=float, default=0.2,
                     help="erosion margin on the projection threshold (default "
                          "0.2). The analysis runs on the eroded design, so a "
                          "member must survive thinning to count as structure. "
                          "0 gives the plain projection, which let stringy webs "
                          "through")
    gen.add_argument("--filter", type=float, default=2.4,
                     help="density filter radius in elements; --min-feature "
                          "is the same knob in millimetres")
    gen.add_argument("--beta", type=float, default=16.0,
                     help="projection sharpness. 0 disables it, leaving the "
                          "grey band the filter creates")
    gen.add_argument("--penalty", type=float, default=3.0)
    gen.add_argument("--no-continuation", action="store_true",
                     help="start at the final penalty instead of ramping")
    gen.add_argument("--density-out", type=Path,
                     help="save the raw density field as .npy")
    gen.add_argument("--metres", action="store_true",
                     help="write the mesh at SI scale instead of millimetres. "
                          "Slicers and CAM assume mm; CFD wants m (GEOM-04)")
    gen.add_argument("--no-drill", action="store_true",
                     help="leave the bolt holes as the grid carved them "
                          "instead of subtracting exact cylinders")
    gen.add_argument("--solver", default="auto",
                     choices=("auto", "direct", "amg"),
                     help="auto picks multigrid above 150k unknowns, where a "
                          "direct factorisation's fill-in stops being viable")
    gen.add_argument("--verbose", action="store_true")
    gen.set_defaults(func=cmd_generate)

    cat = sub.add_parser("catalogue", help="fetch component data, or show terms")
    cat.add_argument("action", choices=["fetch", "sources"])
    cat.add_argument("--root", type=Path, default=Path("data/raw"))
    cat.add_argument("--force", action="store_true")
    cat.set_defaults(func=cmd_catalogue)

    args = p.parse_args(argv)
    return args.func(args)



if __name__ == "__main__":
    raise SystemExit(main())
