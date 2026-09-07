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
    header = f"{'endurance':>10} {'mass':>8}  " + "  ".join(f"{n:>16}" for n in interesting)
    print(header)
    print("-" * len(header))
    seen = set()
    for obj, x in sorted(zip(result.front_objectives, result.front_x),
                         key=lambda t: t[0][1]):
        key = (round(-obj[0] / 60, 2), round(obj[1] * 1000))
        if key in seen:
            continue
        seen.add(key)
        d = dict(zip(names, x))
        cells = "  ".join(f"{d[n]:16.4f}" for n in interesting)
        print(f"{-obj[0]/60:9.2f}m {obj[1]*1000:7.1f}g  {cells}")
    return 0


def main() -> int:
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
    opt.set_defaults(func=cmd_optimise)

    args = p.parse_args()
    return args.func(args)



if __name__ == "__main__":
    raise SystemExit(main())
