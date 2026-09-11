"""Generate an OpenFOAM case for a vehicle in hover or forward flight.

Geometry comes from the same parametric model as everything else (P1), exported
to STL and meshed with snappyHexMesh. Rotors are **prescribed-thrust actuator
discs**, not resolved blades.

That choice is deliberate and it is what keeps the calibration honest. A
blade-resolved simulation would re-derive the rotor's own performance, which
we already measured; worse, feeding a BEMT-derived loading into a rotor model
and then using the answer to correct BEMT is circular. Prescribing the thrust
L1 predicts and measuring the force the *airframe* feels in that wake isolates
the interference term cleanly -- it is the one quantity L1 cannot produce and
CFD can.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import os

from ..spec.units import Vec3
from ..spec.vehicle import Rotor, Vehicle

#: Fraction of the machine a simulation may take. Not 1.0, deliberately.
#:
#: A CFD run that saturates every core makes the machine unusable for anything
#: else and runs the fans flat out, and the last cores buy very little anyway:
#: the solve is memory-bandwidth bound (GPU-01), so ranks beyond about half the
#: cores contend for the same DRAM channels rather than adding throughput.
#: Measured on this machine, 4 ranks gave 1.40x the throughput of 2 -- already
#: only 70% efficient -- so the top 20% of cores are close to free to give up.
CPU_BUDGET = float(os.environ.get("CAMBER_CPU_BUDGET", "1.0"))

#: Scheduler niceness for solver processes. Positive means "yield to
#: everything else", so a long campaign stays in the background where it
#: belongs rather than making the desktop stutter.
SOLVER_NICE = int(os.environ.get("CAMBER_NICE", "5"))


def budgeted_ranks(budget: float = None) -> int:
    """How many MPI ranks a run may use, from the CPU budget."""
    budget = CPU_BUDGET if budget is None else budget
    total = os.cpu_count() or 4
    return max(1, min(total, int(total * budget)))

FOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


@dataclass
class CaseConfig:
    """Domain, mesh and flow settings for one operating point."""

    #: Freestream speed. 0.0 is hover.
    v_inf: float = 0.0
    #: Air velocity past the vehicle as a NED body-axis vector, which is what
    #: forward flight at an attitude needs. Overrides `v_inf` when given.
    #: See `envelope.TestPoint.relative_wind_ned` for the sign.
    u_inf: tuple | None = None
    #: Domain extent as multiples of the vehicle's rotor span.
    domain_radii: float = 6.0
    domain_height_above: float = 5.0
    domain_height_below: float = 8.0
    #: Background cell size as a fraction of rotor radius. The disc must
    #: resolve across several cells or the momentum source smears.
    base_cell_per_radius: float = 0.25
    refinement_levels: int = 2
    n_iterations: int = 800
    #: Defaults to the CPU budget rather than a hardcoded count, so the same
    #: config behaves sensibly on a laptop and on a workstation.
    n_processors: int = 0
    #: ISA sea level.
    rho: float = 1.225
    nu: float = 1.4607e-5


def _write(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FOAM_HEADER.format(cls=cls, obj=obj) + "\n" + body + "\n")


def rotor_span(vehicle: Vehicle) -> float:
    """Largest distance from the origin to a rotor tip. Sizes the domain."""
    reach = [math.hypot(t.position.x, t.position.y) + getattr(t, "radius", 0.0)
             for t in vehicle.thrusters]
    return max(reach) if reach else 0.1


@dataclass
class RotorDisc:
    """One actuator disc: where it is, how big, and how hard it pushes."""

    name: str
    centre: Vec3
    radius: float
    #: Thrust in newtons, taken from L1. Positive along `direction`.
    thrust: float
    direction: Vec3
    #: Disc thickness, a numerical parameter: thin enough to be a disc, thick
    #: enough to contain several cells.
    thickness: float

    @property
    def volume(self) -> float:
        return math.pi * self.radius ** 2 * self.thickness


def discs_from(vehicle: Vehicle, thrusts: dict[str, float],
               cell: float, max_thickness: float | None = None) -> list[RotorDisc]:
    out: list[RotorDisc] = []
    for t in vehicle.thrusters:
        if not isinstance(t, Rotor):
            continue
        if t.name not in thrusts:
            raise KeyError(
                f"no thrust given for rotor {t.name!r}; the disc loading comes "
                f"from L1, so every rotor needs one"
            )
        out.append(RotorDisc(
            name=t.name, centre=t.position, radius=t.radius,
            thrust=thrusts[t.name], direction=t.direction(),
            thickness=(max_thickness if max_thickness is not None
                       else max(3.0 * cell, 0.1 * t.radius)),
        ))
    return out


class DiscGeometryError(ValueError):
    """The actuator disc intersects the airframe.

    Fatal rather than a warning. A momentum source injected into the cells
    surrounding the body makes the measured force on that body meaningless --
    it is reading the source, not a wake -- and the result looks entirely
    plausible: smooth convergence, sensible magnitudes, wrong sign. A campaign
    built on it fits an interference model to an artefact.
    """


class CaseBuilder:
    """Writes a complete, runnable case directory."""

    def __init__(self, vehicle: Vehicle, thrusts: dict[str, float],
                 config: CaseConfig | None = None) -> None:
        self.vehicle = vehicle
        self.config = config or CaseConfig()
        self.span = rotor_span(vehicle)
        radius = max((getattr(t, "radius", 0.0) for t in vehicle.thrusters),
                     default=0.05)
        thickness = self._safe_thickness(radius)
        # The near-field cell must be smaller than the disc is thick, or the
        # refinement region is thinner than the cells it is trying to refine
        # and snappy silently leaves it alone -- which is how disc zones ended
        # up holding two cells (CFD-08).
        self.cell = min(radius * self.config.base_cell_per_radius, thickness / 2.0)
        self.discs = discs_from(vehicle, thrusts, self.cell,
                                max_thickness=thickness)
        self._check_disc_clearance()

    @property
    def ranks(self) -> int:
        """MPI ranks for this case: explicit setting, else the CPU budget."""
        return self.config.n_processors or budgeted_ranks()

    def _body_z_extent(self) -> tuple[float, float]:
        lo, hi = float("inf"), float("-inf")
        for b in self.vehicle.all_bodies():
            h = b.shape.half_extents()
            lo = min(lo, b.position.z - h.z)
            hi = max(hi, b.position.z + h.z)
        return (lo, hi) if lo <= hi else (0.0, 0.0)

    def _safe_thickness(self, radius: float) -> float:
        """Thin enough to be a disc, and to stay clear of the airframe.

        Thickness was previously tied to the cell size, which is tied to the
        radius, so a large rotor produced a disc 0.75 R thick -- taller than
        the vehicle, swallowing the body it was supposed to blow on. The disc
        is a numerical device and should be thin; its thickness is now capped
        by the clearance to the nearest body so it cannot intersect one.
        """
        rotor_z = [t.position.z for t in self.vehicle.thrusters]
        if not rotor_z:
            return radius / 6.0
        plane = float(sum(rotor_z) / len(rotor_z))
        body_lo, body_hi = self._body_z_extent()
        if plane <= body_lo:
            clearance = body_lo - plane
        elif plane >= body_hi:
            clearance = plane - body_hi
        else:
            clearance = 0.0
        # R/3, not R/6. An actuator disc is a numerical device and its
        # thickness only sets how the momentum is spread axially, so it should
        # be chosen to be *resolvable* rather than to look thin.
        #
        # At R/6 the disc was one sixth of a background cell thick, and
        # snappyHexMesh's region refinement could not grip something smaller
        # than the cells it was refining: the zones came out holding **2 cells**
        # instead of ~150, the momentum source became a point force, and those
        # cases reported 66-70% download and tripped floating-point exceptions.
        # Every large-radius case in both campaigns failed this way.
        #
        # The clearance cap still binds first whenever the body is close, which
        # is what CFD-07 guarantees.
        return max(min(radius / 3.0, 1.6 * clearance), radius / 20.0)

    def _check_disc_clearance(self) -> None:
        body_lo, body_hi = self._body_z_extent()
        for d in self.discs:
            z0 = d.centre.z - d.thickness / 2.0
            z1 = d.centre.z + d.thickness / 2.0
            if min(z1, body_hi) - max(z0, body_lo) > 0:
                raise DiscGeometryError(
                    f"actuator disc {d.name!r} spans z {z0:+.4f}..{z1:+.4f} m "
                    f"while the airframe spans {body_lo:+.4f}..{body_hi:+.4f} m. "
                    f"They intersect, so a force measured on the body would be "
                    f"reading the momentum source rather than its wake. Move "
                    f"the rotors clear of the airframe in z."
                )

    @property
    def disc_cell(self) -> float:
        """Cell size inside the disc regions, refined below the base cell."""
        if not self.discs:
            return self.cell
        need = self.discs[0].thickness / 3.0
        cell = self.background_cell
        while cell > need and cell > self.cell / 16.0:
            cell /= 2.0
        return cell

    @property
    def disc_refinement_level(self) -> int:
        levels, cell = 0, self.background_cell
        while cell > self.disc_cell + 1e-12 and levels < 8:
            cell /= 2.0
            levels += 1
        return levels

    def cells_across_disc(self) -> float:
        """Cells through the disc thickness at the refined size."""
        return self.discs[0].thickness / self.disc_cell if self.discs else 0.0

    # -- domain ------------------------------------------------------------

    def bounds(self) -> tuple[Vec3, Vec3]:
        c = self.config
        r = self.span * c.domain_radii
        # z is DOWN (GEOM-02), so the wake goes to +z and needs the long side.
        return (Vec3(-r, -r, -self.span * c.domain_height_above),
                Vec3(r, r, self.span * c.domain_height_below))

    @property
    def background_cell(self) -> float:
        """Coarse cell far from the vehicle.

        The base cell is what the actuator discs need; applying it to the whole
        domain would cost about four million cells before snappy had refined
        anything, which does not fit in this machine's memory. So the
        background is coarser by 2^levels and snappy refines back down around
        the vehicle and the discs, where the gradients actually are. An earlier
        version instead divided the *base* cell by four, which is the same
        arithmetic applied to the wrong quantity: it coarsened the discs
        themselves to three cells each.
        """
        return self.cell * (2 ** self.config.refinement_levels)

    def block_mesh(self) -> str:
        lo, hi = self.bounds()
        c = self.background_cell
        nx = max(8, round((hi.x - lo.x) / c))
        ny = max(8, round((hi.y - lo.y) / c))
        nz = max(8, round((hi.z - lo.z) / c))
        v = [(lo.x, lo.y, lo.z), (hi.x, lo.y, lo.z), (hi.x, hi.y, lo.z),
             (lo.x, hi.y, lo.z), (lo.x, lo.y, hi.z), (hi.x, lo.y, hi.z),
             (hi.x, hi.y, hi.z), (lo.x, hi.y, hi.z)]
        verts = "\n".join(f"    ({x:.6g} {y:.6g} {z:.6g})" for x, y, z in v)
        return f"""scale 1;

vertices
(
{verts}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

boundary
(
    // z is down, so "inlet" is the top face the flow enters from in hover.
    top      {{ type patch;  faces ((0 3 2 1)); }}
    bottom   {{ type patch;  faces ((4 5 6 7)); }}
    sides    {{ type patch;  faces ((0 1 5 4) (2 3 7 6) (1 2 6 5) (0 4 7 3)); }}
);
"""

    def topo_set(self) -> str:
        """One cylindrical cellZone per rotor, for the momentum sources."""
        actions = []
        for d in self.discs:
            half = d.direction * (d.thickness / 2.0)
            p1 = d.centre - half
            p2 = d.centre + half
            actions.append(f"""    {{
        name    {d.name}Cells;
        type    cellSet;
        action  new;
        source  cylinderToCell;
        point1  ({p1.x:.6g} {p1.y:.6g} {p1.z:.6g});
        point2  ({p2.x:.6g} {p2.y:.6g} {p2.z:.6g});
        radius  {d.radius:.6g};
    }}
    {{
        name    {d.name}Zone;
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     {d.name}Cells;
    }}""")
        return "actions\n(\n" + "\n".join(actions) + "\n);\n"

    def fv_options(self) -> str:
        """Prescribed-thrust momentum sources, one per disc.

        `vectorSemiImplicitSource` injects a known total force spread over the
        zone volume, which is exactly an actuator disc of prescribed loading.
        The alternative, `rotorDiskSource`, computes its own blade-element
        loading -- using that to correct blade-element theory would be circular.
        """
        blocks = []
        rho = self.config.rho
        for d in self.discs:
            # TWO things are easy to get wrong here and neither shows up in a
            # residual plot.
            #
            # Sign: the source acts on the *fluid*, and it is the reaction to
            # the thrust. A rotor lifting a vehicle pushes air DOWNWARD, so the
            # source is opposite to the thrust direction. Getting this backwards
            # blows the wake up through the airframe and lifts it, which reads
            # as a large negative download -- physically impossible, and
            # invisible to any check that compares only wake speed.
            #
            # Scale: incompressible solvers divide the momentum equation by
            # density, so U-equation sources are kinematic, [m^4/s^2] = force
            # over rho, not newtons. See DECISIONS.md CFD-05 and CFD-06.
            fx = -d.direction.x * d.thrust / rho
            fy = -d.direction.y * d.thrust / rho
            fz = -d.direction.z * d.thrust / rho
            blocks.append(f"""{d.name}Source
{{
    type            vectorSemiImplicitSource;
    selectionMode   cellZone;
    cellZone        {d.name}Zone;
    volumeMode      absolute;
    sources
    {{
        U   ( ({fx:.6g} {fy:.6g} {fz:.6g}) 0 );
    }}
}}""")
        return "\n\n".join(blocks) + "\n"

    def controls(self) -> str:
        c = self.config
        return f"""application     simpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {c.n_iterations};
deltaT          1;
writeControl    timeStep;
writeInterval   {max(100, c.n_iterations // 4)};
purgeWrite      2;
writeFormat     ascii;
writePrecision  8;
runTimeModifiable true;

functions
{{
    // The quantity the whole tier exists to measure: the force the airframe
    // feels sitting in its own rotor wash. Installed thrust is the sum of the
    // prescribed disc thrusts minus this download.
    bodyForces
    {{
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   50;
        patches         (body);
        rho             rhoInf;
        rhoInf          {c.rho};
        CofR            (0 0 0);
    }}
    residuals
    {{
        type            solverInfo;
        libs            (utilityFunctionObjects);
        fields          (U p k omega);
        writeControl    timeStep;
        writeInterval   1;
    }}
}}
"""

    def background_cells(self) -> int:
        lo, hi = self.bounds()
        c = self.background_cell
        return int(round((hi.x - lo.x) / c) * round((hi.y - lo.y) / c)
                   * round((hi.z - lo.z) / c))

    def cells_per_disc(self) -> float:
        """Cell count in one actuator disc, at the refined size it gets.

        A momentum source spread over a handful of cells is a point force with
        a misleading name. Measured against the *refined* disc cell, not the
        base cell: the disc is deliberately thin -- its thickness is capped by
        the clearance to the airframe -- and is resolved by local refinement
        rather than by making the whole near field that fine.
        """
        if not self.discs:
            return 0.0
        return self.discs[0].volume / (self.disc_cell ** 3)

    def build(self, case_dir: Path, stl: Path) -> Path:
        """Write the whole case. `stl` is the airframe from camber.geometry.cad.

        The STL must be in **metres** (`cad.export(..., metres=True)`); a
        millimetre file is 1000x oversized and silently engulfs the domain.
        """
        case = Path(case_dir)
        if case.exists():
            shutil.rmtree(case)
        (case / "system").mkdir(parents=True)
        (case / "constant" / "triSurface").mkdir(parents=True)
        (case / "0").mkdir()

        shutil.copy(stl, case / "constant" / "triSurface" / "body.stl")

        _write(case / "system" / "blockMeshDict", "dictionary", "blockMeshDict",
               self.block_mesh())
        _write(case / "system" / "topoSetDict", "dictionary", "topoSetDict",
               self.topo_set())
        _write(case / "constant" / "fvOptions", "dictionary", "fvOptions",
               self.fv_options())
        _write(case / "system" / "controlDict", "dictionary", "controlDict",
               self.controls())
        _write(case / "system" / "fvSchemes", "dictionary", "fvSchemes",
               _FV_SCHEMES)
        _write(case / "system" / "fvSolution", "dictionary", "fvSolution",
               _FV_SOLUTION)
        _write(case / "system" / "decomposeParDict", "dictionary",
               "decomposeParDict",
               f"numberOfSubdomains {self.ranks};\n"
               f"method scotch;\n")
        _write(case / "system" / "snappyHexMeshDict", "dictionary",
               "snappyHexMeshDict", self._snappy())
        _write(case / "constant" / "transportProperties", "dictionary",
               "transportProperties",
               f"transportModel Newtonian;\nnu {self.config.nu:.6g};\n")
        # ESI (openfoam.com) names this `turbulenceProperties` with a
        # `RASModel` key; the Foundation fork (openfoam.org) calls the same
        # thing `momentumTransport` with `model`. The two forks diverged and
        # their dictionaries are not interchangeable -- this install is ESI
        # v2606, and getting it wrong costs a FATAL ERROR that names only the
        # missing file, not the reason.
        _write(case / "constant" / "turbulenceProperties", "dictionary",
               "turbulenceProperties",
               "simulationType RAS;\n\nRAS\n{\n    RASModel        kOmegaSST;\n"
               "    turbulence      on;\n    printCoeffs     on;\n}\n")
        self._write_fields(case)
        return case

    def near_field(self) -> tuple[Vec3, Vec3]:
        """Box enclosing the vehicle and one span of wake below it."""
        r = self.span * 1.35
        return (Vec3(-r, -r, -self.span * 0.8),
                Vec3(r, r, self.span * 2.5))

    def _disc_search_geometry(self) -> tuple[str, str]:
        """Each disc as a searchable cylinder, refined enough to resolve it.

        A thin disc needs cells across it or the momentum source is a sheet of
        one cell and the wake it makes is numerical. Refining only these four
        small cylinders is far cheaper than refining the near field to match.
        """
        level = self.disc_refinement_level
        geom, regions = [], []
        for d in self.discs:
            half = d.direction * (d.thickness / 2.0)
            p1, p2 = d.centre - half, d.centre + half
            geom.append(
                f"    {d.name}Disc\n"
                f"    {{\n"
                f"        type    searchableCylinder;\n"
                f"        point1  ({p1.x:.6g} {p1.y:.6g} {p1.z:.6g});\n"
                f"        point2  ({p2.x:.6g} {p2.y:.6g} {p2.z:.6g});\n"
                f"        radius  {d.radius:.6g};\n"
                f"    }}")
            regions.append(
                f"        {d.name}Disc {{ mode inside; "
                f"levels ((1e15 {level})); }}")
        return ("\n".join(geom) + "\n", "\n".join(regions) + "\n")

    def _snappy(self) -> str:
        lo, hi = self.bounds()
        nf_lo, nf_hi = self.near_field()
        disc_geometry, disc_regions = self._disc_search_geometry()
        # A point that is inside the fluid and outside the body: offset well
        # away from the vehicle, near the top of the domain.
        loc = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z * 0.8)
        return f"""castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    body.stl {{ type triSurfaceMesh; name body; }}
    nearField
    {{
        type    searchableBox;
        min     ({nf_lo.x:.6g} {nf_lo.y:.6g} {nf_lo.z:.6g});
        max     ({nf_hi.x:.6g} {nf_hi.y:.6g} {nf_hi.z:.6g});
    }}
{disc_geometry}}}

castellatedMeshControls
{{
    maxLocalCells   2000000;
    maxGlobalCells  8000000;
    minRefinementCells 10;
    nCellsBetweenLevels 3;
    resolveFeatureAngle 30;
    locationInMesh  ({loc.x:.6g} {loc.y:.6g} {loc.z:.6g});
    allowFreeStandingZoneFaces true;
    features ();
    refinementSurfaces
    {{
        body {{ level ({self.config.refinement_levels} {self.config.refinement_levels}); }}
    }}
    refinementRegions
    {{
        // Refine the near field to the base cell size: the vehicle, the discs
        // and the column of wake immediately beneath them. Everything further
        // out stays coarse, which is what keeps the case affordable.
        nearField {{ mode inside; levels ((1e15 {self.config.refinement_levels})); }}
{disc_regions}    }}
}}

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      30;
    nRelaxIter      5;
}}

addLayersControls
{{
    relativeSizes   true;
    layers          {{}}
    expansionRatio  1.2;
    finalLayerThickness 0.5;
    minThickness    0.1;
    nGrow           0;
    featureAngle    60;
    nRelaxIter      3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals  3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter      50;
}}

meshQualityControls
{{
    maxNonOrtho     65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave      80;
    minVol          1e-13;
    minTetQuality   1e-15;
    minArea         -1;
    minTwist        0.02;
    minDeterminant  0.001;
    minFaceWeight   0.02;
    minVolRatio     0.01;
    minTriangleTwist -1;
    nSmoothScale    4;
    errorReduction  0.75;
}}

mergeTolerance 1e-6;
"""

    def _write_fields(self, case: Path) -> None:
        c = self.config
        if c.u_inf is not None:
            u = [float(x) for x in c.u_inf]
        else:
            # Kept for the hover campaign, where it is always zero. As a
            # forward-flight freestream it had the wrong sign -- air arriving
            # from behind -- which is why u_inf exists.
            u = [float(c.v_inf), 0.0, 0.0]
        v = math.sqrt(sum(x * x for x in u))
        inlet = f"({u[0]:.6g} {u[1]:.6g} {u[2]:.6g})"
        if v > 1e-9:
            # Forward flight: freestream far-field. The hover boundaries below
            # hold *total* pressure at zero on every face, which is right when
            # the only flow is what the rotors induce and wrong in a 31 m/s
            # freestream: static pressure is pinned at -U^2/2 wherever air
            # enters and at 0 wherever it leaves, so the boundaries impose a
            # pressure drop across the domain that fights the flow. The first
            # forward-flight case ran that way; its pressure residual stalled at
            # 0.19 and never fell, and its body force was about twice what a
            # flat plate can produce. freestreamVelocity/Pressure switch between
            # inflow and outflow on the flux and hold static pressure at
            # freestream, which is what an external-aero far field is.
            far_u = f"type freestreamVelocity; freestreamValue uniform {inlet}; value uniform {inlet};"
            far_p = "type freestreamPressure; freestreamValue uniform 0; value uniform 0;"
        else:
            far_u = f"type inletOutlet; inletValue uniform {inlet}; value uniform {inlet};"
            far_p = "type totalPressure; p0 uniform 0; value uniform 0;"
        _write(case / "0" / "U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform {inlet};

boundaryField
{{
    top     {{ {far_u} }}
    bottom  {{ {far_u} }}
    sides   {{ {far_u} }}
    body    {{ type noSlip; }}
}}""")
        _write(case / "0" / "p", "volScalarField", "p", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{{
    top     {{ {far_p} }}
    bottom  {{ {far_p} }}
    sides   {{ {far_p} }}
    body    {{ type zeroGradient; }}
}}""")
        # Modest freestream turbulence; the wake generates its own.
        k = max(1e-4, 1.5 * (0.05 * max(v, 1.0)) ** 2)
        omega = max(1e-2, math.sqrt(k) / (0.1 * self.span))
        _write(case / "0" / "k", "volScalarField", "k", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform {k:.6g};

boundaryField
{{
    top     {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}
    bottom  {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}
    sides   {{ type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g}; }}
    body    {{ type kqRWallFunction; value uniform {k:.6g}; }}
}}""")
        _write(case / "0" / "omega", "volScalarField", "omega", f"""dimensions [0 0 -1 0 0 0 0];
internalField uniform {omega:.6g};

boundaryField
{{
    top     {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}
    bottom  {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}
    sides   {{ type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g}; }}
    body    {{ type omegaWallFunction; value uniform {omega:.6g}; }}
}}""")
        _write(case / "0" / "nut", "volScalarField", "nut", """dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;

boundaryField
{
    top     { type calculated; value uniform 0; }
    bottom  { type calculated; value uniform 0; }
    sides   { type calculated; value uniform 0; }
    body    { type nutkWallFunction; value uniform 0; }
}""")


_FV_SCHEMES = """ddtSchemes      { default steadyState; }
gradSchemes     { default cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes   { default limited corrected 0.33; }
wallDist        { method meshWave; }
"""

_FV_SOLUTION = """solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.01;
    }
}

SIMPLE
{
    consistent          yes;
    nNonOrthogonalCorrectors 1;
    residualControl
    {
        p       1e-4;
        U       1e-4;
        "(k|omega)" 1e-4;
    }
}

relaxationFactors
{
    equations { ".*" 0.9; }
}
"""
