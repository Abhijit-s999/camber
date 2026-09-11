"""Build a MuJoCo model (MJCF) from a Camber vehicle.

The airframe is one rigid free body. Its mass and inertia are Camber's, stated
explicitly (`inertiafromgeom="false"`), so MuJoCo never re-derives them from the
visual geometry: a generated frame is a 250,000-facet mesh whose convex hull is
a solid slab, and letting MuJoCo integrate that would give the simulator an
aircraft several times heavier than the one designed.

Geometry is for looking at and for hitting the ground with. Rotor discs are
drawn but do not collide, or a quadcopter sitting on the floor would be resting
on four transparent props.
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from . import frames
from ..spec.shapes import Box, Cylinder, PointMass, Sphere, Tabulated
from ..spec.vehicle import Rotor


def _fmt(a) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.ravel(a))


def _quat(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(matrix, dtype=float).ravel())
    return q


def _align_z(direction: np.ndarray) -> np.ndarray:
    """Quaternion turning +z onto `direction`, for discs and bells."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    h = math.acos(c) / 2.0
    return np.concatenate([[math.cos(h)], math.sin(h) * axis])


def mesh_convention(path: Path, extents_m) -> tuple[float, np.ndarray]:
    """(scale, quat) that put a generated frame's STL into the FLU body frame.

    An STL carries no unit and no orientation, so they are inferred from the
    file against the extents Camber measured when it made the mesh. GEOM-05
    ties the two together: a millimetre export is Z-up -- a half-turn about x
    from NED, which *is* FLU -- and a metre export is still NED and needs the
    same half-turn applied here.
    """
    import trimesh
    m = trimesh.load(path, force="mesh")
    ratio = float(np.max(m.extents)) / (2.0 * float(np.max(extents_m)))
    if 500.0 < ratio < 2000.0:
        return 1e-3, np.array([1.0, 0.0, 0.0, 0.0])      # mm, already Z-up
    if 0.5 < ratio < 2.0:
        return 1.0, np.array([0.0, 1.0, 0.0, 0.0])       # m, NED -> FLU
    raise ValueError(
        f"{path}: extents are {ratio:.3g}x what the spec records; the mesh is "
        f"neither the millimetre nor the metre export of this frame")


def _geom(name: str, shape, pos, rot, *, rgba: str, collide: bool) -> str:
    c = "" if collide else ' contype="0" conaffinity="0"'
    q = _fmt(_quat(rot))
    if isinstance(shape, Box):
        size = _fmt([shape.lx / 2, shape.ly / 2, shape.lz / 2])
        return (f'<geom name={quoteattr(name)} type="box" size="{size}" '
                f'pos="{_fmt(pos)}" quat="{q}" rgba="{rgba}"{c}/>')
    if isinstance(shape, Cylinder):
        size = _fmt([shape.radius, shape.length / 2])
        return (f'<geom name={quoteattr(name)} type="cylinder" size="{size}" '
                f'pos="{_fmt(pos)}" quat="{q}" rgba="{rgba}"{c}/>')
    if isinstance(shape, Sphere):
        return (f'<geom name={quoteattr(name)} type="sphere" '
                f'size="{shape.radius:.9g}" pos="{_fmt(pos)}" rgba="{rgba}"{c}/>')
    return ""                                            # point mass: invisible


def build_mjcf(vehicle, *, course=None, timestep: float = 0.002,
               start=(0.0, 0.0, 0.15)) -> str:
    """The vehicle, a floor and optionally a gate course, as MJCF XML."""
    mp = vehicle.mass_properties()
    cg = frames.vec(mp.cg)
    I = frames.tensor(mp.inertia)
    full = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]

    assets, geoms = [], []
    for b in vehicle.all_bodies():
        pos = frames.vec(b.position)
        rot = frames.rotation(b.placement.rotation.matrix)
        if isinstance(b.shape, Tabulated):
            src = Path(b.shape.source) if b.shape.source else None
            if src is None or not src.exists():
                # No mesh on disk: draw the bounding box, flagged by colour,
                # rather than fail -- the mass properties are exact regardless.
                e = b.shape.extents
                geoms.append(_geom(b.name, Box(2 * e[0], 2 * e[1], 2 * e[2]),
                                   pos, rot, rgba="0.9 0.4 0.3 0.35",
                                   collide=True))
                continue
            scale, q = mesh_convention(src, b.shape.extents)
            assets.append(f'<mesh name="{b.name}_mesh" file={quoteattr(str(src.resolve()))} '
                          f'scale="{scale} {scale} {scale}"/>')
            # The mesh is written in vehicle coordinates already, so it sits
            # at the body origin -- not at the frame's centroid, which is where
            # the Tabulated body's placement points.
            geoms.append(f'<geom name={quoteattr(b.name)} type="mesh" '
                         f'mesh="{b.name}_mesh" quat="{_fmt(q)}" '
                         f'rgba="0.13 0.14 0.15 1" material="carbon"/>')
            continue
        geoms.append(_geom(b.name, b.shape, pos, rot,
                           rgba="0.55 0.6 0.66 1", collide=True))

    for t in vehicle.thrusters:
        pos = frames.vec(t.position)
        d = frames.vec(t.direction())
        q = _fmt(_align_z(d))
        if isinstance(t, Rotor):
            spin = "0.25 0.55 0.85 0.28" if t.spin.value > 0 else "0.85 0.35 0.25 0.28"
            geoms.append(f'<geom name={quoteattr(t.name + "_disc")} type="cylinder" '
                         f'size="{t.radius:.6g} 0.0015" pos="{_fmt(pos)}" quat="{q}" '
                         f'rgba="{spin}" contype="0" conaffinity="0"/>')
        geoms.append(f'<geom name={quoteattr(t.name + "_bell")} type="cylinder" '
                     f'size="0.012 0.008" pos="{_fmt(pos - 0.008 * d)}" quat="{q}" '
                     f'rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0"/>')
        geoms.append(f'<site name={quoteattr(t.name)} pos="{_fmt(pos)}" quat="{q}" '
                     f'size="0.004" rgba="1 1 0 0.6"/>')

    gates = course.mjcf() if course is not None else ""
    return f"""<mujoco model={quoteattr(vehicle.name)}>
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{timestep}" gravity="0 0 -9.80665" density="0" viscosity="0"
          integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-22"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.38" diffuse="0.6 0.6 0.6"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.16 0.18 0.2" rgb2="0.2 0.22 0.25"/>
    <texture name="sky" type="skybox" builtin="gradient" width="512" height="512"
             rgb1="0.55 0.66 0.78" rgb2="0.12 0.14 0.18"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0.08"/>
    <material name="carbon" specular="0.6" shininess="0.5" rgba="0.13 0.14 0.15 1"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" directional="true" castshadow="true"
           diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="60 60 0.1" material="floor"/>
    {gates}
    <body name="airframe" pos="{_fmt(start)}">
      <freejoint name="root"/>
      <inertial pos="{_fmt(cg)}" mass="{mp.mass:.9g}" fullinertia="{_fmt(full)}"/>
      <site name="imu" pos="{_fmt(cg)}" size="0.003"/>
      <camera name="chase" mode="trackcom" pos="-1.6 0 0.55" xyaxes="0 -1 0 0.35 0 1"/>
      <camera name="fpv" pos="{_fmt(cg + [0.05, 0, 0.02])}" xyaxes="0 -1 0 0.5 0 1"/>
      {chr(10).join('      ' + g for g in geoms if g)}
    </body>
  </worldbody>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
  </sensor>
</mujoco>
"""
