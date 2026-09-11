"""One watertight surface for the whole airframe, generated frame included.

`cad.build` makes exact B-rep solids from primitives, which is right for STEP
and wrong for a topology-optimised frame: that is a 250,000-facet mesh, and
converting it to B-rep would take longer than generating it did. The CFD tier
and STL export need a *surface*, not a solid model, so this builds one from
triangles: each primitive tessellated, the generated frame loaded from its own
mesh, and the lot unioned.

The union is not optional. snappyHexMesh decides what is inside the airframe by
casting rays against the surface, and two overlapping closed shells -- a stack
box sitting on the frame plate -- give it regions that are inside one shell and
outside the other, which it resolves into internal faces nobody asked for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spec.shapes import Box, Cylinder, PointMass, Sphere, Tabulated


def frame_mesh_ned(shape: Tabulated):
    """A generated frame's mesh, in metres and NED, from its recorded source.

    The STL carries no unit or orientation; they are recovered by comparing
    its extents with the ones Camber measured when the mesh was made. GEOM-05
    ties them together: a millimetre file is Z-up (a half-turn from NED), a
    metre file is still NED.
    """
    import trimesh

    src = Path(shape.source) if shape.source else None
    if src is None or not src.exists():
        raise FileNotFoundError(
            f"the generated frame's mesh is not on disk ({shape.source!r}); "
            f"its mass properties are exact without it, but a surface is not")
    m = trimesh.load(src, force="mesh")
    ratio = float(np.max(m.extents)) / (2.0 * float(np.max(shape.extents)))
    if 500.0 < ratio < 2000.0:
        m.apply_scale(1e-3)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    elif not 0.5 < ratio < 2.0:
        raise ValueError(f"{src}: extents are {ratio:.3g}x what the spec records")
    return m


def _primitive(shape):
    import trimesh
    if isinstance(shape, Box):
        return trimesh.creation.box((shape.lx, shape.ly, shape.lz))
    if isinstance(shape, Cylinder):
        return trimesh.creation.cylinder(shape.radius, shape.length, sections=40)
    if isinstance(shape, Sphere):
        return trimesh.creation.icosphere(subdivisions=3, radius=shape.radius)
    return None


def airframe_surface(vehicle):
    """Every body with extent, placed and unioned: metres, NED."""
    import trimesh

    parts = []
    for b in vehicle.all_bodies():
        if isinstance(b.shape, PointMass):
            continue
        if isinstance(b.shape, Tabulated):
            parts.append(frame_mesh_ned(b.shape))    # already in vehicle axes
            continue
        m = _primitive(b.shape)
        if m is None:
            continue
        T = np.eye(4)
        T[:3, :3] = b.placement.rotation.matrix
        T[:3, 3] = b.position.as_list()
        m.apply_transform(T)
        parts.append(m)
    if not parts:
        raise ValueError(f"{vehicle.name!r} has no bodies with extent")
    if len(parts) == 1:
        return parts[0]
    return trimesh.boolean.union(parts)
