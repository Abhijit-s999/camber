"""Section lift and drag, tabulated once and interpolated in the inner loop.

Two things make this more than a thin wrapper around NeuralFoil.

**Post-stall.** A propeller blade near the hub at low advance ratio runs at
angles of attack of 30, 60, even 90 degrees. NeuralFoil, like the XFOIL data it
learned from, is meaningful only while the flow is attached -- roughly to 20
degrees. Asking it about 60 degrees returns a number, and that number is
fiction. Beyond stall the section is extended with Viterna's flat-plate model,
which is the standard treatment and is at least the right shape.

**Speed.** The optimiser evaluates blade elements millions of times. NeuralFoil
costs about 12 microseconds per point; a bilinear lookup costs a fraction of
that, and the table is built once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Default blade section. Chosen by sensitivity study over all 115 validation
#: propellers, not by taste: it gives a CT bias of +7.1% against Clark-Y's
#: -15.9%. It remains an assumption -- UIUC does not publish the aerofoil.
#: See DECISIONS.md PHYS-02.
DEFAULT_SECTION = "e63"

#: Beyond this, XFOIL-derived data is not trustworthy and Viterna takes over.
STALL_LIMIT = math.radians(20.0)
#: Below this the blend is pure NeuralFoil.
BLEND_START = math.radians(14.0)


def viterna(alpha: np.ndarray, cl_stall: float, cd_stall: float,
            alpha_stall: float, aspect_ratio: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat-plate post-stall extension (Viterna & Corrigan, 1981).

    The deep-stall limit is a flat plate: lift from the cross-flow component,
    drag approaching that of a normal plate. Matching at the stall angle keeps
    the curve continuous.

    Two corrections to the textbook formula, both needed because a propeller
    blade element genuinely visits the whole circle and the published form does
    not survive there:

    * The lift correction term carries ``cos^2(a)/sin(a)``, which is singular
      at a = 0 and a = pi. Left alone it returns |CL| in the hundreds of
      thousands at 180 degrees. It is a near-stall correction, so it is damped
      out with a window that vanishes by 90 degrees, and its denominator is
      floored at the stall angle.
    * The drag correction ``b2 cos(a)`` goes *negative* for a > 90 degrees.
      Negative drag is unphysical and lets a BEMT solve extract energy from
      nothing, so the term is clipped at zero and a small reversed-flow floor
      is applied instead.
    """
    cd_max = 1.11 + 0.018 * aspect_ratio
    sa, ca = math.sin(alpha_stall), math.cos(alpha_stall)

    a2 = (cl_stall - cd_max * sa * ca) * sa / max(ca ** 2, 1e-6)
    b2 = (cd_stall - cd_max * sa ** 2) / max(ca, 1e-6)

    s, c = np.sin(alpha), np.cos(alpha)

    # Window: 1 at stall, 0 by 90 degrees and beyond.
    window = np.clip((np.pi / 2 - np.abs(alpha)) / (np.pi / 2 - alpha_stall), 0.0, 1.0)
    window = window ** 2

    cl = cd_max / 2.0 * np.sin(2.0 * alpha)
    cl = cl + a2 * c ** 2 / np.maximum(np.abs(s), sa) * window

    cd = cd_max * s ** 2 + np.maximum(b2 * c, 0.0)
    # A section in fully reversed flow still has pressure drag.
    cd = np.maximum(cd, 0.05)
    return cl, cd


@dataclass
class TabulatedPolar:
    """Bilinear lookup over (alpha, log10 Re), with Viterna wings."""

    alphas: np.ndarray
    log_res: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    name: str = "polar"

    def __call__(self, alpha, reynolds):
        """CL and CD at angle of attack (rad) and Reynolds number."""
        alpha = np.asarray(alpha, dtype=float)
        lre = np.log10(np.clip(np.asarray(reynolds, dtype=float), 1e3, 1e8))

        # Wrap into [-pi, pi]; a blade element can genuinely see any angle.
        a = (alpha + np.pi) % (2 * np.pi) - np.pi

        ia = np.clip(np.searchsorted(self.alphas, a) - 1, 0, len(self.alphas) - 2)
        ir = np.clip(np.searchsorted(self.log_res, lre) - 1, 0, len(self.log_res) - 2)

        a0, a1 = self.alphas[ia], self.alphas[ia + 1]
        r0, r1 = self.log_res[ir], self.log_res[ir + 1]
        ta = np.clip((a - a0) / (a1 - a0), 0.0, 1.0)
        tr = np.clip((lre - r0) / (r1 - r0), 0.0, 1.0)

        def blend(grid):
            g00 = grid[ia, ir]
            g10 = grid[ia + 1, ir]
            g01 = grid[ia, ir + 1]
            g11 = grid[ia + 1, ir + 1]
            return ((1 - ta) * (1 - tr) * g00 + ta * (1 - tr) * g10
                    + (1 - ta) * tr * g01 + ta * tr * g11)

        return blend(self.cl), blend(self.cd)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=8)
def build_polar(airfoil: str = DEFAULT_SECTION,
                n_alpha: int = 361,
                re_decades: tuple[float, ...] = (4.0, 4.5, 5.0, 5.5, 6.0),
                model_size: str = "medium") -> TabulatedPolar:
    """Tabulate one section over the full -180..180 range.

    The default section is chosen by measurement (PHYS-02), not preference:
    across all 115 validation propellers it gives the smallest CT bias of the
    candidates tried. It is still an approximation -- real blades vary section
    along the span, high-camber inboard and thin outboard -- and that
    approximation is the single largest error term in this tier.
    """
    import aerosandbox as asb
    import neuralfoil as nf

    af = asb.Airfoil(airfoil)
    alphas = np.linspace(-np.pi, np.pi, n_alpha)
    log_res = np.array(re_decades)

    cl = np.zeros((n_alpha, len(log_res)))
    cd = np.zeros((n_alpha, len(log_res)))

    attached = np.abs(alphas) <= STALL_LIMIT
    deg = np.degrees(alphas[attached])

    for j, lre in enumerate(log_res):
        re = np.full(deg.shape, 10.0 ** lre)
        aero = nf.get_aero_from_airfoil(af, alpha=deg, Re=re, model_size=model_size)
        cl_att = np.asarray(aero["CL"], dtype=float)
        cd_att = np.asarray(aero["CD"], dtype=float)

        # Anchor Viterna at the positive stall angle.
        k = int(np.argmin(np.abs(alphas[attached] - STALL_LIMIT)))
        cl_v, cd_v = viterna(alphas, cl_att[k], cd_att[k], STALL_LIMIT)

        full_cl = cl_v.copy()
        full_cd = cd_v.copy()
        full_cl[attached] = cl_att
        full_cd[attached] = cd_att

        # Blend the two over the last few degrees so the join has no kink --
        # a discontinuity here makes the BEMT root-find chatter.
        w = _smoothstep((np.abs(alphas) - BLEND_START) / (STALL_LIMIT - BLEND_START))
        band = np.abs(alphas) <= STALL_LIMIT
        full_cl[band] = (1 - w[band]) * full_cl[band] + w[band] * cl_v[band]
        full_cd[band] = (1 - w[band]) * full_cd[band] + w[band] * cd_v[band]

        cl[:, j] = full_cl
        cd[:, j] = np.maximum(full_cd, 1e-4)

    return TabulatedPolar(alphas, log_res, cl, cd, name=airfoil)
