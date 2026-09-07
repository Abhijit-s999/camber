"""Validate the blade-element model against the UIUC wind-tunnel database.

This is the gate described in DECISIONS AI-05: the model is checked against
published measurements, not against tests invented alongside it. Until this
reports an error bar, any thrust figure Camber produces is decoration.

What is being measured is the *whole chain* -- geometry parsing, section data,
induction solve, tip loss, integration -- against reality. The dominant error
term is known and named: UIUC publishes chord and twist but not the aerofoil,
so the section is an assumption, and section choice moves predicted thrust by
tens of per cent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..catalogue import uiuc
from .airfoil import build_polar
from .bemt import BladeDefinition, analyse


@dataclass
class PointError:
    key: str
    rpm: float
    advance_ratio: float
    ct_measured: float
    cp_measured: float
    ct_predicted: float
    cp_predicted: float
    #: Peak measured CT of the run this point belongs to. See `ct_error_scaled`.
    ct_peak: float = 0.0
    cp_peak: float = 0.0

    @property
    def ct_error(self) -> float:
        return (self.ct_predicted - self.ct_measured) / self.ct_measured

    @property
    def cp_error(self) -> float:
        return (self.cp_predicted - self.cp_measured) / self.cp_measured

    @property
    def ct_error_scaled(self) -> float:
        """Error as a fraction of the run's *peak* CT, not the local value.

        The honest metric, and the one to quote. CT falls to zero as the
        propeller approaches its zero-thrust advance ratio, so a relative error
        computed against the local value diverges there for arithmetic reasons
        that say nothing about the model: measured points below CT = 0.01
        showed a mean relative error of 800 per cent while points above
        CT = 0.08 sat at 10 per cent, using the very same predictions.

        Scaling by the peak asks the question that matters -- how many newtons
        out is this, against how many the propeller makes -- and does not
        explode where the quantity vanishes.
        """
        return (self.ct_predicted - self.ct_measured) / self.ct_peak if self.ct_peak else float("nan")

    @property
    def cp_error_scaled(self) -> float:
        return (self.cp_predicted - self.cp_measured) / self.cp_peak if self.cp_peak else float("nan")


@dataclass
class ValidationReport:
    airfoil: str
    points: list[PointError] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def _stats(self, values: np.ndarray) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "p90_abs": float(np.percentile(np.abs(values), 90)),
            "within_10pct": float(np.mean(np.abs(values) < 0.10)),
            "within_20pct": float(np.mean(np.abs(values) < 0.20)),
        }

    def summary(self) -> dict:
        if not self.points:
            return {"points": 0}
        return {
            "airfoil": self.airfoil,
            "propellers": len({p.key for p in self.points}),
            "points": len(self.points),
            "ct_scaled": self._stats(np.array([p.ct_error_scaled for p in self.points])),
            "cp_scaled": self._stats(np.array([p.cp_error_scaled for p in self.points])),
            "ct_relative": self._stats(np.array([p.ct_error for p in self.points])),
            "cp_relative": self._stats(np.array([p.cp_error for p in self.points])),
        }

    def by_propeller(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for p in self.points:
            out.setdefault(p.key, []).append(abs(p.ct_error))
        return {k: float(np.median(v)) for k, v in out.items()}


def validate(
    root: Path | str = "data/raw/UIUC-propDB",
    airfoil: str | None = None,
    blades: int = 2,
    stations: int = 30,
    j_min: float = 0.10,
    keys: list[str] | None = None,
    max_props: int | None = None,
) -> ValidationReport:
    """Compare predicted CT and CP against every usable measured point.

    `j_min` excludes the near-static points. At very low advance ratio the
    induction factor is defined against a vanishing freestream and the
    formulation degenerates; static thrust needs its own treatment and is not
    claimed here.
    """
    from .airfoil import DEFAULT_SECTION
    airfoil = airfoil or DEFAULT_SECTION
    db = uiuc.load(root)
    polar = build_polar(airfoil)
    report = ValidationReport(airfoil=airfoil)

    selected = keys if keys is not None else db.validation_keys
    if max_props:
        selected = selected[:max_props]

    for key in selected:
        geometry = db.geometry[key]
        try:
            blade = BladeDefinition.from_uiuc(geometry, blades).resample(stations)
        except ValueError as exc:
            report.skipped.append(f"{key}: {exc}")
            continue

        diameter = geometry.diameter_m
        for run in db.runs_for(key):
            n = run.n_rev_s
            if n <= 0:
                continue
            ct_peak = max((p.ct for p in run.points), default=0.0)
            cp_peak = max((p.cp for p in run.points), default=0.0)
            if ct_peak <= 0 or cp_peak <= 0:
                continue
            for point in run.points:
                if point.advance_ratio < j_min or point.ct <= 1e-4:
                    continue
                v = point.advance_ratio * n * diameter
                try:
                    result = analyse(blade, v, run.rpm, polar)
                except Exception:  # noqa: BLE001
                    continue
                if not result.converged or result.ct <= 0:
                    continue
                report.points.append(PointError(
                    key, run.rpm, point.advance_ratio,
                    point.ct, point.cp, result.ct, result.cp,
                    ct_peak=ct_peak, cp_peak=cp_peak,
                ))
    return report
