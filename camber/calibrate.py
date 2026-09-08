"""The closed loop: CFD corrects the cheap model, the optimiser searches that.

This is the piece ARCH-01 was designed around. CFD is far too slow to evaluate
candidates -- minutes to hours against microseconds -- so it never enters the
optimiser's inner loop. Instead a handful of CFD runs measure the *residual*
between what blade-element theory predicts and what the installed system
actually produces, that residual is fitted against the design variables, and
the correction is folded into L1. The optimiser then searches a cheap model
that has been taught where it is wrong.

The residual being modelled is interference: airframe download, rotor-rotor
interaction, blockage. BEMT cannot express any of it, and all of it varies with
layout, which is exactly what the optimiser changes.

**Why a Gaussian process** rather than a polynomial or a neural fit. Two
reasons, and the second matters more. With ten or twenty CFD runs the data is
tiny, and a GP is well-behaved in that regime where a flexible model would
overfit and then extrapolate confidently into nonsense. More importantly a GP
reports its own uncertainty, which turns "where should the next expensive run
go?" from a guess into arithmetic: run the case where the correction is least
known. Without that, a campaign spends hours re-confirming what it already
knew.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def rbf_kernel(a: np.ndarray, b: np.ndarray, length_scale: np.ndarray,
               variance: float) -> np.ndarray:
    """Squared-exponential kernel with one length scale per input."""
    a = np.atleast_2d(a) / length_scale
    b = np.atleast_2d(b) / length_scale
    d2 = (np.sum(a ** 2, axis=1)[:, None]
          + np.sum(b ** 2, axis=1)[None, :]
          - 2.0 * a @ b.T)
    return variance * np.exp(-0.5 * np.maximum(d2, 0.0))


@dataclass
class GaussianProcess:
    """Exact GP regression. Small data, so the cubic solve is free.

    Inputs are standardised internally: the design variables are arm lengths in
    metres and dimensionless area ratios, which differ by orders of magnitude.
    A single shared length scale across raw units would be meaningless.
    """

    length_scale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-3

    x_train: np.ndarray | None = field(default=None, init=False)
    y_train: np.ndarray | None = field(default=None, init=False)
    _mean: np.ndarray | None = field(default=None, init=False)
    _scale: np.ndarray | None = field(default=None, init=False)
    _y_mean: float = field(default=0.0, init=False)
    _chol: np.ndarray | None = field(default=None, init=False)
    _alpha: np.ndarray | None = field(default=None, init=False)

    def log_marginal_likelihood(self) -> float:
        """Evidence for the current hyperparameters, given the fitted data."""
        if self._chol is None:
            raise RuntimeError("fit first")
        y = self.y_train - self._y_mean
        return float(-0.5 * y @ self._alpha
                     - np.sum(np.log(np.diag(self._chol)))
                     - 0.5 * len(y) * math.log(2.0 * math.pi))

    def tune(self, x: np.ndarray, y: np.ndarray,
             length_scales=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
             noises=(1e-5, 1e-4, 1e-3, 1e-2)) -> "GaussianProcess":
        """Pick hyperparameters by marginal likelihood rather than by hand.

        The interference response spans three orders of magnitude -- 0.04% for
        a wide airframe to 40% when the discs sit over the body -- and no
        single hand-chosen length scale describes both ends. Choosing by
        evidence is the standard remedy and costs nothing at this data size:
        the whole grid is a few dozen Cholesky factorisations of a 10x10
        matrix.

        The signal variance is set from the data's own spread rather than
        searched, since a GP whose prior variance disagrees with its
        observations reports uncertainty that means nothing.
        """
        y = np.asarray(y, dtype=float).ravel()
        best, best_score = None, -np.inf
        variance = max(float(np.var(y)), 1e-8)
        for ls in length_scales:
            for noise in noises:
                trial = GaussianProcess(length_scale=ls, variance=variance,
                                        noise=noise)
                try:
                    trial.fit(x, y)
                    score = trial.log_marginal_likelihood()
                except np.linalg.LinAlgError:
                    continue
                if score > best_score:
                    best, best_score = trial, score
        if best is None:
            raise RuntimeError("no hyperparameters produced a stable fit")
        self.length_scale = best.length_scale
        self.variance = best.variance
        self.noise = best.noise
        return self.fit(x, y)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if len(x) != len(y):
            raise ValueError(f"{len(x)} inputs against {len(y)} targets")
        if len(x) < 1:
            raise ValueError("need at least one observation")

        self._mean = x.mean(axis=0)
        spread = x.std(axis=0)
        # A variable that never varied carries no information; scaling it by
        # zero would divide by zero, so it is left alone and contributes
        # nothing to the distance.
        self._scale = np.where(spread > 1e-12, spread, 1.0)
        self._y_mean = float(y.mean())

        xs = (x - self._mean) / self._scale
        ls = np.full(xs.shape[1], self.length_scale)
        k = rbf_kernel(xs, xs, ls, self.variance)
        k[np.diag_indices_from(k)] += self.noise

        self._chol = np.linalg.cholesky(k)
        self._alpha = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, y - self._y_mean))
        self.x_train, self.y_train = xs, y
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation."""
        if self._chol is None:
            raise RuntimeError("fit the model before predicting")
        x = np.atleast_2d(np.asarray(x, dtype=float))
        xs = (x - self._mean) / self._scale
        ls = np.full(xs.shape[1], self.length_scale)

        k_star = rbf_kernel(xs, self.x_train, ls, self.variance)
        mean = k_star @ self._alpha + self._y_mean

        v = np.linalg.solve(self._chol, k_star.T)
        var = self.variance - np.sum(v ** 2, axis=0)
        return mean, np.sqrt(np.maximum(var, 0.0))


def design_features(vehicle) -> np.ndarray:
    """The layout numbers interference actually depends on.

    Chosen from the physics rather than by throwing every variable at the
    model. Interference is governed by how much airframe sits under the discs,
    how close the discs are to each other, and how far the body is below them:

      0. blockage      -- body planform area / total disc area
      1. spacing       -- rotor separation / diameter (wake overlap)
      2. body offset   -- body-to-disc vertical gap / rotor radius
      3. disc loading  -- weight / total disc area, N/m^2

    Four features against ten or twenty runs is already generous; more would
    make the GP interpolate noise.
    """
    from .control import G
    from .spec.vehicle import Rotor

    rotors = [t for t in vehicle.thrusters if isinstance(t, Rotor)]
    if not rotors:
        raise ValueError("no rotors: interference is undefined")

    disc_area = sum(math.pi * r.radius ** 2 for r in rotors)
    radius = float(np.mean([r.radius for r in rotors]))

    bodies = vehicle.all_bodies()
    planform = sum(b.shape.half_extents().x * b.shape.half_extents().y * 4.0
                   for b in bodies)

    centres = np.array([[r.position.x, r.position.y] for r in rotors])
    if len(centres) > 1:
        gaps = [np.linalg.norm(centres[i] - centres[j])
                for i in range(len(centres)) for j in range(i + 1, len(centres))]
        spacing = float(min(gaps)) / (2.0 * radius)
    else:
        spacing = 10.0

    body_z = float(np.mean([b.position.z for b in bodies])) if bodies else 0.0
    disc_z = float(np.mean([r.position.z for r in rotors]))
    offset = abs(body_z - disc_z) / radius

    mass = vehicle.mass_properties().mass
    loading = mass * G / disc_area if disc_area > 0 else 0.0

    return np.array([planform / disc_area, spacing, offset, loading])


@dataclass
class InterferenceModel:
    """Learned correction from isolated-rotor thrust to installed thrust.

    Stored as a *fraction*, not an absolute force: interference scales with
    thrust, so a design twice as heavy loses roughly twice as much to download.
    A model fitted in newtons would have to relearn that at every scale.
    """

    gp: GaussianProcess = field(default_factory=lambda: GaussianProcess(
        length_scale=1.0, variance=0.02, noise=1e-4))
    fitted: bool = False
    n_observations: int = 0
    #: Set when the GP was fitted to sqrt(loss); predictions are squared back.
    _sqrt_space: bool = False

    def fit(self, features: np.ndarray, loss_fraction: np.ndarray,
            tune: bool = True) -> "InterferenceModel":
        """`loss_fraction` is (isolated - installed) / isolated, per case."""
        if tune:
            self.gp.tune(features, loss_fraction)
        else:
            self.gp.fit(features, loss_fraction)
        self.fitted = True
        self.n_observations = len(np.atleast_2d(features))
        return self

    def correction(self, vehicle) -> tuple[float, float]:
        """Fractional thrust loss for this layout, and its uncertainty.

        Returns (0, 0) with no confidence when unfitted, so an uncalibrated
        model is transparently a no-op rather than a silent guess.
        """
        if not self.fitted:
            return 0.0, 0.0
        mean, std = self.gp.predict(design_features(vehicle)[None, :])
        if self._sqrt_space:
            # Predictions live in sqrt space; square back, and propagate the
            # uncertainty through the same map rather than reporting a sigma
            # that belongs to a different quantity.
            m = max(float(mean[0]), 0.0)
            return m ** 2, 2.0 * m * float(std[0])
        return float(mean[0]), float(std[0])

    def as_penalty(self, vehicle, threshold: float = 0.02) -> float:
        """Interference above `threshold`, as a continuous optimiser penalty.

        This is the recommended use, and the only one the validation supports.
        Across 43 CFD cases the model orders designs correctly 92.7% of the
        time and classifies against a 2% threshold with **no false clears** --
        it never says a layout is fine when it is not. What it cannot do is
        state *how much* download a heavily overlapped layout suffers: mean
        error above 15% loss is 9.25% and the worst case is 48.8%.

        Ordering and a reliable "unacceptable" verdict are what a search needs.
        A corrected thrust would imply an accuracy this model does not have.
        See DECISIONS.md CAL-02.
        """
        loss, _ = self.correction(vehicle)
        return max(0.0, loss - threshold)

    def installed_thrust(self, vehicle, isolated_thrust: float) -> float:
        """Isolated thrust reduced by the predicted loss.

        Prefer `as_penalty()`. This is accurate enough in the flat and
        transition bands but not in the steep one, where the model's magnitude
        error reaches tens of per cent (CAL-02).
        """
        loss, _ = self.correction(vehicle)
        # Clamp: a fit extrapolating past total loss, or past no loss at all,
        # has left the data it was trained on and should not invent physics.
        return isolated_thrust * (1.0 - min(max(loss, -0.2), 0.6))

    def next_case(self, candidates: list, n: int = 1) -> list[int]:
        """Which candidate layouts to spend the next CFD runs on.

        Pure uncertainty sampling: run where the correction is least known.
        With runs costing hours, choosing them by where the model is already
        confident is the most expensive way to learn nothing.
        """
        if not self.fitted:
            # Nothing known yet, so spread the first batch over the extremes of
            # the feature space rather than clustering near one design.
            feats = np.array([design_features(v) for v in candidates])
            spread = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-12)
            return list(np.argsort(-np.linalg.norm(spread, axis=1))[:n])
        stds = [self.correction(v)[1] for v in candidates]
        return list(np.argsort(-np.asarray(stds))[:n])


def save_model(model: InterferenceModel, path: str | Path) -> Path:
    """Persist a fitted model with the data it was fitted on.

    The training inputs are stored alongside the hyperparameters rather than
    just the fitted state, because a correction is only meaningful within the
    region it was measured in, and a future reader needs to be able to ask
    whether their design is inside it.
    """
    import json

    path = Path(path)
    if not model.fitted:
        raise ValueError("refusing to save an unfitted model")
    path.write_text(json.dumps({
        "length_scale": model.gp.length_scale,
        "variance": model.gp.variance,
        "noise": model.gp.noise,
        "sqrt_space": model._sqrt_space,
        "n_observations": model.n_observations,
        "x": np.asarray(model.gp.x_train).tolist(),
        "y": np.asarray(model.gp.y_train).tolist(),
        "x_mean": np.asarray(model.gp._mean).tolist(),
        "x_scale": np.asarray(model.gp._scale).tolist(),
    }, indent=2))
    return path


def load_model(path: str | Path) -> InterferenceModel:
    """Rebuild a saved model. Refits from the stored data, which is cheap and
    avoids serialising a Cholesky factor that may not survive a numpy change."""
    import json

    d = json.loads(Path(path).read_text())
    gp = GaussianProcess(d["length_scale"], d["variance"], d["noise"])
    xs = np.asarray(d["x"]) * np.asarray(d["x_scale"]) + np.asarray(d["x_mean"])
    gp.fit(xs, np.asarray(d["y"]))
    model = InterferenceModel(gp=gp)
    model.fitted = True
    model.n_observations = d["n_observations"]
    model._sqrt_space = d["sqrt_space"]
    return model


def loss_fraction(isolated_total_n: float, body_force_z_n: float) -> float:
    """Turn one CFD result into the quantity the model learns.

    `body_force_z_n` is the z force on the airframe from the `forces` function
    object. z is down (GEOM-02), so a positive value is the body being pushed
    *down* by its own rotor wash -- a loss. Installed thrust is the prescribed
    disc thrust less that download.
    """
    if isolated_total_n <= 0:
        raise ValueError("isolated thrust must be positive")
    return body_force_z_n / isolated_total_n
