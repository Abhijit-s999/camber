"""NSGA-II over a DesignProblem.

Multi-objective because the objectives genuinely conflict: endurance wants a
big slow disc and a heavy pack, speed wants neither. Collapsing that into one
weighted score hides the trade-off the user most needs to see, and buries the
weighting -- an arbitrary choice made by whoever wrote the line -- inside the
answer. A Pareto front hands the trade-off back to the person who should be
making it.

**Constrained domination** is what keeps the search honest. Comparing two
candidates: a feasible one always beats an infeasible one; between two
infeasible ones the smaller total violation wins; between two feasible ones the
usual Pareto rule applies. Without this, a design that violates a clearance
constraint but scores brilliantly on endurance dominates the front and drags
the population into the infeasible region, where the objectives are meaningless
because the aircraft cannot be built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby

import numpy as np

from .pareto import crowding_distance, non_dominated_sort
from .problem import DesignProblem, Evaluation


@dataclass
class Result:
    """The final front, and enough history to see whether it converged."""

    x: np.ndarray
    objectives: np.ndarray
    evaluations: list[Evaluation]
    front_indices: list[int]
    history: list[dict] = field(default_factory=list)

    @property
    def front_x(self) -> np.ndarray:
        return self.x[self.front_indices]

    @property
    def front_objectives(self) -> np.ndarray:
        return self.objectives[self.front_indices]

    @property
    def feasible_count(self) -> int:
        return sum(1 for e in self.evaluations if e.feasible)


def _constrained_ranks(evals: list[Evaluation], objectives: np.ndarray) -> list[list[int]]:
    """Fronts under constrained domination.

    Infeasible candidates are placed strictly after every feasible one, so the
    objective values of an unbuildable design never compete with those of a
    real one. Among themselves they are ranked by violation alone, one front
    per level: the smaller violation dominates. They used to share a single
    front, which tied them all, so the tournament and the survival cut chose
    between them by spread in objective space. A population that started
    infeasible then had nothing pulling it toward feasibility: racer5 with
    seed 0 finished 0/32 after 40 generations (DECISIONS OPT-09).
    """
    feasible = [i for i, e in enumerate(evals) if e.feasible]
    infeasible = [i for i, e in enumerate(evals) if not e.feasible]

    fronts: list[list[int]] = []
    if feasible:
        sub = non_dominated_sort(objectives[feasible])
        fronts.extend([[feasible[j] for j in front] for front in sub])
    infeasible.sort(key=lambda i: evals[i].violation)
    for _, level in groupby(infeasible, key=lambda i: evals[i].violation):
        fronts.append(list(level))
    return fronts


def _tournament(rank: np.ndarray, crowding: np.ndarray, rng: np.random.Generator,
                n: int) -> np.ndarray:
    """Binary tournament: lower rank wins; ties broken by more elbow room."""
    a = rng.integers(0, len(rank), n)
    b = rng.integers(0, len(rank), n)
    better = np.where(
        rank[a] != rank[b],
        np.where(rank[a] < rank[b], a, b),
        np.where(crowding[a] > crowding[b], a, b),
    )
    return better


def _sbx(parents: np.ndarray, lower: np.ndarray, upper: np.ndarray,
         rng: np.random.Generator, eta: float = 15.0, prob: float = 0.9) -> np.ndarray:
    """Simulated binary crossover: children near their parents, occasionally far."""
    children = parents.copy()
    for i in range(0, len(parents) - 1, 2):
        if rng.random() > prob:
            continue
        p1, p2 = parents[i], parents[i + 1]
        u = rng.random(p1.shape)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                        (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        children[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        children[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(children, lower, upper)


def _mutate(x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
            rng: np.random.Generator, eta: float = 20.0,
            prob: float | None = None) -> np.ndarray:
    """Polynomial mutation, one variable at a time on average."""
    prob = prob if prob is not None else 1.0 / max(x.shape[1], 1)
    span = upper - lower
    u = rng.random(x.shape)
    delta = np.where(u < 0.5,
                     (2 * u) ** (1 / (eta + 1)) - 1.0,
                     1.0 - (2 * (1 - u)) ** (1 / (eta + 1)))
    hit = rng.random(x.shape) < prob
    return np.clip(x + hit * delta * span, lower, upper)


def minimise(problem: DesignProblem, population: int = 48, generations: int = 30,
             seed: int = 0, verbose: bool = False) -> Result:
    """Search the design space. Returns the whole final population and its front."""
    if population % 2:
        population += 1  # crossover pairs them up
    rng = np.random.default_rng(seed)
    lower, upper = problem.bounds

    x = rng.uniform(lower, upper, size=(population, problem.n_var))
    # Seed one member at the spec's own layout: the author's design is a real
    # candidate and often a good one, and starting entirely at random throws
    # that information away.
    x[0] = np.clip((lower + upper) / 2, lower, upper)

    evals = [problem.evaluate(row) for row in x]
    objectives = np.array([e.objectives for e in evals])
    history: list[dict] = []

    for gen in range(generations):
        fronts = _constrained_ranks(evals, objectives)
        rank = np.empty(len(evals))
        crowd = np.zeros(len(evals))
        for r, front in enumerate(fronts):
            rank[front] = r
            if len(front) > 2:
                crowd[front] = crowding_distance(objectives[front])
            else:
                crowd[front] = np.inf

        picks = _tournament(rank, crowd, rng, population)
        children = _sbx(x[picks], lower, upper, rng)
        children = _mutate(children, lower, upper, rng)

        child_evals = [problem.evaluate(row) for row in children]
        child_obj = np.array([e.objectives for e in child_evals])

        pooled_x = np.vstack([x, children])
        pooled_evals = evals + child_evals
        pooled_obj = np.vstack([objectives, child_obj])

        # Elitist survival: refill from the best fronts, breaking the last
        # front by crowding so the retained set stays spread out rather than
        # clustering on one attractive corner.
        chosen: list[int] = []
        for front in _constrained_ranks(pooled_evals, pooled_obj):
            if len(chosen) + len(front) <= population:
                chosen.extend(front)
                continue
            room = population - len(chosen)
            if room > 0:
                d = (crowding_distance(pooled_obj[front]) if len(front) > 2
                     else np.full(len(front), np.inf))
                chosen.extend([front[i] for i in np.argsort(-d)[:room]])
            break

        x = pooled_x[chosen]
        evals = [pooled_evals[i] for i in chosen]
        objectives = pooled_obj[chosen]

        feasible = sum(1 for e in evals if e.feasible)
        best = objectives[[i for i, e in enumerate(evals) if e.feasible]].min(axis=0) \
            if feasible else None
        history.append({"generation": gen, "feasible": feasible,
                        "best": None if best is None else best.tolist()})
        if verbose:
            shown = "none feasible" if best is None else np.round(best, 4)
            print(f"  gen {gen:3d}  feasible {feasible:3d}/{population}  best {shown}",
                  flush=True)

    final = _constrained_ranks(evals, objectives)
    return Result(x, objectives, evals, final[0] if final else [], history)
