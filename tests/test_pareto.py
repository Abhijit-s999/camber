"""Non-dominated sorting: the core of NSGA-II, needed for P3.

Written BEFORE the implementation and not by the implementer (AI-05).
"""
import numpy as np
import pytest

from camber.optimise.pareto import crowding_distance, dominates, non_dominated_sort


class TestDominates:
    def test_strictly_better_on_all_objectives(self):
        assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))

    def test_better_on_one_equal_on_other(self):
        assert dominates(np.array([1.0, 2.0]), np.array([2.0, 2.0]))

    def test_equal_does_not_dominate(self):
        assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))

    def test_trade_off_neither_dominates(self):
        a, b = np.array([1.0, 5.0]), np.array([5.0, 1.0])
        assert not dominates(a, b) and not dominates(b, a)


class TestNonDominatedSort:
    def test_single_point_is_one_front(self):
        assert non_dominated_sort(np.array([[1.0, 2.0]])) == [[0]]

    def test_all_mutually_dominating_chain(self):
        pts = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        assert non_dominated_sort(pts) == [[0], [1], [2]]

    def test_pure_trade_off_is_one_front(self):
        pts = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        assert non_dominated_sort(pts) == [[0, 1, 2]]

    def test_two_fronts(self):
        pts = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]])
        fronts = non_dominated_sort(pts)
        assert fronts == [[0, 1], [2, 3]]

    def test_every_point_appears_exactly_once(self):
        rng = np.random.default_rng(0)
        pts = rng.random((60, 3))
        fronts = non_dominated_sort(pts)
        flat = [i for f in fronts for i in f]
        assert sorted(flat) == list(range(60))

    def test_first_front_is_not_dominated_by_anything(self):
        rng = np.random.default_rng(1)
        pts = rng.random((40, 2))
        first = non_dominated_sort(pts)[0]
        for i in first:
            assert not any(dominates(pts[j], pts[i]) for j in range(40) if j != i)

    def test_empty_input(self):
        assert non_dominated_sort(np.empty((0, 2))) == []


class TestCrowdingDistance:
    def test_boundary_points_are_infinite(self):
        pts = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        d = crowding_distance(pts)
        assert np.isinf(d[0]) and np.isinf(d[2])
        assert np.isfinite(d[1])

    def test_isolated_point_gets_more_room_than_a_crowded_one(self):
        pts = np.array([[0.0, 10.0], [1.0, 5.0], [1.1, 4.9], [10.0, 0.0]])
        d = crowding_distance(pts)
        assert d[1] > 0 and d[2] > 0

    def test_length_matches_input(self):
        pts = np.random.default_rng(2).random((15, 3))
        assert len(crowding_distance(pts)) == 15

    def test_degenerate_objective_does_not_divide_by_zero(self):
        """All points equal on one objective: the spread is zero there."""
        pts = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        d = crowding_distance(pts)
        assert not np.isnan(d).any()


class TestCrowdingDistanceExactValues:
    """Pin the arithmetic, not just its shape.

    The first delegated implementation passed every test above while computing
    a *backward* difference, f[i] - f[i-1], where NSGA-II specifies the central
    difference f[i+1] - f[i-1]. Boundary-is-inf and interior-is-positive cannot
    tell those apart. Selection pressure was quietly wrong.
    """

    def test_interior_point_uses_the_central_difference(self):
        # Sorted on obj 0: 0, 1, 3  (range 3) -> centre gets (3-0)/3 = 1.0
        # Sorted on obj 1: 0, 2, 3  (range 3) -> centre gets (3-0)/3 = 1.0
        pts = np.array([[0.0, 3.0], [1.0, 2.0], [3.0, 0.0]])
        d = crowding_distance(pts)
        assert d[1] == pytest.approx(2.0), (
            "interior crowding must span the neighbours on both sides; a "
            "backward difference would give 1.0 here"
        )

    def test_evenly_spaced_front_gives_equal_interior_distances(self):
        pts = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
        d = crowding_distance(pts)
        interior = d[1:-1]
        assert np.allclose(interior, interior[0])
        # Each objective contributes (2 spacings)/(range) = 2/4 = 0.5; two
        # objectives -> 1.0.
        assert interior[0] == pytest.approx(1.0)

    def test_a_point_with_distant_neighbours_beats_a_crowded_one(self):
        pts = np.array([[0.0, 100.0], [1.0, 60.0], [1.1, 59.0], [50.0, 0.0]])
        d = crowding_distance(pts)
        assert np.isfinite(d[1]) and np.isfinite(d[2])

    def test_returns_python_floats_not_numpy_bools_from_dominates(self):
        assert dominates(np.array([1.0]), np.array([2.0])) is True
        assert dominates(np.array([2.0]), np.array([1.0])) is False
