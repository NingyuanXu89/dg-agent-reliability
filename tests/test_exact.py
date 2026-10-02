import numpy as np
import pytest
from scipy.optimize import brentq

from burgers_dg.exact import (characteristic_foot, exact_gradient,
                              exact_max_abs_gradient, exact_solution,
                              shock_position, shock_time)

X = np.linspace(0.0, 2 * np.pi, 201)


def test_initial_condition_recovered():
    assert np.allclose(exact_solution(X, 0.0), 0.5 + np.sin(X), atol=1e-14)


@pytest.mark.parametrize("t", [0.2, 0.6, 0.95])
def test_characteristic_equation_and_independent_root_finder(t):
    eta = characteristic_foot(X, t)
    y = np.mod(X - 0.5 * t, 2 * np.pi)
    assert np.allclose(eta + t * np.sin(eta), y, atol=1e-13)
    for x in X[::25]:
        xi = brentq(lambda z: z + t * (0.5 + np.sin(z)) - x, x - 1.5 * t - 1e-9, x + 0.5 * t + 1e-9)
        assert abs(exact_solution(x, t) - (0.5 + np.sin(xi))) < 1e-10


def test_shock_time_and_peak_gradient():
    assert shock_time(1.0) == 1.0
    t = 0.7
    xs = shock_position(t, 0.5)       # the steepest point travels with the shock-to-be
    assert np.isclose(exact_gradient(xs, t), -exact_max_abs_gradient(t))


def test_post_shock_solution_is_entropy_admissible_and_satisfies_rh():
    t = 1.4
    xs = shock_position(t, 0.5)
    uL = exact_solution(xs - 1e-10, t)
    uR = exact_solution(xs + 1e-10, t)
    assert uL > uR                                   # Lax entropy condition
    assert np.isclose(0.5 * (uL + uR), 0.5)          # RH speed = shock speed c
    # away from the shock the solution is a root on the monotone branch
    x = np.array([xs - 1.0, xs + 1.0])
    eta = characteristic_foot(x, t)
    assert np.all(1.0 + t * np.cos(eta) > 0)


def test_mass_conserved_by_exact_solution_after_shock():
    from burgers_dg.basis import gauss_legendre
    r, w = gauss_legendre(20)
    edges = np.linspace(0, 2 * np.pi, 401)
    # cells straddling the shock are tiny (h ~ 0.016) so the quadrature error is O(h^2)
    xm, hm = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    xq = xm[:, None] + 0.5 * hm[:, None] * r
    mass = np.sum(0.5 * hm[:, None] * w * exact_solution(xq, 1.5))
    assert abs(mass - np.pi) < 1e-3
