import numpy as np
import pytest

from dgburgers.exact import (exact_energy, exact_slope, exact_solution, initial_condition,
                             shock_position)

x = np.linspace(0, 2 * np.pi, 2001, endpoint=False)


def test_initial_data():
    np.testing.assert_allclose(exact_solution(x, 0.0), initial_condition(x), atol=1e-15)


@pytest.mark.parametrize("t", [0.2, 0.6, 0.95])
@pytest.mark.parametrize("c,A", [(0.5, 1.0), (0.0, 1.0), (-1.0, 2.0)])
def test_smooth_solution_satisfies_characteristics(t, c, A):
    u = exact_solution(x, t, c, A)
    np.testing.assert_allclose(u, initial_condition(x - u * t, c, A), atol=1e-12)


def test_post_shock_characteristics_and_entropy_condition():
    t, c = 2.0, 0.5
    xs = shock_position(t, c)
    u = exact_solution(x, t, c)
    away = np.abs(np.mod(x - xs + np.pi, 2 * np.pi) - np.pi) > 1e-3
    np.testing.assert_allclose(u[away], initial_condition(x[away] - u[away] * t, c), atol=1e-12)
    uL, uR = exact_solution(xs - 1e-9, t, c), exact_solution(xs + 1e-9, t, c)
    assert uL > c > uR                               # Lax entropy condition, speed c
    assert abs(0.5 * (uL + uR) - c) < 1e-6           # Rankine-Hugoniot speed (uL + uR)/2 = c


def test_slope_matches_finite_difference():
    t, eps = 0.7, 1e-6
    fd = (exact_solution(x + eps, t) - exact_solution(x - eps, t)) / (2 * eps)
    np.testing.assert_allclose(exact_slope(x, t), fd, atol=1e-6)
    assert abs(exact_slope(np.pi + 0.5 * t, t) + 1 / (1 - t)) < 1e-9


def test_energy_conserved_before_and_dissipated_after_shock():
    e0 = exact_energy(0.0)
    assert abs(e0 - (2 * np.pi * 0.25 + np.pi) / 2) < 1e-12
    assert abs(exact_energy(0.8) - e0) < 1e-10
    e = [exact_energy(t) for t in (1.2, 1.6, 2.4)]
    assert e0 > e[0] > e[1] > e[2]
