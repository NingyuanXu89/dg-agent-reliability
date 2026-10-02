import numpy as np

from burgers_dg.exact import SineInitialData, characteristic_foot, exact_solution, hopf_lax_solution

DATA = SineInitialData()


def test_characteristic_foot_solves_implicit_equation():
    x = np.linspace(0, 1, 501)
    for frac in (0.1, 0.5, 0.9, 0.99):
        t = frac * DATA.shock_time
        xi = characteristic_foot(x, t, DATA)
        np.testing.assert_allclose(xi + t * DATA.u0(xi), x, atol=1e-13)


def test_hopf_lax_matches_characteristics_before_shock():
    x = np.linspace(0, 1, 301)
    t = 0.7 * DATA.shock_time
    np.testing.assert_allclose(hopf_lax_solution(x, t, DATA), DATA.u0(characteristic_foot(x, t, DATA)), atol=1e-12)


def test_shock_time_and_max_gradient():
    assert np.isclose(DATA.shock_time, 1 / (2 * np.pi))
    t = 0.8 * DATA.shock_time
    x = np.linspace(0, 1, 200001)
    u = exact_solution(x, t, DATA)
    numerical = np.abs(np.diff(u) / np.diff(x)).max()
    assert abs(numerical - DATA.max_gradient(t)) / DATA.max_gradient(t) < 1e-3


def test_post_shock_entropy_solution():
    t = 1.5 * DATA.shock_time
    x = (np.arange(40000) + 0.5) / 40000
    u = exact_solution(x, t, DATA)
    # Mass is conserved through the shock.
    assert abs(u.mean() - DATA.mean) < 1e-4
    # One jump, of the admissible (decreasing) sign; it travels at the mean speed 0.5.
    du = np.diff(u)
    j = np.argmin(du)
    assert du[j] < -1.0
    assert np.sort(du)[-1] < 1e-3
    assert abs(x[j] - (0.5 + DATA.mean * t)) < 1e-3
