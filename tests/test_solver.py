import numpy as np
import pytest

from burgers_dg import BurgersDG, SineInitialData, error_norms, exact_solution, observed_orders
from burgers_dg.diagnostics import energy, total_mass, total_variation_of_means

DATA = SineInitialData()


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_projection_reproduces_polynomials(degree):
    solver = BurgersDG(7, degree)
    poly = lambda x: sum(c * x ** k for k, c in enumerate([0.3, -1.0, 2.0, -1.5][: degree + 1]))
    C = solver.project(poly)
    xi = np.linspace(-1, 1, 5)
    np.testing.assert_allclose(solver.evaluate(C, xi), poly(solver.x_physical(xi)), atol=1e-13)


@pytest.mark.parametrize("limiter", [None, "tvb"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_constant_state_preserved(degree, limiter):
    solver = BurgersDG(16, degree, limiter=limiter)
    C0 = solver.project(lambda x: np.full_like(x, -0.7))
    C, steps = solver.advance(C0, 0.0, 2.0)
    assert steps > 25
    np.testing.assert_allclose(C, C0, atol=1e-14)


@pytest.mark.parametrize("flux", ["godunov", "llf"])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_semidiscrete_energy_stability(degree, flux):
    """With exact quadrature and an E-flux, d/dt int u^2/2 <= 0 for every u_h (Jiang-Shu)."""
    rng = np.random.default_rng(degree)
    solver = BurgersDG(12, degree, flux=flux)
    mass_weights = solver.h / (2 * solver.modes + 1)
    for _ in range(50):
        C = rng.normal(size=(12, degree + 1))
        assert np.sum(mass_weights * C * solver.rhs(C)) <= 1e-12


@pytest.mark.parametrize("limiter", [None, "tvb"])
def test_mass_conserved_through_shock(limiter):
    solver = BurgersDG(40, 2, limiter=limiter, tvb_M=26.3)
    C0 = solver.project(DATA.u0)
    t_end = 0.8 * DATA.shock_time if limiter is None else 1.5 * DATA.shock_time
    C, _ = solver.advance(C0, 0.0, t_end)
    assert abs(total_mass(solver, C) - total_mass(solver, C0)) < 1e-13
    assert energy(solver, C) <= energy(solver, C0) + 1e-14


@pytest.mark.parametrize("mean", [1.0, -1.0])
def test_transport_direction(mean):
    """A small perturbation of a constant state u = mean is carried at speed ~mean."""
    data = SineInitialData(mean=mean, amplitude=0.01)
    solver = BurgersDG(64, 2)
    t = 0.2
    C, _ = solver.advance(solver.project(data.u0), 0.0, t)
    xi = np.linspace(-1, 1, 21)
    x, u = solver.x_physical(xi).ravel(), solver.evaluate(C, xi).ravel()
    # Peak of 0.01 sin(2 pi x) starts at x = 0.25 and moves to 0.25 + mean * t (mod 1).
    expected_peak = (0.25 + mean * t) % 1.0
    assert abs(x[np.argmax(u)] - expected_peak) < 0.02
    assert error_norms(solver, C, lambda y: exact_solution(y, t, data))["Linf"] < 1e-6


def test_periodic_wrap():
    """Data translated by exactly one period returns to (the exact solution at) its start."""
    data = SineInitialData(mean=2.0, amplitude=0.02)
    solver = BurgersDG(40, 3)
    t = 0.5                                   # mean * t = 1 period
    C, _ = solver.advance(solver.project(data.u0), 0.0, t)
    assert error_norms(solver, C, lambda y: exact_solution(y, t, data))["L2"] < 1e-7
    # Wrapped bump is close to the initial data (nonlinear steepening is O(amplitude^2 t)).
    assert error_norms(solver, C, data.u0)["Linf"] < 5e-3


@pytest.mark.parametrize("degree", [1, 2])
def test_convergence_rate(degree):
    t = 0.5 * DATA.shock_time
    hs, errs = [], []
    for n in (20, 40, 80):
        solver = BurgersDG(n, degree)
        C, _ = solver.advance(solver.project(DATA.u0), 0.0, t)
        errs.append(error_norms(solver, C, lambda x: exact_solution(x, t, DATA))["L2"])
        hs.append(solver.h)
    assert observed_orders(errs, hs)[-1] > degree + 0.8


def test_minmod_limiter_is_tvdm_through_shock():
    solver = BurgersDG(50, 1, limiter="tvb", tvb_M=0.0, cfl=0.5)
    C = solver.apply_limiter(solver.project(DATA.u0))
    tv = [total_variation_of_means(C)]
    t, dt = 0.0, 0.002
    while t < 2 * DATA.shock_time:
        C = solver.ssp_rk3_step(C, dt)
        t += dt
        tv.append(total_variation_of_means(C))
    assert np.all(np.diff(tv) <= 1e-12)
    # Post-shock solution stays within the initial range (no Gibbs overshoot in the means).
    assert C[:, 0].max() <= DATA.u_max + 1e-12 and C[:, 0].min() >= DATA.u_min - 1e-12


def test_limiter_preserves_cell_averages_and_ignores_smooth_interior():
    solver = BurgersDG(40, 2, limiter="tvb", tvb_M=0.0)
    C = solver.project(lambda x: 0.2 + 0.5 * x)    # linear except for the periodic jump at x = 0
    L = solver.apply_limiter(C)
    np.testing.assert_array_equal(L[:, 0], C[:, 0])
    assert set(np.flatnonzero(solver.last_limited)) <= {0, solver.n_cells - 1}
    np.testing.assert_allclose(L[1:-1], C[1:-1], atol=1e-15)
