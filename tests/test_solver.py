import numpy as np
import pytest

from dgburgers import DGBurgers, exact_solution, initial_condition, shock_position
from dgburgers.diagnostics import (energy, error_norms, extrema, mass, observed_orders,
                                   total_variation)
from dgburgers.solver import burgers_flux, godunov_flux, llf_flux
from dgburgers.timestepping import INTEGRATORS, ORDER


# ------------------------------------------------------------------ fluxes
@pytest.mark.parametrize("F", [godunov_flux, llf_flux])
def test_flux_consistency(F):
    u = np.linspace(-3, 3, 61)
    np.testing.assert_allclose(F(u, u), burgers_flux(u), atol=1e-15)


@pytest.mark.parametrize("F", [godunov_flux, llf_flux])
def test_flux_monotone(F):
    a, b = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-2, 2, 81), indexing="ij")
    Fab = F(a, b)
    assert np.all(np.diff(Fab, axis=0) >= -1e-14)   # non-decreasing in the left state
    assert np.all(np.diff(Fab, axis=1) <= 1e-14)    # non-increasing in the right state


def test_godunov_riemann_cases():
    assert godunov_flux(1.0, 2.0) == 0.5          # rarefaction moving right: f(uL)
    assert godunov_flux(-2.0, -1.0) == 0.5        # rarefaction moving left: f(uR)
    assert godunov_flux(-1.0, 1.0) == 0.0         # transonic rarefaction: f(0)
    assert godunov_flux(2.0, -1.0) == 2.0         # shock, speed 0.5 > 0: f(uL)
    assert godunov_flux(1.0, -2.0) == 2.0         # shock, speed -0.5 < 0: f(uR)


# ---------------------------------------------------------- time steppers
@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_time_integrator_order(name):
    # y' = -y^2, y(0) = 1  =>  y(1) = 1/2
    L = lambda y: -y * y
    errs = []
    for n in (20, 40, 80):
        y = np.array([1.0])
        for _ in range(n):
            y = INTEGRATORS[name](L, lambda v: v, y, 1.0 / n)
        errs.append(abs(y[0] - 0.5))
    rates = observed_orders([1 / 20, 1 / 40, 1 / 80], errs)
    assert rates[-1] == pytest.approx(ORDER[name], abs=0.2)


# --------------------------------------------------------- discretisation
@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_projection_reproduces_polynomials(p):
    s = DGBurgers(7, p, domain=(0.0, 1.0))
    f = lambda x: sum((k + 1) * x ** k for k in range(p + 1))
    xi = np.linspace(-1, 1, 9)
    np.testing.assert_allclose(s.evaluate(s.project(f), xi), f(s.x_at(xi)), atol=1e-12)


@pytest.mark.parametrize("p", [0, 1, 2, 3])
@pytest.mark.parametrize("flux", ["godunov", "llf"])
@pytest.mark.parametrize("limiter", [None, "minmod"])
def test_constant_state_preserved(p, flux, limiter):
    s = DGBurgers(16, p, flux=flux, limiter=limiter)
    U0 = s.project(lambda x: np.full_like(x, 0.7))
    # Round-off in the quadrature of int P_l' is amplified by (2l+1)/h.
    np.testing.assert_allclose(s.rhs(U0), 0.0, atol=1e-12)
    U = s.solve(U0, 1.0)["U"]
    np.testing.assert_allclose(U[:, 0], 0.7, atol=1e-13)
    np.testing.assert_allclose(U[:, 1:], 0.0, atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("limiter", [None, "minmod"])
@pytest.mark.parametrize("flux", ["godunov", "llf"])
def test_mass_conserved_through_shock(p, limiter, flux):
    s = DGBurgers(40, p, flux=flux, limiter=limiter, tvb_M=1.0)
    U0 = s.project(initial_condition)
    m0 = mass(s, U0)
    assert m0 == pytest.approx(2 * np.pi * 0.5, rel=1e-13)
    U = s.solve(U0, 2.5)["U"]
    assert abs(mass(s, U) - m0) < 1e-12 * abs(m0)


@pytest.mark.parametrize("c", [3.0, -3.0])
def test_transport_direction_and_periodic_wrap(c):
    # All characteristic speeds have the sign of c; by t = 1.5 the wave has
    # moved |c| t = 4.5 (> 2 pi * 0.7) through the periodic boundary.
    A, t = 0.5, 1.5
    s = DGBurgers(80, 2)
    U = s.solve(s.project(lambda x: initial_condition(x, c, A)), t)["U"]
    err = error_norms(s, U, lambda x: exact_solution(x, t, c, A))
    assert err["Linf"] < 1e-3
    # Peak of u_h lies where the exact peak is, i.e. it moved in the direction of c.
    xi = np.linspace(-1, 1, 41)
    xs, us = s.x_at(xi).ravel(), s.evaluate(U, xi).ravel()
    x_peak = xs[np.argmax(us)]
    x_exact = np.linspace(0, 2 * np.pi, 20001)
    x_peak_exact = x_exact[np.argmax(exact_solution(x_exact, t, c, A))]
    d = abs(np.mod(x_peak - x_peak_exact + np.pi, 2 * np.pi) - np.pi)
    assert d < s.h
    # Wrong direction would place the peak 2|c|t (mod 2 pi) away from the exact one.
    assert abs(np.mod(2 * c * t + np.pi, 2 * np.pi) - np.pi) > 5 * s.h


@pytest.mark.parametrize("p", [1, 2, 3])
def test_smooth_convergence_order(p):
    t, errs, hs = 0.5, [], []
    for N in (40, 80):
        s = DGBurgers(N, p)
        U = s.solve(s.project(initial_condition), t, integrator="ssprk104")["U"]
        errs.append(error_norms(s, U, lambda x: exact_solution(x, t))["L2"])
        hs.append(s.h)
    assert observed_orders(hs, errs)[0] > p + 0.8


def test_energy_never_increases_through_shock():
    # Exact quadrature + monotone flux gives a cell entropy inequality for
    # u^2/2 (Jiang & Shu), so the energy is non-increasing even without a limiter.
    s = DGBurgers(48, 2)
    e = []
    s.solve(s.project(initial_condition), 2.5, callback=lambda t, U: e.append(energy(s, U)))
    e = np.array(e)
    assert np.all(np.diff(e) <= 1e-12 * e[0])
    assert e[-1] < 0.8 * e[0]


def test_post_shock_convergence_and_shock_location():
    t, xs = 2.0, shock_position(2.0)
    away, glob, hs = [], [], []
    for N in (40, 80, 160):
        s = DGBurgers(N, 2, limiter="minmod", tvb_M=1.0)
        U = s.solve(s.project(initial_condition), t)["U"]
        ex = lambda x: exact_solution(x, t)
        glob.append(error_norms(s, U, ex, discontinuities=(xs,))["L1"])
        away.append(error_norms(s, U, ex, exclude=(xs, 0.5))["L1"])
        hs.append(s.h)
    assert np.all(observed_orders(hs, away) > 2.5)          # high order away from the shock
    assert np.polyfit(np.log(hs), np.log(glob), 1)[0] > 0.8  # ~first order globally
    # The largest interface jump sits at the exact shock location.
    left, right = s.traces(U)
    j = np.argmax(np.abs(np.roll(left, -1) - right))
    assert abs(s.edges[j + 1] - xs) < 2 * s.h


# ----------------------------------------------------------------- limiter
def test_limiter_inactive_on_resolved_smooth_data():
    s = DGBurgers(80, 2, limiter="minmod", tvb_M=1.0)
    U0 = s.project(initial_condition)
    np.testing.assert_array_equal(s.limit(U0), U0)


def test_limiter_preserves_means_and_removes_overshoot():
    s = DGBurgers(64, 2, limiter="minmod", tvb_M=1.0)
    U0 = s.project(initial_condition)
    U = s.solve(U0, 2.0)["U"]
    np.testing.assert_array_equal(s.limit(U)[:, 0], U[:, 0])
    lo, hi = extrema(s, U)
    assert lo > -0.5 - 1e-2 and hi < 1.5 + 1e-2
    unlimited = DGBurgers(64, 2)
    lo_u, hi_u = extrema(unlimited, unlimited.solve(U0, 2.0)["U"])
    assert hi_u > 1.7                                        # Gibbs overshoot without limiter


def test_tvd_limiter_keeps_means_total_variation_non_increasing():
    s = DGBurgers(64, 1, limiter="minmod", tvb_M=0.0)
    tv = []
    s.solve(s.project(initial_condition), 2.0, cfl=0.3,
            callback=lambda t, U: tv.append(total_variation(U)))
    assert np.all(np.diff(tv) <= 1e-12)


def test_blow_up_is_reported():
    s = DGBurgers(32, 2)
    with np.errstate(all="ignore"), pytest.raises(FloatingPointError):
        s.solve(s.project(initial_condition), 100.0, dt=0.5)   # ~13x the stable step
