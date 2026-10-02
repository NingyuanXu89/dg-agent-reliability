import numpy as np
import pytest

from dg_burgers import DGBurgers, diagnostics as d, exact
from dg_burgers.basis import gauss_legendre


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_gauss_quadrature_exact_to_degree_2n_minus_1(n):
    xi, w = gauss_legendre(n)
    for k in range(2 * n):
        expected = 0.0 if k % 2 else 2.0 / (k + 1)
        assert np.isclose(np.sum(w * xi ** k), expected, atol=1e-14)


@pytest.mark.parametrize("p", [0, 1, 2, 3, 4])
def test_projection_reproduces_polynomials(p):
    s = DGBurgers(7, p)
    f = lambda x: sum((0.3 * x) ** k for k in range(p + 1))
    U = s.project(f)
    x, uh = s.evaluate(U, np.linspace(-1, 1, 9))
    assert np.max(np.abs(uh - f(x))) < 1e-12


@pytest.mark.parametrize("flux", ["godunov", "rusanov"])
@pytest.mark.parametrize("limiter", [False, True])
@pytest.mark.parametrize("c", [-0.7, 0.0, 1.3])
def test_constant_state_preserved(flux, limiter, c):
    s = DGBurgers(16, 2, flux=flux, limiter=limiter)
    U0 = s.project(lambda x: np.full_like(x, c))
    U, n = s.run(U0, 0.5)
    assert n > 0 or c == 0.0
    assert np.max(np.abs(U - U0)) < 1e-13


@pytest.mark.parametrize("limiter", [False, True])
def test_mass_conserved_through_shock(limiter):
    s = DGBurgers(64, 2, limiter=limiter, tvb_M=5.0)
    U0 = s.project(exact.u0)
    U, _ = s.run(U0, 1.5 * exact.breaking_time())
    assert abs(d.total_mass(s, U) - d.total_mass(s, U0)) < 1e-12
    assert np.isclose(d.total_mass(s, U0), exact.U_MEAN * exact.L_DOMAIN, atol=1e-12)


def test_energy_conserved_before_and_dissipated_after_shock():
    x = np.linspace(0, 2 * np.pi, 400001)[:-1]
    E_exact = np.mean(exact.entropy_solution(x, 1.5) ** 2 / 2) * 2 * np.pi
    deficits = []
    for N in [64, 128, 256]:
        s = DGBurgers(N, 2, limiter=True, tvb_M=5.0)
        U0 = s.project(exact.u0)
        E0 = d.total_energy(s, U0)
        U1, _ = s.run(U0, 0.5)
        U2, _ = s.run(U1, 1.5, t_start=0.5)
        assert abs(d.total_energy(s, U1) - E0) / E0 < 1e-5    # smooth: conserved
        assert d.total_energy(s, U2) < E0 - 0.1               # shock: dissipated
        deficits.append(E_exact - d.total_energy(s, U2))
    # Numerical shock dissipation exceeds the exact one by O(h).
    assert all(dd > 0 for dd in deficits)
    assert deficits[1] / deficits[0] < 0.6 and deficits[2] / deficits[1] < 0.6


def test_exact_solution_satisfies_characteristics():
    x = np.linspace(0, 2 * np.pi, 501)
    for t in [0.0, 0.2, 0.7, 0.99]:
        xi = exact.foot_of_characteristic(x, t)
        assert np.max(np.abs(xi + t * exact.u0(xi) - x)) < 1e-12
        assert np.max(np.abs(exact.entropy_solution(x, t) - exact.exact_solution(x, t))) < 1e-10
    assert exact.breaking_time() == 1.0
    assert np.isclose(exact.shock_formation_point(), np.pi + 0.5)
    with pytest.raises(ValueError):
        exact.foot_of_characteristic(x, 1.0)


def test_entropy_solution_has_correct_shock():
    t = 1.5
    xs = exact.shock_position(t)
    uL = exact.entropy_solution(xs - 1e-9, t)
    uR = exact.entropy_solution(xs + 1e-9, t)
    assert uL > uR                                     # Lax entropy condition
    assert np.isclose(0.5 * (uL + uR), exact.U_MEAN)   # Rankine-Hugoniot speed = shock speed


def test_transport_direction_and_speed():
    """The crest (u = 1.5 at x = pi/2) travels right with speed 1.5."""
    s = DGBurgers(200, 3)
    t = 0.2
    U, _ = s.run(s.project(exact.u0), t)
    x, u = d.dense_profile(s, U, npts=40)
    crest = x[np.nanargmax(u)]
    assert abs(crest - (np.pi / 2 + 1.5 * t)) < 2 * s.h


def test_periodic_boundary_matches_shifted_problem():
    """Shifting the data by whole cells must shift the solution by the same cells."""
    N, shift = 40, 13
    s = DGBurgers(N, 2)
    U0 = s.project(exact.u0)
    Ua, _ = s.run(U0, 0.6)
    Ub, _ = s.run(np.roll(U0, shift, axis=0), 0.6)
    assert np.max(np.abs(np.roll(Ua, shift, axis=0) - Ub)) < 1e-13


def test_periodic_wraparound_of_moving_shock():
    """By t = 1 + 2*pi/0.5 the shock has wrapped once; the profile must repeat."""
    s = DGBurgers(80, 1, limiter=True, tvb_M=5.0)
    U, _ = s.run(s.project(exact.u0), 2.0)
    ref = lambda x: exact.entropy_solution(x, 2.0)
    assert d.error_norms(s, U, ref)["L1"] < 0.05


@pytest.mark.parametrize("p", [1, 2])
def test_convergence_order_smooth(p):
    t = 0.5
    hs, errs = [], []
    for N in [20, 40, 80]:
        s = DGBurgers(N, p)
        U, _ = s.run(s.project(exact.u0), t)
        errs.append(d.error_norms(s, U, lambda x: exact.exact_solution(x, t))["L2"])
        hs.append(s.h)
    assert d.observed_orders(hs, errs)[-1] > p + 0.8


def test_godunov_and_rusanov_agree_on_smooth_solution():
    t = 0.5
    sols = []
    for flux in ["godunov", "rusanov"]:
        s = DGBurgers(80, 2, flux=flux)
        sols.append(s.run(s.project(exact.u0), t)[0])
    assert np.max(np.abs(sols[0] - sols[1])) < 1e-3


def test_limiter_keeps_linear_data_and_cell_averages():
    s = DGBurgers(20, 2, limiter=True, tvb_M=0.0)
    U = s.project(lambda x: 0.1 * x)
    # A global linear function (away from the periodic wrap) is not limited.
    V = s.limit(U)
    interior = slice(1, -1)
    assert np.allclose(V[interior], U[interior], atol=1e-14)
    # Step data: limiter preserves averages and removes over/undershoots.
    step = s.project(lambda x: np.where(x < np.pi, 1.0, 0.0))
    W = s.limit(step)
    assert np.allclose(W[:, 0], step[:, 0])
    _, u = s.evaluate(W, np.linspace(-1, 1, 11))
    assert u.max() <= 1.0 + 1e-12 and u.min() >= -1e-12


def test_limiter_prevents_post_shock_oscillations():
    t = 1.3
    s = DGBurgers(100, 2, limiter=True, tvb_M=5.0)
    U, _ = s.run(s.project(exact.u0), t)
    _, u = d.dense_profile(s, U)
    assert np.nanmax(u) < 1.5 + 0.02 and np.nanmin(u) > -0.5 - 0.02
