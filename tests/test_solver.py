import numpy as np
import pytest

from burgers_dg import DGBurgers, TVBLimiter, exact_solution

U0 = lambda x: 0.5 + np.sin(x)


@pytest.mark.parametrize("p", [0, 1, 2, 3])
def test_constant_state_preserved(p):
    s = DGBurgers(16, p)
    c = s.project(lambda x: 0.7 + 0 * x)
    c1, _ = s.run(c, 1.0)
    assert np.max(np.abs(c1 - c)) < 1e-13


@pytest.mark.parametrize("limited", [False, True])
def test_mass_conservation(limited):
    s = DGBurgers(40, 2, limiter=TVBLimiter() if limited else None)
    c = s.project(U0)
    m0 = s.mass(c)
    assert np.isclose(m0, np.pi, atol=1e-12)
    c1, _ = s.run(c, 1.3 if limited else 0.9)
    assert abs(s.mass(c1) - m0) < 1e-13


def test_energy_matches_exact_integral():
    # (1/2) int_0^{2 pi} (0.5 + sin x)^2 dx = 3 pi / 4
    s = DGBurgers(32, 3)
    assert np.isclose(s.energy(s.project(U0)), 0.75 * np.pi, rtol=1e-10)


def test_energy_conserved_while_smooth_and_dissipated_after_shock():
    s = DGBurgers(64, 2, limiter=TVBLimiter())
    c0 = s.project(U0)
    _, snaps = s.run(c0, 1.3, output_times=[0.5, 1.3])
    E0 = s.energy(c0)
    assert abs(s.energy(snaps[0.5]) - E0) / E0 < 1e-4
    assert s.energy(snaps[1.3]) < 0.98 * E0


def test_transport_direction():
    # small-amplitude wave: the crest moves right at speed u_max = 1.1
    s = DGBurgers(64, 2)
    u0 = lambda x: 1.0 + 0.1 * np.sin(x)
    c, _ = s.run(s.project(u0), 0.5)
    x, u = s.sample(c, 32)
    ok = ~np.isnan(u)
    crest = x[ok][np.argmax(u[ok])]
    assert abs(crest - (np.pi / 2 + 1.1 * 0.5)) < 0.03


def test_periodic_wrap():
    # crest travels pi/2 + 1.1*5 = 7.07 > 2 pi, so it must reappear near 0.787
    s = DGBurgers(32, 2)
    u0 = lambda x: 1.0 + 0.1 * np.sin(x)
    c, _ = s.run(s.project(u0), 5.0)
    exact = lambda x: exact_solution(x, 5.0, c=1.0, a=0.1)
    L1, L2, Linf = s.errors(c, exact)
    assert Linf < 1e-3
    x, u = s.sample(c, 32)
    ok = ~np.isnan(u)
    assert abs(x[ok][np.argmax(u[ok])] - (np.pi / 2 + 5.5 - 2 * np.pi)) < 0.03


@pytest.mark.parametrize("p", [1, 2])
def test_smooth_convergence_order(p):
    T = 0.4
    err = []
    for K in (16, 32):
        s = DGBurgers(K, p)
        c, _ = s.run(s.project(U0), T)
        err.append(s.errors(c, lambda x: exact_solution(x, T))[1])
    assert np.log2(err[0] / err[1]) > p + 0.5


def test_output_times_are_hit_exactly():
    s = DGBurgers(8, 1)
    seen = []
    _, snaps = s.run(s.project(U0), 0.3, output_times=[0.0, 0.1, 0.25], on_step=lambda t, c: seen.append(t))
    assert sorted(snaps) == [0.0, 0.1, 0.25]
    assert np.isclose(seen[-1], 0.3)
    assert any(np.isclose(t, 0.1) for t in seen) and any(np.isclose(t, 0.25) for t in seen)


def test_limiter_leaves_smooth_data_and_linear_ramp_alone():
    lim = TVBLimiter(M=1.0)
    s = DGBurgers(64, 2)
    c = s.project(U0)
    c2, troubled = lim(c, s.h)
    assert not troubled.any() and np.array_equal(c, c2)
    # a linear ramp (non-periodic) is untouched away from the wrap-around cells
    ramp = DGBurgers(16, 2, domain=(0.0, 1.0)).project(lambda x: 3 * x)
    c3, troubled = TVBLimiter(M=0.0)(ramp, 1 / 16)
    assert not troubled[1:-1].any()
    assert np.allclose(c3[1:-1], ramp[1:-1])


def test_limiter_removes_new_extrema_at_a_step():
    s = DGBurgers(32, 3)
    # jump at x = 3.3 lies inside a cell, so the projection overshoots there
    c = s.project(lambda x: np.where(x < 3.3, 1.0, 0.0))
    c2, troubled = TVBLimiter(M=0.0)(c, s.h)
    assert troubled.any()
    assert np.allclose(c2[:, 0], c[:, 0])                     # means untouched
    edges = np.stack([c2 @ np.ones(4), c2 @ (-1.0) ** np.arange(4)])
    assert edges.max() <= 1.0 + 1e-14 and edges.min() >= -1e-14


def test_limited_shock_is_nonoscillatory_and_well_located():
    s = DGBurgers(128, 2, limiter=TVBLimiter())
    c, _ = s.run(s.project(U0), 1.3)
    x, u = s.sample(c, 16)
    u = u[~np.isnan(u)]
    assert u.max() < 1.5 + 1e-4 and u.min() > -0.5 - 1e-4
    k = np.argmin(np.diff(c[:, 0]))
    assert abs(s.edges[k + 1] - (np.pi + 0.65)) < 1.5 * s.h
