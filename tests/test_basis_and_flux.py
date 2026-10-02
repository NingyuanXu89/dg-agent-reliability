import numpy as np
import pytest

from burgers_dg.basis import gauss_legendre, legendre, legendre_derivative
from burgers_dg.solver import godunov_flux, llf_flux


def test_legendre_orthogonality():
    xi, w = gauss_legendre(10)
    P = legendre(xi, 6)
    gram = (P * w[:, None]).T @ P
    np.testing.assert_allclose(gram, np.diag(2.0 / (2 * np.arange(7) + 1)), atol=1e-14)


def test_legendre_matches_numpy_and_derivative():
    xi = np.linspace(-1, 1, 23)
    P, D = legendre(xi, 5), legendre_derivative(xi, 5)
    for k in range(6):
        e = np.zeros(k + 1)
        e[k] = 1.0
        np.testing.assert_allclose(P[:, k], np.polynomial.legendre.legval(xi, e), atol=1e-14)
        np.testing.assert_allclose(D[:, k], np.polynomial.legendre.legval(xi, np.polynomial.legendre.legder(e)),
                                   atol=1e-13)
    np.testing.assert_allclose(P[-1], 1.0)
    np.testing.assert_allclose(P[0], (-1.0) ** np.arange(6))


@pytest.mark.parametrize("flux", [godunov_flux, llf_flux])
def test_flux_consistency(flux):
    u = np.linspace(-3, 3, 41)
    np.testing.assert_allclose(flux(u, u), 0.5 * u ** 2, atol=1e-15)


def test_godunov_riemann_cases():
    uL = np.array([2.0, -3.0, -1.0, 1.0, 2.0, 1.0])
    uR = np.array([3.0, -2.0, 1.0, -1.0, -3.0, -0.5])
    # right-moving rarefaction, left-moving rarefaction, transonic rarefaction,
    # stationary shock, left-moving shock, right-moving shock
    expected = np.array([2.0, 2.0, 0.0, 0.5, 4.5, 0.5])
    np.testing.assert_allclose(godunov_flux(uL, uR), expected)


@pytest.mark.parametrize("flux", [godunov_flux, llf_flux])
def test_flux_is_monotone(flux):
    rng = np.random.default_rng(0)
    a, b = rng.uniform(-2, 2, (2, 2000))
    eps = 1e-6
    assert np.all(flux(a + eps, b) - flux(a, b) >= -1e-14)   # nondecreasing in u_left
    assert np.all(flux(a, b + eps) - flux(a, b) <= 1e-14)    # nonincreasing in u_right
