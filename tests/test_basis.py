import numpy as np
import pytest

from dgburgers.basis import gauss_legendre, legendre


@pytest.mark.parametrize("n", range(1, 12))
def test_gauss_matches_numpy(n):
    x, w = gauss_legendre(n)
    xr, wr = np.polynomial.legendre.leggauss(n)
    np.testing.assert_allclose(x, xr, atol=1e-14)
    np.testing.assert_allclose(w, wr, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_gauss_exact_up_to_degree_2n_minus_1(n):
    x, w = gauss_legendre(n)
    for k in range(2 * n):
        exact = (1 - (-1) ** (k + 1)) / (k + 1)
        assert abs(w @ x ** k - exact) < 1e-13


def test_legendre_values_and_derivatives():
    x = np.linspace(-1, 1, 37)
    P, dP = legendre(7, x)
    for l in range(7):
        c = np.zeros(l + 1)
        c[l] = 1.0
        np.testing.assert_allclose(P[l], np.polynomial.legendre.legval(x, c), atol=1e-13)
        np.testing.assert_allclose(dP[l], np.polynomial.legendre.legval(x, np.polynomial.legendre.legder(c)),
                                   atol=1e-12)
    np.testing.assert_allclose(P[:, -1], 1.0)                       # P_l(1) = 1
    np.testing.assert_allclose(P[:, 0], (-1.0) ** np.arange(7))     # P_l(-1) = (-1)^l


def test_legendre_orthogonality():
    x, w = gauss_legendre(10)
    P, _ = legendre(8, x)
    M = (P * w) @ P.T
    np.testing.assert_allclose(M, np.diag(2.0 / (2 * np.arange(8) + 1)), atol=1e-14)
