import numpy as np
import pytest

from burgers_dg.basis import LegendreBasis, gauss_legendre, legendre


@pytest.mark.parametrize("p", [0, 1, 2, 3, 5])
def test_legendre_orthogonality_matches_diagonal_mass(p):
    r, w = gauss_legendre(p + 2)
    P, _ = legendre(p, r)
    M = (P * w) @ P.T
    assert np.allclose(M, np.diag(2.0 / (2 * np.arange(p + 1) + 1)), atol=1e-14)
    assert np.allclose(LegendreBasis(p).mass_diag * LegendreBasis(p).mass_inv, 1.0)


def test_legendre_values_and_derivatives_against_numpy():
    r = np.linspace(-1, 1, 11)
    P, dP = legendre(4, r)
    for n in range(5):
        coef = np.zeros(n + 1)
        coef[n] = 1.0
        assert np.allclose(P[n], np.polynomial.legendre.legval(r, coef), atol=1e-14)
        dcoef = np.polynomial.legendre.legder(coef) if n else [0.0]
        assert np.allclose(dP[n], np.polynomial.legendre.legval(r, dcoef), atol=1e-13)
    assert np.allclose(P[:, -1], 1.0)                                # P_n(1) = 1
    assert np.allclose(P[:, 0], (-1.0) ** np.arange(5))             # P_n(-1) = (-1)^n


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_gauss_quadrature_exact_to_degree_2n_minus_1(n):
    r, w = gauss_legendre(n)
    for d in range(2 * n):
        exact = 0.0 if d % 2 else 2.0 / (d + 1)
        assert abs(np.sum(w * r**d) - exact) < 1e-13
