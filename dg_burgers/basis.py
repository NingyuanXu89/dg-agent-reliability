"""Modal Legendre basis on the reference cell [-1, 1] and Gauss quadrature."""

import numpy as np
from numpy.polynomial import legendre as L


def gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [-1, 1]; exact for degree <= 2n - 1."""
    return L.leggauss(n)


def legendre_values(p, xi):
    """Return V[k, j] = P_j(xi_k) for j = 0..p."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    V = np.empty((xi.size, p + 1))
    for j in range(p + 1):
        c = np.zeros(j + 1)
        c[j] = 1.0
        V[:, j] = L.legval(xi, c)
    return V


def legendre_derivatives(p, xi):
    """Return D[k, j] = P_j'(xi_k) for j = 0..p."""
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    D = np.zeros((xi.size, p + 1))
    for j in range(1, p + 1):
        c = np.zeros(j + 1)
        c[j] = 1.0
        D[:, j] = L.legval(xi, L.legder(c))
    return D


def mass_inverse_diag(p, h):
    """Inverse of the (diagonal) mass matrix: M_jj = h / (2j + 1)."""
    return (2.0 * np.arange(p + 1) + 1.0) / h
