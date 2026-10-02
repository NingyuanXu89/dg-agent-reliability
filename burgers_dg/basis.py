"""Legendre modal basis and Gauss-Legendre quadrature on the reference cell [-1, 1]."""

import numpy as np


def gauss_legendre(n):
    """Return the n-point Gauss-Legendre nodes and weights on [-1, 1].

    The rule integrates polynomials of degree <= 2n - 1 exactly.
    """
    return np.polynomial.legendre.leggauss(n)


def legendre(xi, degree):
    """Evaluate P_0, ..., P_degree at xi.

    Returns an array of shape ``xi.shape + (degree + 1,)`` built with the
    three-term (Bonnet) recurrence (k + 1) P_{k+1} = (2k + 1) xi P_k - k P_{k-1}.
    """
    xi = np.asarray(xi, dtype=float)
    P = np.empty(xi.shape + (degree + 1,))
    P[..., 0] = 1.0
    if degree >= 1:
        P[..., 1] = xi
    for k in range(1, degree):
        P[..., k + 1] = ((2 * k + 1) * xi * P[..., k] - k * P[..., k - 1]) / (k + 1)
    return P


def legendre_derivative(xi, degree):
    """Evaluate dP_k/dxi for k = 0, ..., degree at xi.

    Uses P'_{k+1} = P'_{k-1} + (2k + 1) P_k.
    """
    P = legendre(xi, degree)
    D = np.zeros_like(P)
    if degree >= 1:
        D[..., 1] = 1.0
    for k in range(1, degree):
        D[..., k + 1] = D[..., k - 1] + (2 * k + 1) * P[..., k]
    return D
