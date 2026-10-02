"""Legendre polynomials and Gauss-Legendre quadrature on the reference cell [-1, 1]."""

import numpy as np


def legendre(n_modes, x):
    """Evaluate P_0..P_{n_modes-1} and their derivatives at points ``x``.

    Uses the three-term recurrence
        (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1},
    and P'_{l+1} = P'_{l-1} + (2l+1) P_l.

    Returns ``(P, dP)``, each of shape ``(n_modes,) + x.shape``.
    """
    x = np.asarray(x, dtype=float)
    P = np.zeros((n_modes,) + x.shape)
    dP = np.zeros_like(P)
    P[0] = 1.0
    if n_modes > 1:
        P[1] = x
        dP[1] = 1.0
    for l in range(1, n_modes - 1):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
        dP[l + 1] = dP[l - 1] + (2 * l + 1) * P[l]
    return P, dP


def gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [-1, 1] via the Golub-Welsch algorithm.

    The rule with ``n`` points integrates polynomials of degree <= 2n-1 exactly.
    """
    if n < 1:
        raise ValueError("need at least one quadrature point")
    k = np.arange(1, n)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    jacobi = np.diag(beta, 1) + np.diag(beta, -1)
    nodes, vecs = np.linalg.eigh(jacobi)
    weights = 2.0 * vecs[0] ** 2
    return nodes, weights
