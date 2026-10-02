"""Modal Legendre basis and Gauss-Legendre quadrature on the reference cell [-1, 1]."""

import numpy as np


def legendre(p, r):
    """Evaluate Legendre polynomials P_0..P_p and their derivatives at points r.

    Uses the three-term recurrence
        (n+1) P_{n+1} = (2n+1) r P_n - n P_{n-1},
    and the derivative identity P'_{n+1} = P'_{n-1} + (2n+1) P_n.

    Returns arrays of shape (p+1, *r.shape).
    """
    r = np.asarray(r, dtype=float)
    P = np.zeros((p + 1,) + r.shape)
    dP = np.zeros_like(P)
    P[0] = 1.0
    if p >= 1:
        P[1] = r
        dP[1] = 1.0
    for n in range(1, p):
        P[n + 1] = ((2 * n + 1) * r * P[n] - n * P[n - 1]) / (n + 1)
        dP[n + 1] = dP[n - 1] + (2 * n + 1) * P[n]
    return P, dP


def gauss_legendre(n):
    """n-point Gauss-Legendre nodes and weights on [-1, 1] (exact to degree 2n-1)."""
    return np.polynomial.legendre.leggauss(n)


class LegendreBasis:
    """Precomputed basis data for degree p with an n_q-point volume quadrature.

    With the modal basis the reference mass matrix is diagonal,
        int_{-1}^{1} P_m P_n dr = 2 / (2m + 1) * delta_mn,
    so its inverse is stored as the vector ``mass_inv``.
    """

    def __init__(self, p, n_quad=None):
        if p < 0:
            raise ValueError("polynomial degree must be non-negative")
        self.p = p
        self.n_modes = p + 1
        # f(u_h) P'_m has degree 3p - 1; an n_q-point rule is exact when
        # 2 n_q - 1 >= 3p - 1.  The default adds a margin, so the volume
        # integral has no aliasing.
        self.n_quad = n_quad if n_quad is not None else max(p + 2, (3 * p + 2) // 2 + 1)
        self.r, self.w = gauss_legendre(self.n_quad)
        self.V, self.dV = legendre(p, self.r)          # (p+1, n_q)
        m = np.arange(self.n_modes)
        self.at_right = np.ones(self.n_modes)            # P_m(+1)
        self.at_left = (-1.0) ** m                       # P_m(-1)
        self.mass_diag = 2.0 / (2 * m + 1)
        self.mass_inv = (2 * m + 1) / 2.0
