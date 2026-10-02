"""Modal discontinuous Galerkin discretisation of the periodic inviscid Burgers equation

    u_t + (u^2 / 2)_x = 0,    x in [a, b) with periodic boundaries.

On cell I_j = [x_{j-1/2}, x_{j+1/2}] of width h the solution is
u_h = sum_k c_{j,k} P_k(xi), with xi = 2 (x - x_j) / h in [-1, 1] and P_k the
Legendre polynomials. Testing the weak form against P_k and using the diagonal
mass matrix (h / (2k + 1)) gives the semi-discrete system

    dc_{j,k}/dt = (2k + 1)/h * [ int_{-1}^{1} f(u_h) P_k'(xi) dxi
                                 - F_{j+1/2} P_k(1) + F_{j-1/2} P_k(-1) ],

where F is a monotone numerical flux. The volume integral is evaluated with a
Gauss rule that is exact for the degree-(3p - 1) integrand, so the scheme
satisfies the Jiang-Shu cell entropy inequality for the square entropy.
Time integration uses the three-stage SSP Runge-Kutta method, optionally with
the Cockburn-Shu TVB minmod limiter applied after every stage.
"""

import numpy as np

from .basis import gauss_legendre, legendre, legendre_derivative


def burgers_flux(u):
    return 0.5 * u * u


def godunov_flux(u_left, u_right):
    """Exact Riemann-solver (Godunov) flux for the convex flux f(u) = u^2 / 2."""
    f_left = burgers_flux(u_left)
    f_right = burgers_flux(u_right)
    # u_left > u_right: shock, F = max over [u_right, u_left] of f (attained at an endpoint).
    shock = np.maximum(f_left, f_right)
    # u_left <= u_right: rarefaction, F = min over [u_left, u_right] of f (0 if transonic).
    rarefaction = np.where(u_left > 0.0, f_left, np.where(u_right < 0.0, f_right, 0.0))
    return np.where(u_left > u_right, shock, rarefaction)


def llf_flux(u_left, u_right):
    """Local Lax-Friedrichs (Rusanov) flux."""
    alpha = np.maximum(np.abs(u_left), np.abs(u_right))
    return 0.5 * (burgers_flux(u_left) + burgers_flux(u_right)) - 0.5 * alpha * (u_right - u_left)


FLUXES = {"godunov": godunov_flux, "llf": llf_flux}


def minmod(a, b, c):
    """Elementwise minmod of three arrays."""
    s = np.sign(a)
    same_sign = (np.sign(b) == s) & (np.sign(c) == s)
    return np.where(same_sign, s * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)


def tvb_minmod(a, b, c, threshold):
    """TVB-modified minmod: leave ``a`` untouched when |a| <= threshold."""
    return np.where(np.abs(a) <= threshold, a, minmod(a, b, c))


class BurgersDG:
    """DG-P``degree`` solver for periodic Burgers on ``n_cells`` uniform cells.

    Parameters
    ----------
    limiter : None or "tvb"
        "tvb" applies the Cockburn-Shu TVB minmod limiter with constant ``tvb_M``
        (``tvb_M = 0`` gives the TVD minmod limiter).
    cfl : float
        Time step is ``cfl * h / ((2p + 1) * max|u|)``.
    """

    def __init__(self, n_cells, degree, domain=(0.0, 1.0), flux="godunov",
                 limiter=None, tvb_M=0.0, cfl=0.8):
        if limiter not in (None, "tvb"):
            raise ValueError(f"unknown limiter {limiter!r}")
        self.n_cells = int(n_cells)
        self.degree = int(degree)
        self.domain = (float(domain[0]), float(domain[1]))
        self.length = self.domain[1] - self.domain[0]
        self.h = self.length / self.n_cells
        self.edges = self.domain[0] + self.h * np.arange(self.n_cells + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.flux_name = flux
        self.flux = FLUXES[flux]
        self.limiter = limiter
        self.tvb_M = float(tvb_M)
        self.cfl = float(cfl)

        p = self.degree
        self.modes = np.arange(p + 1)
        # Integrand f(u_h) P_k' has degree 2p + (p - 1) = 3p - 1; n-point Gauss is exact to 2n - 1.
        n_quad = max(1, (3 * p) // 2 + 1)
        self.xi_q, self.w_q = gauss_legendre(n_quad)
        self.V_q = legendre(self.xi_q, p)                                   # (n_quad, p+1)
        self.Dw_q = legendre_derivative(self.xi_q, p) * self.w_q[:, None]   # (n_quad, p+1)
        self.inv_mass = (2 * self.modes + 1) / self.h                       # diagonal M^{-1}
        self.left_sign = (-1.0) ** self.modes                               # P_k(-1)
        self.last_limited = np.zeros(self.n_cells, dtype=bool)

    # ------------------------------------------------------------------ geometry
    def x_physical(self, xi):
        """Physical coordinates of reference points xi in every cell, shape (N, len(xi))."""
        return self.centers[:, None] + 0.5 * self.h * np.asarray(xi)[None, :]

    # ------------------------------------------------------------------ fields
    def project(self, func, n_quad=None):
        """L2 projection of ``func(x)`` onto the DG space; returns coefficients (N, p+1)."""
        n_quad = n_quad or self.degree + 10
        xi, w = gauss_legendre(n_quad)
        values = func(self.x_physical(xi))
        return 0.5 * (2 * self.modes + 1) * ((values * w) @ legendre(xi, self.degree))

    def evaluate(self, C, xi):
        """Values of u_h at reference points xi in every cell, shape (N, len(xi))."""
        return C @ legendre(xi, self.degree).T

    def gradient(self, C, xi):
        """du_h/dx at reference points xi in every cell."""
        return (2.0 / self.h) * (C @ legendre_derivative(xi, self.degree).T)

    def traces(self, C):
        """(u at left edge, u at right edge) of every cell."""
        return C @ self.left_sign, C.sum(axis=1)

    def max_speed(self, C):
        u_left, u_right = self.traces(C)
        return max(np.abs(C @ self.V_q.T).max(), np.abs(u_left).max(), np.abs(u_right).max())

    # ------------------------------------------------------------------ spatial operator
    def rhs(self, C):
        u_q = C @ self.V_q.T
        volume = burgers_flux(u_q) @ self.Dw_q
        u_left, u_right = self.traces(C)
        # Interface j+1/2 separates the right trace of cell j and the left trace of cell j+1.
        F_right = self.flux(u_right, np.roll(u_left, -1))
        F_left = np.roll(F_right, 1)
        return self.inv_mass * (volume - F_right[:, None] + F_left[:, None] * self.left_sign)

    # ------------------------------------------------------------------ limiter
    def apply_limiter(self, C):
        if self.limiter is None or self.degree == 0:
            self.last_limited = np.zeros(self.n_cells, dtype=bool)
            return C
        avg = C[:, 0]
        d_plus = np.roll(avg, -1) - avg
        d_minus = avg - np.roll(avg, 1)
        u_left, u_right = self.traces(C)
        threshold = self.tvb_M * self.h ** 2
        dev_right = u_right - avg
        dev_left = avg - u_left
        limited = ((tvb_minmod(dev_right, d_plus, d_minus, threshold) != dev_right)
                   | (tvb_minmod(dev_left, d_plus, d_minus, threshold) != dev_left))
        self.last_limited = limited
        if not limited.any():
            return C
        C = C.copy()
        # Troubled cells: drop to P1 and limit the slope; the cell average is untouched.
        C[limited, 1] = tvb_minmod(C[limited, 1], d_plus[limited], d_minus[limited], threshold)
        C[limited, 2:] = 0.0
        return C

    # ------------------------------------------------------------------ time stepping
    def stable_dt(self, C):
        return self.cfl * self.h / ((2 * self.degree + 1) * max(self.max_speed(C), 1e-14))

    def ssp_rk3_step(self, C, dt):
        limit, L = self.apply_limiter, self.rhs
        u1 = limit(C + dt * L(C))
        u2 = limit(0.75 * C + 0.25 * (u1 + dt * L(u1)))
        return limit(C / 3.0 + 2.0 / 3.0 * (u2 + dt * L(u2)))

    def advance(self, C, t_start, t_end, dt_max=np.inf):
        """Integrate from t_start to exactly t_end. Returns (C, number of steps)."""
        t, steps = t_start, 0
        while t_end - t > 1e-14 * max(1.0, abs(t_end)):
            dt = min(self.stable_dt(C), dt_max, t_end - t)
            C = self.ssp_rk3_step(C, dt)
            t += dt
            steps += 1
        return C, steps
