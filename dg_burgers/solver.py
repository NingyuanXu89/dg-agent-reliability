"""Modal discontinuous Galerkin solver for periodic 1D inviscid Burgers' equation.

    u_t + (u^2 / 2)_x = 0,   x in [x_left, x_right), periodic.

On each cell I_i = [x_{i-1/2}, x_{i+1/2}] of width h the solution is
u_h = sum_j U[i, j] P_j(xi), with P_j the Legendre polynomials and
xi = 2 (x - x_i) / h. Testing with P_j and using P_j(+-1) = (+-1)^j gives

    dU_ij/dt = (2j+1)/h * [ int_{-1}^{1} f(u_h) P_j'(xi) dxi
                            - ( F_{i+1/2} - (-1)^j F_{i-1/2} ) ],

where F is a numerical flux. Time integration uses SSP-RK3 (Shu-Osher), with
an optional TVB minmod slope limiter (Cockburn-Shu) after each stage.
"""

import math

import numpy as np

from .basis import gauss_legendre, legendre_derivatives, legendre_values


def flux(u):
    return 0.5 * u * u


def godunov_flux(a, b):
    """Exact Riemann-solver (Godunov) flux for Burgers with left state a, right state b."""
    fa, fb = flux(a), flux(b)
    # a <= b (rarefaction): min of f over [a, b], which is 0 if the interval contains 0
    rare = np.where((a <= 0.0) & (b >= 0.0), 0.0, np.minimum(fa, fb))
    # a > b (shock): max of f over [b, a]
    shock = np.maximum(fa, fb)
    return np.where(a <= b, rare, shock)


def rusanov_flux(a, b):
    """Local Lax-Friedrichs (Rusanov) flux."""
    s = np.maximum(np.abs(a), np.abs(b))
    return 0.5 * (flux(a) + flux(b)) - 0.5 * s * (b - a)


FLUXES = {"godunov": godunov_flux, "rusanov": rusanov_flux}


def minmod(a, b, c):
    s = np.sign(a)
    same = (np.sign(b) == s) & (np.sign(c) == s)
    return np.where(same, s * np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c)), 0.0)


def tvb_minmod(a, b, c, Mh2):
    """TVB-modified minmod: leave a untouched if |a| <= M h^2."""
    return np.where(np.abs(a) <= Mh2, a, minmod(a, b, c))


class DGBurgers:
    """DG discretisation with N uniform cells and polynomial degree p."""

    def __init__(self, N, p, x_left=0.0, x_right=2.0 * np.pi, flux="godunov",
                 limiter=False, tvb_M=0.0, cfl=0.5):
        if p < 0:
            raise ValueError("p must be >= 0")
        self.N, self.p = int(N), int(p)
        self.x_left, self.x_right = float(x_left), float(x_right)
        self.length = self.x_right - self.x_left
        self.h = self.length / self.N
        self.edges = self.x_left + self.h * np.arange(self.N + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.flux_name = flux
        self.numerical_flux = FLUXES[flux]
        self.limiter = limiter
        self.tvb_M = float(tvb_M)
        self.cfl = float(cfl)

        # Volume quadrature: integrand f(u_h) P_j' has degree 2p + (p - 1) = 3p - 1.
        nq = max(self.p + 1, math.ceil(1.5 * self.p) + 1)
        self.xq, self.wq = gauss_legendre(nq)
        self.Vq = legendre_values(self.p, self.xq)          # (nq, p+1)
        self.Dq = legendre_derivatives(self.p, self.xq)     # (nq, p+1)
        self.WDq = self.wq[:, None] * self.Dq               # weighted derivative matrix
        j = np.arange(self.p + 1)
        self.sign_left = (-1.0) ** j                        # P_j(-1)
        self.scale = (2.0 * j + 1.0) / self.h               # inverse mass matrix

    # ------------------------------------------------------------------ setup
    def project(self, f, nq=None):
        """L2 projection of f(x) onto the DG space (high-order quadrature)."""
        nq = nq or (self.p + 8)
        xi, w = gauss_legendre(nq)
        V = legendre_values(self.p, xi)
        x = self.centers[:, None] + 0.5 * self.h * xi[None, :]
        fx = f(x)
        j = np.arange(self.p + 1)
        return 0.5 * (2.0 * j + 1.0) * ((fx * w) @ V)

    def evaluate(self, U, xi):
        """Values of u_h at reference points xi in every cell -> (x, u) arrays of shape (N, len(xi))."""
        xi = np.atleast_1d(xi)
        x = self.centers[:, None] + 0.5 * self.h * xi[None, :]
        return x, U @ legendre_values(self.p, xi).T

    def cell_averages(self, U):
        return U[:, 0].copy()

    # ------------------------------------------------------------- operators
    def traces(self, U):
        """Values at the right (xi=+1) and left (xi=-1) end of each cell."""
        return U.sum(axis=1), U @ self.sign_left

    def rhs(self, U):
        uq = U @ self.Vq.T                                   # (N, nq)
        vol = flux(uq) @ self.WDq                            # (N, p+1)
        u_right_end, u_left_end = self.traces(U)
        # Interface i+1/2: left state from cell i, right state from cell i+1 (periodic).
        F_right = self.numerical_flux(u_right_end, np.roll(u_left_end, -1))
        F_left = np.roll(F_right, 1)
        surf = F_right[:, None] - F_left[:, None] * self.sign_left[None, :]
        return self.scale[None, :] * (vol - surf)

    def limit(self, U):
        """TVB minmod slope limiter (Cockburn & Shu). Preserves cell averages."""
        if not self.limiter or self.p == 0:
            return U
        ubar = U[:, 0]
        dplus = np.roll(ubar, -1) - ubar
        dminus = ubar - np.roll(ubar, 1)
        u_right_end, u_left_end = self.traces(U)
        Mh2 = self.tvb_M * self.h ** 2
        dev_r = u_right_end - ubar
        dev_l = ubar - u_left_end
        dev_r_mod = tvb_minmod(dev_r, dplus, dminus, Mh2)
        dev_l_mod = tvb_minmod(dev_l, dplus, dminus, Mh2)
        flagged = (dev_r_mod != dev_r) | (dev_l_mod != dev_l)
        if not np.any(flagged):
            return U
        V = U.copy()
        slope = tvb_minmod(U[flagged, 1], dplus[flagged], dminus[flagged], Mh2)
        V[flagged, 1] = slope
        V[flagged, 2:] = 0.0
        return V

    # -------------------------------------------------------- time stepping
    def stable_dt(self, U):
        umax = max(np.max(np.abs(U @ self.Vq.T)), np.max(np.abs(U.sum(axis=1))),
                   np.max(np.abs(U @ self.sign_left)), 1e-12)
        return self.cfl * self.h / ((2 * self.p + 1) * umax)

    def ssprk3_step(self, U, dt):
        U1 = self.limit(U + dt * self.rhs(U))
        U2 = self.limit(0.75 * U + 0.25 * (U1 + dt * self.rhs(U1)))
        return self.limit(U / 3.0 + 2.0 / 3.0 * (U2 + dt * self.rhs(U2)))

    def run(self, U, t_end, t_start=0.0, output_times=None, callback=None, dt_factor=1.0):
        """Advance U from t_start to t_end.

        Steps are shortened to land exactly on each time in output_times, where
        callback(t, U) is invoked (also at t_start if listed). Returns (U, nsteps).
        """
        U = self.limit(np.array(U, dtype=float, copy=True))
        t = float(t_start)
        outputs = set(float(s) for s in (output_times if output_times is not None else [])
                      if t_start <= s <= t_end)
        if callback is not None and t in outputs:
            callback(t, U)
        stops = sorted(s for s in outputs if s > t)
        if not stops or stops[-1] < t_end:
            stops.append(float(t_end))
        nsteps = 0
        for stop in stops:
            while t < stop - 1e-14 * max(1.0, abs(stop)):
                dt = min(dt_factor * self.stable_dt(U), stop - t)
                U = self.ssprk3_step(U, dt)
                t += dt
                nsteps += 1
            t = stop
            if callback is not None and stop in outputs:
                callback(t, U)
        return U, nsteps
