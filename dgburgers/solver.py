"""Modal discontinuous Galerkin (RKDG) solver for the 1-D inviscid Burgers equation.

    u_t + f(u)_x = 0,   f(u) = u^2 / 2,   periodic on [a, b).

In cell I_j = [x_j - h/2, x_j + h/2] the solution is
    u_h(x, t) = sum_l U_{j,l}(t) P_l(xi),   xi = 2 (x - x_j) / h,
with Legendre polynomials P_l, l = 0..p. Testing with P_l and using the
orthogonality  int P_l P_m dxi = 2/(2l+1) delta_lm  gives

    dU_{j,l}/dt = (2l+1)/h * [ int_{-1}^{1} f(u_h) P_l'(xi) dxi
                               - F_{j+1/2} P_l(1) + F_{j-1/2} P_l(-1) ],

where F is a monotone numerical flux. The volume integral is evaluated with
enough Gauss points to be exact for the polynomial integrand (degree 3p-1),
which gives the semi-discrete scheme the cell entropy inequality for the
square entropy (Jiang & Shu 1994). U_{j,0} is the cell average.
"""

import numpy as np

from .basis import gauss_legendre, legendre
from .timestepping import INTEGRATORS


def burgers_flux(u):
    return 0.5 * u * u


def godunov_flux(a, b):
    """Exact Riemann-solver (Godunov) flux for the convex flux u^2/2.

    For convex f with minimum at 0 this is max(f(max(a, 0)), f(min(b, 0))),
    which covers shocks, rarefactions and the transonic case in one formula.
    """
    return np.maximum(burgers_flux(np.maximum(a, 0.0)), burgers_flux(np.minimum(b, 0.0)))


def llf_flux(a, b):
    """Local Lax-Friedrichs (Rusanov) flux."""
    alpha = np.maximum(np.abs(a), np.abs(b))
    return 0.5 * (burgers_flux(a) + burgers_flux(b)) - 0.5 * alpha * (b - a)


FLUXES = {"godunov": godunov_flux, "llf": llf_flux}


def minmod(a, b, c):
    s = np.sign(a)
    agree = (np.sign(b) == s) & (np.sign(c) == s)
    return np.where(agree, s * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0)


def tvb_minmod(a, b, c, threshold):
    """Modified minmod of Cockburn & Shu: leave ``a`` alone if |a| <= M h^2."""
    return np.where(np.abs(a) <= threshold, a, minmod(a, b, c))


class DGBurgers:
    """Modal DG discretisation on a uniform periodic mesh.

    Parameters
    ----------
    n_cells, degree : mesh size and polynomial degree p.
    domain          : (a, b), periodic.
    flux            : "godunov" or "llf".
    limiter         : None or "minmod" (TVB-modified minmod slope limiter).
    tvb_M           : TVB constant M; M = 0 gives the TVD minmod limiter.
    """

    def __init__(self, n_cells, degree, domain=(0.0, 2.0 * np.pi), flux="godunov",
                 limiter=None, tvb_M=0.0, n_quad=None):
        if flux not in FLUXES:
            raise ValueError(f"unknown flux {flux!r}; choose from {sorted(FLUXES)}")
        if limiter not in (None, "minmod"):
            raise ValueError(f"unknown limiter {limiter!r}")
        self.n_cells = int(n_cells)
        self.degree = int(degree)
        self.n_modes = self.degree + 1
        self.domain = (float(domain[0]), float(domain[1]))
        self.length = self.domain[1] - self.domain[0]
        self.h = self.length / self.n_cells
        self.edges = self.domain[0] + self.h * np.arange(self.n_cells + 1)
        self.centers = self.edges[:-1] + 0.5 * self.h
        self.flux_name = flux
        self.flux = FLUXES[flux]
        self.limiter = limiter
        self.tvb_M = float(tvb_M)

        # Exact quadrature for f(u_h) P_l' (degree 3p - 1): n >= (3p + 1) / 2.
        self.n_quad = n_quad or max(1, (3 * self.degree + 2) // 2)
        self.xi_q, self.w_q = gauss_legendre(self.n_quad)
        P, dP = legendre(self.n_modes, self.xi_q)
        self._V = P.T                                  # (nq, nm): u_q = U @ V.T
        self._WdP = self.w_q[:, None] * dP.T           # (nq, nm): vol = f_q @ WdP
        l = np.arange(self.n_modes)
        self._left_sign = (-1.0) ** l                  # P_l(-1); P_l(+1) = 1
        self._inv_mass = (2 * l + 1) / self.h
        self._half_norm = (2 * l + 1) / 2.0            # inverse of int P_l^2 dxi
        self.last_flagged = np.zeros(self.n_cells, dtype=bool)

    # ------------------------------------------------------------------ geometry
    def x_at(self, xi):
        """Physical coordinates of reference points ``xi`` in every cell: (N, len(xi))."""
        return self.centers[:, None] + 0.5 * self.h * np.atleast_1d(xi)[None, :]

    # ----------------------------------------------------------- data transfer
    def project(self, func, n_quad=None):
        """L2 projection of ``func(x)`` onto the DG space (coefficients, shape (N, p+1))."""
        xi, w = gauss_legendre(n_quad or self.degree + 8)
        P, _ = legendre(self.n_modes, xi)
        return (func(self.x_at(xi)) * w) @ P.T * self._half_norm

    def evaluate(self, U, xi):
        """u_h at reference points ``xi`` in every cell: (N, len(xi))."""
        P, _ = legendre(self.n_modes, np.atleast_1d(xi))
        return U @ P

    def evaluate_slope(self, U, xi):
        """du_h/dx at reference points ``xi`` in every cell: (N, len(xi))."""
        _, dP = legendre(self.n_modes, np.atleast_1d(xi))
        return (2.0 / self.h) * (U @ dP)

    def traces(self, U):
        """(left, right) one-sided values of u_h at each cell's own edges."""
        return U @ self._left_sign, U.sum(axis=1)

    # -------------------------------------------------------------- operators
    def rhs(self, U):
        """Semi-discrete DG operator L(U)."""
        f_q = burgers_flux(U @ self._V.T)
        volume = f_q @ self._WdP
        left, right = self.traces(U)
        F = self.flux(right, np.roll(left, -1))        # F[j] lives at x_{j+1/2}
        surface = F[:, None] - np.roll(F, 1)[:, None] * self._left_sign[None, :]
        return self._inv_mass * (volume - surface)

    def limit(self, U):
        """TVB minmod slope limiter (Cockburn & Shu 1989), applied cell-wise.

        A cell is flagged when either edge deviation from the cell mean is
        modified by the TVB minmod; flagged cells are reduced to a limited
        linear polynomial. The cell means, hence the mass, are untouched.
        """
        if self.limiter is None or self.degree == 0:
            return U
        mean = U[:, 0]
        d_plus = np.roll(mean, -1) - mean
        d_minus = mean - np.roll(mean, 1)
        left, right = self.traces(U)
        dev_r = right - mean
        dev_l = mean - left
        thr = self.tvb_M * self.h ** 2
        flagged = ((tvb_minmod(dev_r, d_plus, d_minus, thr) != dev_r)
                   | (tvb_minmod(dev_l, d_plus, d_minus, thr) != dev_l))
        self.last_flagged = flagged
        if not flagged.any():
            return U
        V = U.copy()
        V[flagged, 1] = minmod(U[flagged, 1], d_plus[flagged], d_minus[flagged])
        V[flagged, 2:] = 0.0
        return V

    def troubled_cells(self, U):
        """Boolean mask of cells the limiter would modify (diagnostic).

        A limited state is a fixed point of the limiter, so for limited runs
        use ``last_flagged`` (set by the most recent limiter call) instead.
        """
        if self.degree == 0:
            return np.zeros(self.n_cells, dtype=bool)
        saved, saved_flags = self.limiter, self.last_flagged
        self.limiter = "minmod"
        try:
            return np.any(self.limit(U) != U, axis=1)
        finally:
            self.limiter, self.last_flagged = saved, saved_flags

    # ---------------------------------------------------------- time stepping
    def max_speed(self, U):
        left, right = self.traces(U)
        return max(np.abs(U @ self._V.T).max(), np.abs(left).max(), np.abs(right).max())

    def stable_dt(self, U, cfl):
        """dt = cfl * h / ((2p + 1) max|u|), the usual RKDG CFL scaling."""
        return cfl * self.h / ((2 * self.degree + 1) * max(self.max_speed(U), 1e-12))

    def solve(self, U0, t_end, cfl=0.5, integrator="ssprk3", dt=None, save_times=None,
              callback=None):
        """Advance ``U0`` from t = 0 to ``t_end``.

        The step is recomputed from the CFL condition every step (or fixed if
        ``dt`` is given) and shortened to land exactly on each entry of
        ``save_times`` and on ``t_end``. Returns a dict with the final state,
        the snapshots at ``save_times`` and the number of steps taken.
        ``callback(t, U)`` is called after every step (and at t = 0).
        """
        step = INTEGRATORS[integrator]
        saves = {float(s) for s in ([] if save_times is None else save_times) if 0.0 <= s <= t_end}
        stops = sorted(saves | {float(t_end)})
        U = self.limit(np.array(U0, dtype=float))
        t = 0.0
        snapshots = {}
        if 0.0 in saves:
            snapshots[0.0] = U.copy()
        if callback is not None:
            callback(t, U)
        n_steps = 0
        for stop in stops:
            while t < stop:
                k = dt if dt is not None else self.stable_dt(U, cfl)
                landing = t + k >= stop * (1.0 - 1e-14)
                if landing:
                    k = stop - t
                U = step(self.rhs, self.limit, U, k)
                t = stop if landing else t + k
                n_steps += 1
                if not np.all(np.isfinite(U)):
                    raise FloatingPointError(f"solution became non-finite at t = {t:.6g}")
                if callback is not None:
                    callback(t, U)
            if stop in saves:
                snapshots[stop] = U.copy()
        return {"U": U, "t": t, "snapshots": snapshots, "n_steps": n_steps}
