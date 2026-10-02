"""Modal discontinuous Galerkin solver for periodic inviscid Burgers, u_t + (u^2/2)_x = 0.

Discretization
--------------
Uniform mesh of K cells on [x0, x1), width h.  On cell k, with x = x_k + (h/2) r,
    u_h(x, t) = sum_{m=0}^{p} c_{k,m}(t) P_m(r).
Testing against P_m and integrating by parts gives, with the diagonal mass
matrix (h/2) * 2/(2m+1),
    dc_{k,m}/dt = (2m+1)/h * [ int_{-1}^{1} f(u_h) P'_m dr
                               - F_{k+1/2} P_m(1) + F_{k-1/2} P_m(-1) ],
where F is a numerical flux of the two traces at an interface.  The volume
integral is evaluated by Gauss-Legendre quadrature (no aliasing by default).
Time integration: three-stage SSP Runge-Kutta (Shu-Osher), with an optional
limiter after every stage.
"""

import numpy as np

from .basis import LegendreBasis, gauss_legendre, legendre
from .flux import FLUXES, burgers_flux


class DGBurgers:
    def __init__(self, K, p, domain=(0.0, 2.0 * np.pi), flux="godunov",
                 limiter=None, n_quad=None, cfl=0.3):
        self.K = K
        self.p = p
        self.x0, self.x1 = domain
        self.h = (self.x1 - self.x0) / K
        self.edges = self.x0 + self.h * np.arange(K + 1)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.basis = LegendreBasis(p, n_quad)
        self.flux = FLUXES[flux] if isinstance(flux, str) else flux
        self.limiter = limiter
        self.cfl = cfl
        self.troubled = np.zeros(K, dtype=bool)   # flags from the latest limiter call

    # ------------------------------------------------------------------ geometry
    def x_at(self, r):
        """Physical coordinates of reference points r in every cell: (K, len(r))."""
        return self.centers[:, None] + 0.5 * self.h * np.asarray(r)[None, :]

    # ---------------------------------------------------------- projection/eval
    def project(self, func, n_quad=None):
        """L2 projection of func onto the DG space (exact up to quadrature)."""
        nq = n_quad if n_quad is not None else self.p + 8
        r, w = gauss_legendre(nq)
        P, _ = legendre(self.p, r)
        fq = func(self.x_at(r))                               # (K, nq)
        c = (fq * w) @ P.T * self.basis.mass_inv               # (K, p+1)
        return self.apply_limiter(c)

    def evaluate(self, c, r):
        P, _ = legendre(self.p, r)
        return c @ P

    def gradient(self, c, r):
        _, dP = legendre(self.p, r)
        return (2.0 / self.h) * (c @ dP)

    def sample(self, c, n_per_cell=8):
        """Points and values per cell, NaN-separated so plots show inter-cell jumps."""
        r = np.linspace(-1.0, 1.0, n_per_cell)
        x = self.x_at(r)
        u = self.evaluate(c, r)
        pad = np.full((self.K, 1), np.nan)
        return np.hstack([x, pad]).ravel(), np.hstack([u, pad]).ravel()

    # -------------------------------------------------------------- diagnostics
    def mass(self, c):
        """Integral of u_h over the domain: sum_k h c_{k,0}."""
        return self.h * c[:, 0].sum()

    def energy(self, c):
        """(1/2) * integral of u_h^2 = (1/2) sum_k (h/2) sum_m c_{k,m}^2 * 2/(2m+1)."""
        return 0.25 * self.h * np.sum(c * c * self.basis.mass_diag)

    def max_abs_gradient(self, c, n_per_cell=16):
        return np.max(np.abs(self.gradient(c, np.linspace(-1.0, 1.0, n_per_cell))))

    def errors(self, c, exact, n_quad=None, n_sample=24):
        """L1, L2 errors by Gauss quadrature and L_inf on a dense per-cell sample."""
        nq = n_quad if n_quad is not None else self.p + 5
        r, w = gauss_legendre(nq)
        e = self.evaluate(c, r) - exact(self.x_at(r))
        jac = 0.5 * self.h
        L1 = jac * np.sum(np.abs(e) * w)
        L2 = np.sqrt(jac * np.sum(e * e * w))
        rs = np.linspace(-1.0, 1.0, n_sample)
        Linf = np.max(np.abs(self.evaluate(c, rs) - exact(self.x_at(rs))))
        return L1, L2, Linf

    # ---------------------------------------------------------------- operators
    def apply_limiter(self, c):
        if self.limiter is None:
            return c
        c, self.troubled = self.limiter(c, self.h)
        return c

    def rhs(self, c):
        b = self.basis
        uq = c @ b.V                                           # (K, nq)
        volume = (burgers_flux(uq) * b.w) @ b.dV.T             # (K, p+1)
        u_right = c @ b.at_right                               # trace at x_{k+1/2}^-
        u_left = c @ b.at_left                                 # trace at x_{k-1/2}^+
        F_right = self.flux(u_right, np.roll(u_left, -1))      # F_{k+1/2}
        F_left = np.roll(F_right, 1)                           # F_{k-1/2}
        surface = F_right[:, None] * b.at_right - F_left[:, None] * b.at_left
        return (2.0 / self.h) * b.mass_inv * (volume - surface)

    def stable_dt(self, c):
        amax = np.max(np.abs(c @ self.basis.V))
        amax = max(amax, 1e-12)
        return self.cfl * self.h / ((2 * self.p + 1) * amax)

    def step(self, c, dt):
        """One SSP-RK3 step (Shu & Osher 1988), limiting after every stage."""
        L = self.apply_limiter
        u1 = L(c + dt * self.rhs(c))
        u2 = L(0.75 * c + 0.25 * (u1 + dt * self.rhs(u1)))
        return L(c / 3.0 + 2.0 / 3.0 * (u2 + dt * self.rhs(u2)))

    def run(self, c, t_final, t0=0.0, output_times=(), dt_max=None, on_step=None):
        """Advance to t_final, landing exactly on each requested output time.

        Returns (c_final, snapshots) where snapshots maps each output time to a
        copy of the coefficients.  on_step(t, c) is called after every step.
        """
        targets = {float(s) for s in output_times if t0 <= s <= t_final}
        stops = sorted(targets | {float(t_final)})
        snapshots = {}
        t = t0
        for stop in stops:
            while stop - t > 1e-14 * max(1.0, abs(stop)):
                dt = self.stable_dt(c)
                if dt_max is not None:
                    dt = min(dt, dt_max)
                dt = min(dt, stop - t)
                c = self.step(c, dt)
                t += dt
                if on_step is not None:
                    on_step(t, c)
            t = stop
            if stop in targets:
                snapshots[stop] = c.copy()
        return c, snapshots
