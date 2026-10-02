"""Reference solutions of the inviscid Burgers equation with periodic sine initial data.

Before the shock time the solution is constant along characteristics,
u(x, t) = u0(xi) with xi + t u0(xi) = x. For any t > 0 the entropy solution is
given by the Hopf-Lax formula

    u(x, t) = (x - y*) / t,   y* = argmin_y [ U0(y) + (x - y)^2 / (2t) ],

where U0 is an antiderivative of u0. Both are evaluated to near machine precision
(the Hopf-Lax minimiser is located on a grid and then polished with Newton).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SineInitialData:
    """u0(x) = mean + amplitude * sin(2 pi x / length), amplitude > 0."""

    mean: float = 0.5
    amplitude: float = 1.0
    length: float = 1.0

    @property
    def wavenumber(self):
        return 2.0 * np.pi / self.length

    def u0(self, x):
        return self.mean + self.amplitude * np.sin(self.wavenumber * x)

    def du0(self, x):
        return self.amplitude * self.wavenumber * np.cos(self.wavenumber * x)

    def U0(self, x):
        """Antiderivative of u0."""
        return self.mean * x - self.amplitude / self.wavenumber * np.cos(self.wavenumber * x)

    @property
    def u_min(self):
        return self.mean - self.amplitude

    @property
    def u_max(self):
        return self.mean + self.amplitude

    @property
    def shock_time(self):
        """Breaking time t_s = -1 / min u0'."""
        return 1.0 / (self.amplitude * self.wavenumber)

    @property
    def shock_location(self):
        """Where the shock first forms: the characteristic from the steepest descent point L/2."""
        return (0.5 * self.length + self.shock_time * self.mean) % self.length

    def max_gradient(self, t):
        """max_x |u_x(x, t)| = a k / (1 - a k t) for t < t_s (infinite afterwards)."""
        ak = self.amplitude * self.wavenumber
        return ak / (1.0 - ak * t) if t < self.shock_time else np.inf


def _safeguarded_newton(g, dg, lo, hi, x0, tol=1e-15, max_iter=200):
    """Vectorised root of increasing-through-zero g on brackets [lo, hi] with g(lo) <= 0 <= g(hi)."""
    lo, hi, x = lo.copy(), hi.copy(), np.clip(x0, lo, hi)
    for _ in range(max_iter):
        gx = g(x)
        lo = np.where(gx < 0.0, x, lo)
        hi = np.where(gx > 0.0, x, hi)
        slope = dg(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_new = x - gx / slope
        reject = ~((x_new > lo) & (x_new < hi)) | ~(slope > 0.0)
        x_new = np.where(gx == 0.0, x, np.where(reject, 0.5 * (lo + hi), x_new))
        done = np.abs(x_new - x).max(initial=0.0) <= tol * max(1.0, np.abs(x).max(initial=0.0))
        x = x_new
        if done:
            break
    return x


def characteristic_foot(x, t, data):
    """Solve xi + t u0(xi) = x for xi (valid for t < shock time)."""
    x = np.asarray(x, dtype=float)
    if t == 0.0:
        return x.copy()
    lo = x - t * data.u_max
    hi = x - t * data.u_min
    return _safeguarded_newton(
        lambda xi: xi + t * data.u0(xi) - x,
        lambda xi: 1.0 + t * data.du0(xi),
        lo, hi, x - t * data.u0(x),
    )


def hopf_lax_solution(x, t, data, n_grid=4001, chunk=256):
    """Entropy solution at time t > 0 via the Hopf-Lax formula."""
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    u = np.empty_like(flat)
    s = np.linspace(0.0, 1.0, n_grid)
    for start in range(0, flat.size, chunk):
        xc = flat[start:start + chunk]
        # Every minimiser is a characteristic foot, so it lies in [x - t u_max, x - t u_min].
        lo = xc - t * data.u_max
        hi = xc - t * data.u_min
        y = lo[:, None] + (hi - lo)[:, None] * s[None, :]
        objective = data.U0(y) + (xc[:, None] - y) ** 2 / (2.0 * t)
        i = np.clip(np.argmin(objective, axis=1), 1, n_grid - 2)
        rows = np.arange(xc.size)
        y_grid = y[rows, i]
        a, b = y[rows, i - 1], y[rows, i + 1]
        # d(objective)/dy = (y + t u0(y) - x) / t changes sign from - to + at the minimiser.
        g = lambda yy: yy + t * data.u0(yy) - xc
        bracketed = (g(a) <= 0.0) & (g(b) >= 0.0)
        y_star = _safeguarded_newton(g, lambda yy: 1.0 + t * data.du0(yy), a, b, y_grid)
        y_star = np.where(bracketed, y_star, y_grid)
        u[start:start + chunk] = (xc - y_star) / t
    return u.reshape(x.shape)


def exact_solution(x, t, data):
    """Entropy solution u(x, t): characteristics before the shock, Hopf-Lax afterwards."""
    if t < data.shock_time:
        return data.u0(characteristic_foot(x, t, data))
    return hopf_lax_solution(x, t, data)
