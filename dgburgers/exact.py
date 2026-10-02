"""Exact entropy solution of Burgers' equation for sinusoidal periodic data.

Problem:  u_t + (u^2/2)_x = 0  on [0, 2*pi), periodic,
          u(x, 0) = c + A sin(x),   A > 0.

Writing u = c + A v(y, tau) with y = x - c t and tau = A t reduces the problem
to v_tau + v v_y = 0 with v(y, 0) = sin(y). Along characteristics
y = xi + tau sin(xi), so the smooth solution is v = sin(xi). The gradient
blows up at xi = pi when tau = 1, so the shock forms at t_b = 1/A at
x = pi + c t_b. By the odd symmetry of v about y = pi, the shock then stays at
y = pi (that is, x_s = pi + c t), and the entropy solution on each side is
the unique root of xi + tau sin(xi) = y with xi in [0, pi] for y < pi and in
[pi, 2*pi] for y > pi. The same root selection also holds for tau < 1, so a
single bracketed bisection gives the exact solution for all t >= 0.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def initial_condition(x, c=0.5, A=1.0):
    return c + A * np.sin(x)


def breaking_time(A=1.0):
    """Time at which the characteristics first cross (gradient catastrophe)."""
    return 1.0 / A


def shock_position(t, c=0.5):
    """Location of the (single) shock for t >= t_b, wrapped into [0, 2*pi)."""
    return np.mod(np.pi + c * t, TWO_PI)


def _foot_point(y, tau, iters=64):
    """Solve xi + tau*sin(xi) = y by bracketed bisection (vectorised)."""
    left = y < np.pi
    lo = np.where(left, 0.0, np.pi)
    hi = np.where(left, np.pi, TWO_PI)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = mid + tau * np.sin(mid) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def exact_solution(x, t, c=0.5, A=1.0):
    """Exact entropy solution u(x, t), valid before and after shock formation."""
    if A <= 0:
        raise ValueError("amplitude A must be positive")
    x = np.asarray(x, dtype=float)
    y = np.mod(x - c * t, TWO_PI)
    xi = _foot_point(y, A * t)
    return c + A * np.sin(xi)


def exact_slope(x, t, c=0.5, A=1.0):
    """du/dx of the exact solution (valid away from the shock)."""
    y = np.mod(np.asarray(x, dtype=float) - c * t, TWO_PI)
    xi = _foot_point(y, A * t)
    return A * np.cos(xi) / (1.0 + A * t * np.cos(xi))


def exact_energy(t, c=0.5, A=1.0, n=400):
    """int_0^{2 pi} u^2 / 2 dx of the exact solution.

    The integrand is smooth on each side of y = pi (where the shock sits for
    t > t_b), so Gauss quadrature is applied separately on the two halves.
    """
    from .basis import gauss_legendre

    xi, w = gauss_legendre(n)
    total = 0.0
    for lo, hi in ((0.0, np.pi), (np.pi, TWO_PI)):
        y = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xi
        u = exact_solution(y + c * t, t, c, A)
        total += 0.25 * (hi - lo) * (u ** 2 @ w)
    return float(total)


def characteristic_curve(t, c=0.5, A=1.0, n=2001):
    """The (possibly multivalued) curve traced by characteristics at time t.

    Returns ``(x, u)`` with x wrapped into [0, 2*pi); NaNs are inserted where
    the curve wraps around the periodic boundary so it plots cleanly.
    """
    xi = np.linspace(0.0, TWO_PI, n)
    u = initial_condition(xi, c, A)
    x = np.mod(xi + u * t, TWO_PI)
    jumps = np.where(np.abs(np.diff(x)) > np.pi)[0] + 1
    return np.insert(x, jumps, np.nan), np.insert(u, jumps, np.nan)
