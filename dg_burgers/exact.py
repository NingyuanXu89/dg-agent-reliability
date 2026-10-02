"""Exact (pre-shock) solution of inviscid Burgers' equation by characteristics.

For u_t + (u^2/2)_x = 0 with smooth u0, u(x, t) = u0(xi) where xi solves
xi + t * u0(xi) = x. The map xi -> xi + t*u0(xi) is strictly increasing for
t < t_s = 1 / max(-u0'), so the root is unique.
"""

import numpy as np

L_DOMAIN = 2.0 * np.pi
U_MEAN = 0.5
AMPLITUDE = 1.0


def u0(x):
    """Default initial condition: 0.5 + sin(x) on the periodic domain [0, 2*pi]."""
    return U_MEAN + AMPLITUDE * np.sin(x)


def du0(x):
    return AMPLITUDE * np.cos(x)


def breaking_time():
    """t_s = 1 / max(-u0'(x)) = 1 / AMPLITUDE."""
    return 1.0 / AMPLITUDE


def shock_formation_point():
    """Location where the shock first appears: characteristic from xi = pi."""
    ts = breaking_time()
    return np.pi + u0(np.pi) * ts


def foot_of_characteristic(x, t, tol=1e-14, maxiter=100):
    """Solve xi + t*u0(xi) = x for xi (vectorized safeguarded Newton).

    Valid for 0 <= t < t_s. Uses a bracket [x - t*umax, x - t*umin] that
    always contains the root and falls back to bisection when Newton leaves it.
    """
    x = np.asarray(x, dtype=float)
    if t == 0.0:
        return x.copy()
    if t >= breaking_time():
        raise ValueError("characteristic solution is multivalued for t >= t_s")
    umin, umax = U_MEAN - AMPLITUDE, U_MEAN + AMPLITUDE
    lo = x - t * umax
    hi = x - t * umin
    xi = x - t * u0(x)
    xi = np.clip(xi, lo, hi)
    for _ in range(maxiter):
        g = xi + t * u0(xi) - x
        # g is increasing in xi, so update the bracket from the sign of g
        lo = np.where(g < 0, xi, lo)
        hi = np.where(g > 0, xi, hi)
        dg = 1.0 + t * du0(xi)
        step = g / dg
        xn = xi - step
        bad = (xn <= lo) | (xn >= hi) | ~np.isfinite(xn)
        xn = np.where(bad, 0.5 * (lo + hi), xn)
        if np.max(np.abs(xn - xi)) < tol:
            return xn
        xi = xn
    return xi


def exact_solution(x, t):
    """u(x, t) for 0 <= t < t_s."""
    return u0(foot_of_characteristic(x, t))


def shock_position(t):
    """Shock location for t >= t_s (mod 2*pi).

    In the frame moving with U_MEAN the data A*sin is odd about pi, so the
    shock stays at pi in that frame (equal and opposite states, zero RH speed).
    """
    return np.mod(np.pi + U_MEAN * t, L_DOMAIN)


def entropy_solution(x, t, iters=80):
    """Exact entropy solution for any t >= 0 (specific to u0 = U_MEAN + A sin x).

    With y = x - U_MEAN*t (moving frame), w = u - U_MEAN solves Burgers with
    w0 = A sin(y), whose shock (after t_s) is pinned at y = pi. For y in [0, pi]
    the physical characteristic starts at xi in [0, pi] on the monotone branch
    of g(xi) = xi + t*A*sin(xi); y in [pi, 2*pi] follows by odd symmetry.
    Roots are found by vectorized bisection.
    """
    x = np.asarray(x, dtype=float)
    y = np.mod(x - U_MEAN * t, L_DOMAIN)
    right = y > np.pi
    yr = np.where(right, L_DOMAIN - y, y)          # reflect to [0, pi]
    at = AMPLITUDE * t
    xi_max = np.pi if at <= 1.0 else np.arccos(-1.0 / at)
    lo = np.zeros_like(yr)
    hi = np.full_like(yr, xi_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g = mid + at * np.sin(mid) - yr
        lo = np.where(g < 0, mid, lo)
        hi = np.where(g < 0, hi, mid)
    w = AMPLITUDE * np.sin(0.5 * (lo + hi))
    return U_MEAN + np.where(right, -w, w)


def max_gradient_exact(t):
    """max |u_x| of the exact solution: |u0'| / (1 + t u0') maximised at xi = pi."""
    return AMPLITUDE / (1.0 - AMPLITUDE * t)
