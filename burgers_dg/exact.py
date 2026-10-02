"""Exact entropy solution of periodic Burgers for u0(x) = c + a sin(x) on [0, 2*pi).

Along characteristics u is constant: u(x, t) = u0(xi) with xi + t u0(xi) = x.
In the frame moving with the mean speed c, y = x - c t, eta = xi - c t gives
    eta + a t sin(eta) = y   (mod 2 pi),     u = c + a sin(eta).
In that frame the data are odd about eta = pi, so the shock that forms at
t_s = 1/a stays at y = pi, i.e. x_s(t) = pi + c t (mod 2 pi).  Its speed c is
the Rankine-Hugoniot speed (u_left + u_right)/2 = c, consistent with symmetry.

For t > t_s, g(eta) = eta + a t sin(eta) folds near pi.  The entropy solution
takes, left of the shock (y < pi), the root on [0, eta*], and right of the
shock the root on [2 pi - eta*, 2 pi], where eta* = arccos(-1/(a t)) is where
g' = 0.  On those intervals g is monotone, so bisection is unconditionally
robust.  Before t_s, eta* = pi and the two intervals cover the whole period.
"""

import numpy as np

TWO_PI = 2.0 * np.pi


def shock_time(a):
    return 1.0 / a


def shock_position(t, c):
    """Shock location for t >= t_s, wrapped into [0, 2 pi)."""
    return np.mod(np.pi + c * t, TWO_PI)


def _fold_point(at):
    """eta* such that g is monotone on [0, eta*]; pi before the shock."""
    return np.pi if at <= 1.0 else float(np.arccos(-1.0 / at))


def characteristic_foot(x, t, c=0.5, a=1.0, iters=64):
    """Solve for eta (moving-frame foot point) at positions x and time t."""
    x = np.asarray(x, dtype=float)
    y = np.mod(x - c * t, TWO_PI)
    at = a * t
    eta_star = _fold_point(at)
    left = y < np.pi
    lo = np.where(left, 0.0, TWO_PI - eta_star)
    hi = np.where(left, eta_star, TWO_PI)
    # g is increasing on [lo, hi]; bisect g(eta) - y.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        gm = mid + at * np.sin(mid) - y
        lo = np.where(gm < 0.0, mid, lo)
        hi = np.where(gm < 0.0, hi, mid)
    return 0.5 * (lo + hi)


def exact_solution(x, t, c=0.5, a=1.0):
    eta = characteristic_foot(x, t, c, a)
    return c + a * np.sin(eta)


def exact_gradient(x, t, c=0.5, a=1.0):
    """u_x = u0'(xi) / (1 + t u0'(xi)) along characteristics (smooth part)."""
    eta = characteristic_foot(x, t, c, a)
    du0 = a * np.cos(eta)
    return du0 / (1.0 + t * du0)


def exact_max_abs_gradient(t, a=1.0):
    """Peak |u_x| before the shock: attained at eta = pi, equals a / (1 - a t)."""
    return a / (1.0 - a * t)
