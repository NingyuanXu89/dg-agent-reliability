"""Independent entropy solution for u0(x)=1+sin(x); no DG machinery."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq

LENGTH = 2*np.pi


def shock_position(t, origin=0.):
    return origin + (np.pi+t-origin) % LENGTH


@lru_cache(maxsize=1024)
def shock_foot(t):
    if t <= 1:
        return np.pi
    turning = np.arccos(-1/t)
    return brentq(lambda q: q+t*np.sin(q)-np.pi, 0., turning,
                  xtol=5e-15, rtol=1e-14)


def exact_solution(x, t):
    """Return the characteristic/entropy solution; define the shock point as 1.

    After breaking, select the surviving monotone characteristic branches.
    This is specific to the default initial condition, not a general PDE solver.
    """
    if not np.isfinite(t) or t < 0:
        raise ValueError("reference time must be finite and nonnegative")
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("reference coordinates must be finite")
    if t == 0:
        return 1+np.sin(x)
    y = (x-t) % LENGTH
    result = np.empty(y.size)
    q = shock_foot(float(t))
    for i, yi in enumerate(y.ravel()):
        if abs(yi-np.pi) < 2e-14:
            result[i] = 1.
            continue
        if t <= 1:
            bracket = (0., LENGTH)
        elif yi < np.pi:
            bracket = (0., q)
        else:
            bracket = (LENGTH-q, LENGTH)
        xi = brentq(lambda z: z+t*np.sin(z)-yi, *bracket,
                    xtol=5e-15, rtol=1e-14)
        result[i] = 1+np.sin(xi)
    return result.reshape(x.shape)
