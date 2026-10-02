"""From-scratch discontinuous Galerkin solver for the periodic inviscid Burgers equation."""

from .basis import LegendreBasis, gauss_legendre, legendre
from .exact import (exact_gradient, exact_max_abs_gradient, exact_solution,
                    shock_position, shock_time)
from .flux import godunov, rusanov
from .limiter import TVBLimiter, minmod
from .solver import DGBurgers

__all__ = [
    "DGBurgers", "LegendreBasis", "TVBLimiter", "exact_gradient",
    "exact_max_abs_gradient", "exact_solution", "gauss_legendre", "godunov",
    "legendre", "minmod", "rusanov", "shock_position", "shock_time",
]
