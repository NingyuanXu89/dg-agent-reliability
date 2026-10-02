"""Discontinuous Galerkin solver for the 1-D inviscid Burgers equation."""

from .basis import gauss_legendre, legendre
from .exact import breaking_time, exact_solution, initial_condition, shock_position
from .solver import DGBurgers, godunov_flux, llf_flux

__all__ = [
    "DGBurgers",
    "breaking_time",
    "exact_solution",
    "gauss_legendre",
    "godunov_flux",
    "initial_condition",
    "legendre",
    "llf_flux",
    "shock_position",
]
