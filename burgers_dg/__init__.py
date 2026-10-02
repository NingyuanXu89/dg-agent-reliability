"""Discontinuous Galerkin solver for the periodic inviscid Burgers equation."""

from .solver import BurgersDG, godunov_flux, llf_flux
from .exact import SineInitialData, exact_solution, characteristic_foot, hopf_lax_solution
from .diagnostics import error_norms, observed_orders, total_mass, energy

__all__ = [
    "BurgersDG", "godunov_flux", "llf_flux",
    "SineInitialData", "exact_solution", "characteristic_foot", "hopf_lax_solution",
    "error_norms", "observed_orders", "total_mass", "energy",
]
