"""From-scratch periodic modal DG for inviscid Burgers."""
from .core import Config, DG, Result, simulate
from .reference import exact_solution

__all__ = ["Config", "DG", "Result", "simulate", "exact_solution"]
