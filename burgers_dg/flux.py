"""Numerical fluxes for the Burgers flux f(u) = u^2 / 2."""

import numpy as np


def burgers_flux(u):
    return 0.5 * u * u


def godunov(uL, uR):
    """Exact Godunov flux for the convex flux u^2/2.

    F = min_{u in [uL, uR]} f(u)  if uL <= uR   (rarefaction, 0 at a sonic point)
        max_{u in [uR, uL]} f(u)  if uL >  uR   (shock)
    """
    fL = burgers_flux(uL)
    fR = burgers_flux(uR)
    rarefaction = np.where((uL <= 0.0) & (uR >= 0.0), 0.0, np.minimum(fL, fR))
    shock = np.maximum(fL, fR)
    return np.where(uL <= uR, rarefaction, shock)


def rusanov(uL, uR):
    """Local Lax-Friedrichs (Rusanov) flux."""
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (burgers_flux(uL) + burgers_flux(uR)) - 0.5 * alpha * (uR - uL)


FLUXES = {"godunov": godunov, "rusanov": rusanov}
