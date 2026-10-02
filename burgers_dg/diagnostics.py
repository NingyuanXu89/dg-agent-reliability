"""Error norms and conserved/dissipated quantities of a DG solution."""

import numpy as np

from .basis import gauss_legendre


def error_norms(solver, C, exact, n_quad=16):
    """L1, L2 and Linf errors of u_h against ``exact(x)``.

    Integrals use an ``n_quad``-point Gauss rule per cell; Linf is the maximum over
    those quadrature points (interior points, so the two-valued traces are avoided).
    """
    xi, w = gauss_legendre(n_quad)
    err = solver.evaluate(C, xi) - exact(solver.x_physical(xi))
    jac = 0.5 * solver.h
    return {
        "L1": jac * np.sum(np.abs(err) * w),
        "L2": np.sqrt(jac * np.sum(err ** 2 * w)),
        "Linf": np.abs(err).max(),
    }


def observed_orders(errors, h):
    """Experimental orders of convergence log(e_{i-1}/e_i) / log(h_{i-1}/h_i)."""
    errors, h = np.asarray(errors, float), np.asarray(h, float)
    return np.log(errors[:-1] / errors[1:]) / np.log(h[:-1] / h[1:])


def total_mass(solver, C):
    """int u_h dx (only the cell averages contribute)."""
    return solver.h * C[:, 0].sum()


def energy(solver, C):
    """int u_h^2 / 2 dx, using Legendre orthogonality: int_{I_j} P_k^2 dx = h / (2k + 1)."""
    return 0.5 * solver.h * np.sum(C ** 2 / (2 * solver.modes + 1))


def max_abs_gradient(solver, C, n_points=12):
    """max |du_h/dx| sampled inside every cell."""
    return np.abs(solver.gradient(C, np.linspace(-1.0, 1.0, n_points))).max()


def total_variation_of_means(C):
    avg = C[:, 0]
    return np.abs(np.roll(avg, -1) - avg).sum()
