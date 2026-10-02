"""Error norms, observed convergence orders and conserved quantities."""

import numpy as np

from .basis import gauss_legendre, legendre_derivatives


def error_norms(solver, U, reference, nq=None, exclude=None):
    """L1, L2 and Linf norms of u_h - reference over the domain.

    reference(x) is evaluated at an nq-point Gauss rule in every cell (default
    p + 6 points, well above the polynomial degree). Linf is the max over those
    points plus the cell end points. exclude(x) -> bool array drops points
    (e.g. a window around a shock) from all three norms.
    """
    nq = nq or (solver.p + 6)
    xi, w = gauss_legendre(nq)
    xi_all = np.concatenate([xi, [-1.0, 1.0]])
    x, uh = solver.evaluate(U, xi_all)
    e = uh - reference(x)
    if exclude is not None:
        e = np.where(exclude(x), 0.0, e)
    eq = e[:, :nq]
    jac = 0.5 * solver.h
    L1 = np.sum(np.abs(eq) * w) * jac
    L2 = np.sqrt(np.sum(eq ** 2 * w) * jac)
    Linf = np.max(np.abs(e))
    return {"L1": L1, "L2": L2, "Linf": Linf}


def cell_l2_errors(solver, U, reference, nq=None):
    """Per-cell L2 error (array of length N)."""
    nq = nq or (solver.p + 6)
    xi, w = gauss_legendre(nq)
    x, uh = solver.evaluate(U, xi)
    e = uh - reference(x)
    return np.sqrt(np.sum(e ** 2 * w, axis=1) * 0.5 * solver.h)


def observed_orders(h, err):
    """Observed order between consecutive resolutions: log(e_k/e_{k+1}) / log(h_k/h_{k+1})."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def total_mass(solver, U):
    """int u_h dx = h * sum of cell averages."""
    return solver.h * np.sum(U[:, 0])


def total_energy(solver, U):
    """int u_h^2 / 2 dx, using orthogonality: int_cell P_j^2 dx = h / (2j + 1)."""
    j = np.arange(solver.p + 1)
    return 0.5 * solver.h * np.sum(U ** 2 / (2.0 * j + 1.0))


def max_gradient(solver, U, npts=8):
    """max |du_h/dx| sampled inside cells (does not include interface jumps)."""
    xi = np.linspace(-1.0, 1.0, npts)
    D = legendre_derivatives(solver.p, xi)
    return np.max(np.abs(U @ D.T)) * 2.0 / solver.h


def max_jump(solver, U):
    """max |u_h(x^+) - u_h(x^-)| over interfaces: a direct shock indicator."""
    right_end, left_end = solver.traces(U)
    return np.max(np.abs(np.roll(left_end, -1) - right_end))


def interface_jumps(solver, U):
    right_end, left_end = solver.traces(U)
    return np.roll(left_end, -1) - right_end


def dense_profile(solver, U, npts=12):
    """Flattened (x, u) for plotting; inserts NaN between cells to show jumps."""
    xi = np.linspace(-1.0, 1.0, npts)
    x, u = solver.evaluate(U, xi)
    nan = np.full((solver.N, 1), np.nan)
    return np.hstack([x, nan]).ravel(), np.hstack([u, nan]).ravel()

