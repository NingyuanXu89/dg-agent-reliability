"""Error norms and conserved/monitored quantities for DG solutions."""

import numpy as np

from .basis import gauss_legendre, legendre


def _periodic_distance(x, x0, length):
    d = np.mod(x - x0, length)
    return np.minimum(d, length - d)


def error_norms(solver, U, exact, n_quad=None, discontinuities=(), exclude=None):
    """L1, L2 and max-norm errors of the DG solution against ``exact(x)``.

    Integrals are computed cell by cell with Gauss quadrature. Cells that
    contain a point listed in ``discontinuities`` are split there, so a shock
    inside a cell is integrated accurately. ``exclude=(x0, width)`` drops every
    cell within ``width`` of ``x0`` (periodic distance) to measure the error
    away from a shock. The max norm is sampled at the quadrature points.
    """
    nq = n_quad or solver.degree + 8
    xi, w = gauss_legendre(nq)
    P, _ = legendre(solver.n_modes, xi)
    x = solver.x_at(xi)
    err = U @ P - exact(x)
    h = solver.h
    cell_l1 = 0.5 * h * (np.abs(err) @ w)
    cell_l2 = 0.5 * h * (err ** 2 @ w)
    cell_max = np.abs(err).max(axis=1)

    for xd in discontinuities:
        j = int(np.floor((np.mod(xd - solver.domain[0], solver.length)) / h)) % solver.n_cells
        a, b = solver.edges[j], solver.edges[j + 1]
        xd_local = solver.domain[0] + np.mod(xd - solver.domain[0], solver.length)
        l1 = l2 = 0.0
        for lo, hi in ((a, xd_local), (xd_local, b)):
            if hi - lo <= 0.0:
                continue
            xs = 0.5 * (lo + hi) + 0.5 * (hi - lo) * xi
            Ps, _ = legendre(solver.n_modes, 2.0 * (xs - solver.centers[j]) / h)
            e = U[j] @ Ps - exact(xs)
            l1 += 0.5 * (hi - lo) * (np.abs(e) @ w)
            l2 += 0.5 * (hi - lo) * (e ** 2 @ w)
        cell_l1[j], cell_l2[j] = l1, l2

    keep = np.ones(solver.n_cells, dtype=bool)
    if exclude is not None:
        x0, width = exclude
        # A cell is dropped if any part of it lies within ``width`` of x0.
        keep = _periodic_distance(solver.centers, x0, solver.length) > width + 0.5 * h
    return {
        "L1": float(cell_l1[keep].sum()),
        "L2": float(np.sqrt(cell_l2[keep].sum())),
        "Linf": float(cell_max[keep].max()),
    }


def mass(solver, U):
    """Integral of u_h over the domain (exactly conserved by the scheme)."""
    return float(solver.h * U[:, 0].sum())


def energy(solver, U):
    """Integral of u_h^2 / 2 (non-increasing for the entropy solution)."""
    norms = 2.0 / (2.0 * np.arange(solver.n_modes) + 1.0)
    return float(0.25 * solver.h * (U ** 2 @ norms).sum())


def total_variation(U):
    """Total variation of the cell averages."""
    mean = U[:, 0]
    return float(np.abs(np.roll(mean, -1) - mean).sum())


def extrema(solver, U, n_points=None):
    """(min, max) of u_h sampled at Gauss points and cell edges."""
    xi, _ = gauss_legendre(n_points or solver.degree + 4)
    vals = solver.evaluate(U, np.concatenate(([-1.0], xi, [1.0])))
    return float(vals.min()), float(vals.max())


def steepest_slope(solver, U):
    """Most negative du_h/dx inside cells (Gauss points)."""
    xi, _ = gauss_legendre(solver.degree + 4)
    return float(solver.evaluate_slope(U, xi).min())


def interface_jumps(solver, U):
    """|u_h(x_{j+1/2}^+) - u_h(x_{j+1/2}^-)| at every interface."""
    left, right = solver.traces(U)
    return np.abs(np.roll(left, -1) - right)


def smoothness_indicator(solver, U):
    """Persson-Peraire indicator: fraction of the cell's L2 energy in mode p.

    S_j = ||u_h - Pi_{p-1} u_h||^2 / ||u_h||^2 on cell j. For smooth data it
    decays like h^(2p); a discontinuity inside the cell gives S = O(h^0).
    """
    if solver.degree == 0:
        return np.zeros(solver.n_cells)
    l = np.arange(solver.n_modes)
    e = U ** 2 * (2.0 / (2.0 * l + 1.0))
    total = e.sum(axis=1)
    return np.where(total > 1e-300, e[:, -1] / np.maximum(total, 1e-300), 0.0)


def observed_orders(h, err):
    """log2-style observed convergence orders between consecutive refinements."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def piecewise_curve(solver, U, n_per_cell=12):
    """(x, u) samples of u_h with NaN breaks between cells, for plotting."""
    xi = np.linspace(-1.0, 1.0, n_per_cell)
    x = solver.x_at(xi)
    u = solver.evaluate(U, xi)
    pad = np.full((solver.n_cells, 1), np.nan)
    return np.hstack([x, pad]).ravel(), np.hstack([u, pad]).ravel()
