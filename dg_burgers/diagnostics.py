"""Quadrature errors, temporal comparisons, and reproducible output files."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from .core import DG
from .reference import exact_solution, shock_position


def integration_grid(dg, t, points=16):
    """Split the cell containing the physical shock before error quadrature."""
    z, w = leggauss(max(points, 2*dg.p+4))
    xs, ws, ids = [], [], []
    shock = shock_position(t, dg.config.origin) if t > 1 else None
    for cell, (left, right) in enumerate(zip(dg.edges[:-1], dg.edges[1:])):
        cuts = [left, right]
        if shock is not None and left+1e-14 < shock < right-1e-14:
            cuts.insert(1, shock)
        for a, b in zip(cuts[:-1], cuts[1:]):
            xs.append((a+b)/2+(b-a)/2*z)
            ws.append((b-a)/2*w)
            ids.append(np.full(len(z), cell, dtype=int))
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(ids)


def errors(dg, coefficients, t):
    x, w, cells = integration_grid(dg, t)
    error = dg.evaluate(coefficients, x, cells)-exact_solution(x, t)
    return dict(l1=float(np.sum(w*np.abs(error))),
                l2=float(np.sqrt(np.sum(w*error**2))),
                linf_sampled=float(np.max(np.abs(error))))


def coefficient_distance(dg, a, b, t, norm="l2"):
    if norm == "l2":
        return float(np.sqrt(np.sum((a-b)**2*dg.mass)))
    x, w, cells = integration_grid(dg, t)
    return float(np.sum(w*np.abs(dg.evaluate(a-b, x, cells))))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_result(result, directory, with_errors=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dg = DG(result.config)
    rows = []
    for t, c, d in zip(result.times, result.coefficients, result.diagnostics):
        row = dict(d)
        if with_errors:
            row.update(errors(dg, c, t))
        rows.append(row)
    write_csv(directory/"diagnostics.csv", rows)
    np.savez_compressed(directory/"snapshots.npz", times=result.times,
                        coefficients=result.coefficients, edges=dg.edges)
    metadata = dict(config=asdict(result.config), steps=result.steps, runtime_seconds=result.runtime,
                    reference="periodic entropy solution for 1+sin(x)",
                    linf_definition="maximum error at split-cell quadrature nodes; not a true supremum",
                    limiter_fraction_definition="flagged cell stages / (3 * cells * interval steps)")
    (directory/"config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return rows
