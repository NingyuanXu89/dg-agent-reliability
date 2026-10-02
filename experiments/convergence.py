"""Convergence study for DG-Pp on periodic Burgers with smooth (pre-shock) sine data.

Runs P0..P3 on a sequence of uniform meshes up to t = 0.5 t_s (well before the
shock), compares with the exact characteristic solution, and writes

    results/convergence.csv          all errors, observed orders, conservation data
    results/convergence.md           Markdown table of the same
    results/convergence.png          log-log error plots with reference slopes
    results/error_profiles.png       pointwise error on the finest mesh
    results/convergence_summary.json fitted orders and time-step sensitivity check

Usage:  python -m experiments.convergence [--degrees 0 1 2 3] [--cells 10 20 ...]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from burgers_dg import BurgersDG, SineInitialData, error_norms, exact_solution, observed_orders
from burgers_dg.diagnostics import energy, total_mass

NORMS = ("L1", "L2", "Linf")


def dt_cap(solver, data, h_ref):
    """Time-step cap so the O(dt^3) RK3 error stays below the O(h^{p+1}) spatial error.

    For p <= 2 this is the usual CFL step; for p = 3 it shrinks like h^{4/3}.
    """
    p = solver.degree
    exponent = max(1.0, (p + 1) / 3.0)
    return solver.cfl * h_ref / ((2 * p + 1) * data.u_max) * (solver.h / h_ref) ** exponent


def run_case(n_cells, degree, data, t_final, flux, h_ref, dt_factor=1.0):
    solver = BurgersDG(n_cells, degree, domain=(0.0, data.length), flux=flux)
    C0 = solver.project(data.u0)
    start = time.perf_counter()
    C, steps = solver.advance(C0, 0.0, t_final, dt_max=dt_factor * dt_cap(solver, data, h_ref))
    runtime = time.perf_counter() - start
    errors = error_norms(solver, C, lambda x: exact_solution(x, t_final, data))
    return solver, C, {
        "degree": degree,
        "cells": n_cells,
        "h": solver.h,
        **errors,
        "mass_drift": abs(total_mass(solver, C) - total_mass(solver, C0)),
        # The exact solution conserves int u^2/2 while smooth; any loss is numerical dissipation.
        "energy_change": energy(solver, C) - energy(solver, C0),
        "steps": steps,
        "runtime_s": runtime,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--cells", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--time-fraction", type=float, default=0.5, help="final time as a fraction of t_s")
    parser.add_argument("--flux", choices=["godunov", "llf"], default="godunov")
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    if not 0.0 < args.time_fraction < 1.0:
        parser.error("--time-fraction must be in (0, 1) so the solution is still smooth")

    data = SineInitialData()
    t_final = args.time_fraction * data.shock_time
    cells = sorted(args.cells)
    h_ref = data.length / cells[0]
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"u0 = {data.mean} + {data.amplitude} sin(2 pi x) on [0, {data.length}), "
          f"t_s = {data.shock_time:.6f}, t_final = {t_final:.6f}, flux = {args.flux}")

    rows, finest = [], {}
    for p in args.degrees:
        for n in cells:
            solver, C, row = run_case(n, p, data, t_final, args.flux, h_ref)
            rows.append(row)
            finest[p] = (solver, C)
        block = [r for r in rows if r["degree"] == p]
        for norm in NORMS:
            orders = observed_orders([r[norm] for r in block], [r["h"] for r in block])
            block[0][f"order_{norm}"] = np.nan
            for r, q in zip(block[1:], orders):
                r[f"order_{norm}"] = q
        print(f"P{p}: L2 = " + " ".join(f"{r['L2']:.2e}" for r in block)
              + " | orders " + " ".join(f"{r['order_L2']:.2f}" for r in block[1:]))

    # Time-step sensitivity: halving dt on the finest mesh must not change the error noticeably.
    dt_check = {}
    for p in args.degrees:
        base = next(r for r in rows if r["degree"] == p and r["cells"] == cells[-1])
        _, _, half = run_case(cells[-1], p, data, t_final, args.flux, h_ref, dt_factor=0.5)
        dt_check[f"P{p}"] = {
            "L2": base["L2"], "L2_half_dt": half["L2"],
            "relative_change": abs(half["L2"] - base["L2"]) / base["L2"],
        }

    fieldnames = ["degree", "cells", "h"] + [k for n in NORMS for k in (n, f"order_{n}")] + [
        "mass_drift", "energy_change", "steps", "runtime_s"]
    with open(args.outdir / "convergence.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(args.outdir / "convergence.md", rows, data, t_final, args.flux, dt_check)
    plot_convergence(args.outdir / "convergence.png", rows, args.degrees, t_final, data)
    plot_error_profiles(args.outdir / "error_profiles.png", finest, data, t_final)

    summary = {
        "initial_data": {"mean": data.mean, "amplitude": data.amplitude, "length": data.length},
        "shock_time": data.shock_time,
        "t_final": t_final,
        "flux": args.flux,
        "cells": cells,
        "asymptotic_orders": {
            f"P{p}": {norm: next(r for r in rows if r["degree"] == p and r["cells"] == cells[-1])[f"order_{norm}"]
                      for norm in NORMS}
            for p in args.degrees
        },
        "expected_order": {f"P{p}": p + 1 for p in args.degrees},
        "max_mass_drift": max(r["mass_drift"] for r in rows),
        "max_energy_change": max(r["energy_change"] for r in rows),
        "dt_halving_check": dt_check,
    }
    (args.outdir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["asymptotic_orders"], indent=2))
    print(f"max mass drift = {summary['max_mass_drift']:.2e}; outputs in {args.outdir}/")
    return summary


def write_markdown(path, rows, data, t_final, flux, dt_check):
    lines = [
        "# DG convergence study: inviscid Burgers",
        "",
        f"u0(x) = {data.mean} + {data.amplitude} sin(2πx) on [0, {data.length}), periodic. "
        f"Shock time t_s = 1/(2π) = {data.shock_time:.6f}; errors at t = {t_final:.6f} "
        f"({t_final / data.shock_time:.2f} t_s) against the exact characteristic solution. "
        f"Flux: {flux}; time integration: SSP-RK3.",
        "",
        "| p | N | L1 | order | L2 | order | L∞ | order | mass drift | ΔE | steps |",
        "|---|---|----|-------|----|-------|----|-------|------------|----|-------|",
    ]
    fmt_order = lambda q: "–" if np.isnan(q) else f"{q:.2f}"
    for r in rows:
        lines.append(
            f"| {r['degree']} | {r['cells']} | {r['L1']:.3e} | {fmt_order(r['order_L1'])} "
            f"| {r['L2']:.3e} | {fmt_order(r['order_L2'])} | {r['Linf']:.3e} | {fmt_order(r['order_Linf'])} "
            f"| {r['mass_drift']:.1e} | {r['energy_change']:.2e} | {r['steps']} |")
    lines += ["", "ΔE = E(t) − E(0) with E = ∫u²/2 dx, which the exact smooth solution conserves.", "",
              "## Time-step sensitivity (finest mesh, dt halved)", "",
              "| p | L2 | L2 with dt/2 | relative change |", "|---|----|--------------|-----------------|"]
    for key, v in dt_check.items():
        lines.append(f"| {key[1:]} | {v['L2']:.4e} | {v['L2_half_dt']:.4e} | {v['relative_change']:.1e} |")
    path.write_text("\n".join(lines) + "\n")


def plot_convergence(path, rows, degrees, t_final, data):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(degrees)))
    for ax, norm in zip(axes, NORMS):
        for color, p in zip(colors, degrees):
            block = [r for r in rows if r["degree"] == p]
            h = np.array([r["h"] for r in block])
            e = np.array([r[norm] for r in block])
            ax.loglog(h, e, "o-", color=color, label=f"P{p}  (order {block[-1][f'order_{norm}']:.2f})")
            # Reference slope p+1 anchored at the finest mesh.
            ax.loglog(h, e[-1] * (h / h[-1]) ** (p + 1), ":", color=color, lw=1)
        ax.set_title(f"{norm} error at t = {t_final / data.shock_time:.2f} $t_s$")
        ax.set_xlabel("h")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("error")
    fig.suptitle("DG-Pp for Burgers: dotted lines show the expected slope p+1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_profiles(path, finest, data, t_final):
    fig, (ax_u, ax_e) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    xi = np.linspace(-1.0, 1.0, 9)
    x_fine = np.linspace(0.0, data.length, 1000)
    ax_u.plot(x_fine, data.u0(x_fine), color="0.6", lw=1, label="u0")
    ax_u.plot(x_fine, exact_solution(x_fine, t_final, data), "k", lw=1.5, label="exact")
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(finest)))
    for color, (p, (solver, C)) in zip(colors, finest.items()):
        x = solver.x_physical(xi).ravel()
        err = np.abs(solver.evaluate(C, xi).ravel() - exact_solution(x, t_final, data))
        ax_e.semilogy(x, np.maximum(err, 1e-17), color=color, lw=0.8, label=f"P{p}, N={solver.n_cells}")
    ax_u.set_ylabel("u")
    ax_u.legend()
    ax_u.set_title(f"Solution and pointwise error at t = {t_final:.4f}")
    ax_e.set_ylabel("|u_h − u|")
    ax_e.set_xlabel("x")
    ax_e.legend(fontsize=8, ncol=2)
    ax_e.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
