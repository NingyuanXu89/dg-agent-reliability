"""Mesh-convergence study for the DG Burgers solver.

Three regimes of u0 = 0.5 + sin(x) (breaking time t_b = 1):
  1. smooth,        t = 0.5 : unlimited DG, p = 0..3, Godunov and LLF fluxes.
                              Expected order p + 1.
  2. near breaking, t = 0.9 : max|u_x| = 1/(1 - t) = 10, so the asymptotic
                              rate appears only once h resolves the steep front.
  3. after shock,   t = 2.0 : TVB-limited DG, p = 1, 2. The global L1 error is
                              O(h) because the shock is smeared over O(1) cells;
                              away from the shock the order is p + 1 again.
A time-step check (halving the CFL number on the finest meshes) confirms that
the temporal error does not pollute the spatial rates.

Outputs (in results/): convergence.csv, convergence.md, convergence_smooth.png,
convergence_regimes.png
"""

import argparse
import csv
import os
import time

import _common  # noqa: F401  (sets sys.path)
from _common import A, C, RESULTS, TVB_M

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgburgers import DGBurgers, exact_solution, initial_condition, shock_position
from dgburgers.diagnostics import error_norms, observed_orders

EXCLUDE_WIDTH = 0.5   # half-width of the window around the shock left out of "away" norms


def run_case(n_cells, degree, t, flux="godunov", limiter=None, cfl=0.5, integrator="ssprk104"):
    s = DGBurgers(n_cells, degree, flux=flux, limiter=limiter, tvb_M=TVB_M)
    U0 = s.project(lambda x: initial_condition(x, C, A))
    tic = time.perf_counter()
    out = s.solve(U0, t, cfl=cfl, integrator=integrator)
    wall = time.perf_counter() - tic
    exact = lambda x: exact_solution(x, t, C, A)
    shocked = t > 1.0 / A
    row = {
        "t": t, "flux": flux, "limiter": limiter or "none", "p": degree, "N": n_cells,
        "h": s.h, "cfl": cfl, "steps": out["n_steps"], "wall_s": wall,
    }
    disc = (shock_position(t, C),) if shocked else ()
    for k, v in error_norms(s, out["U"], exact, discontinuities=disc).items():
        row[k] = v
    if shocked:
        away = error_norms(s, out["U"], exact, exclude=(shock_position(t, C), EXCLUDE_WIDTH))
        for k, v in away.items():
            row[f"{k}_away"] = v
    return row


def study(meshes, degrees, t, **kw):
    rows = []
    for p in degrees:
        for N in meshes(p):
            rows.append(run_case(N, p, t, **kw))
    return rows


def with_orders(rows, key):
    """Attach observed orders for ``key`` within each (t, flux, limiter, p) group."""
    groups = {}
    for r in rows:
        groups.setdefault((r["t"], r["flux"], r["limiter"], r["p"], r["cfl"]), []).append(r)
    for g in groups.values():
        orders = observed_orders([r["h"] for r in g], [r[key] for r in g])
        g[0][f"order_{key}"] = np.nan
        for r, o in zip(g[1:], orders):
            r[f"order_{key}"] = o
    return rows


def markdown_table(rows, keys):
    head = ["p", "N"] + [k for k in keys]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [str(r["p"]), str(r["N"])]
        for k in keys:
            v = r.get(k, np.nan)
            if k.startswith("order"):
                cells.append("—" if not np.isfinite(v) else f"{v:.2f}")
            else:
                cells.append(f"{v:.3e}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="coarser meshes only (for a fast check)")
    args = ap.parse_args()
    top = 160 if args.quick else 640

    def meshes(p):
        n = [10 * 2 ** k for k in range(20) if 10 * 2 ** k <= top]
        return n + [2 * top] if p == 0 and not args.quick else n

    norms = ["L1", "L2", "Linf"]
    smooth, fluxes = [], ["godunov", "llf"]
    for flux in fluxes:
        smooth += study(meshes, [0, 1, 2, 3], 0.5, flux=flux)
    near = study(meshes, [1, 2, 3], 0.9)
    shock = study(meshes, [1, 2], 2.0, limiter="minmod")
    # Temporal-error check: finest meshes at half the CFL number.
    dtcheck = [run_case(meshes(p)[-1], p, 0.5, cfl=0.25) for p in (1, 2, 3)]

    for k in norms:
        with_orders(smooth, k)
        with_orders(near, k)
        with_orders(shock, k)
        with_orders(shock, f"{k}_away")

    # ------------------------------------------------------------- write CSV
    all_rows = smooth + near + shock + dtcheck
    order = (["t", "flux", "limiter", "p", "N", "h", "cfl", "steps", "wall_s"]
             + [f"{pre}{k}{suf}" for pre in ("", "order_") for suf in ("", "_away") for k in norms])
    fields = [k for k in order if any(k in r for r in all_rows)]
    with open(os.path.join(RESULTS, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(all_rows)

    # -------------------------------------------------------- write Markdown
    md = ["# DG Burgers convergence study", "",
          f"Problem: u_t + (u²/2)_x = 0 on [0, 2π), periodic, u₀ = {C} + {A}·sin x, "
          f"breaking time t_b = {1 / A:g}. Errors against the exact entropy solution; "
          "time stepping SSPRK(10,4), CFL 0.5 (dt = 0.5 h / ((2p+1) max|u|)).", ""]
    for flux in fluxes:
        rows = [r for r in smooth if r["flux"] == flux]
        md += [f"## Smooth regime, t = 0.5, {flux} flux, no limiter", "",
               markdown_table(rows, ["L1", "order_L1", "L2", "order_L2", "Linf", "order_Linf"]), ""]
    md += ["## Near breaking, t = 0.9 (max|u_x| = 10), Godunov, no limiter", "",
           markdown_table(near, ["L1", "order_L1", "L2", "order_L2", "Linf", "order_Linf"]), ""]
    md += [f"## After the shock, t = 2.0, Godunov, TVB minmod limiter (M = {TVB_M:g})", "",
           f"`*_away` excludes every cell within {EXCLUDE_WIDTH} of the shock at x_s = π + 0.5t.", "",
           markdown_table(shock, ["L1", "order_L1", "L1_away", "order_L1_away",
                                  "Linf_away", "order_Linf_away"]), "",
           "The global L1 order oscillates because the error depends on where the shock sits "
           "inside its cell; a least-squares fit over all meshes gives:", ""]
    for p in (1, 2):
        g = [r for r in shock if r["p"] == p]
        lh = np.log([r["h"] for r in g])
        fit = lambda k: np.polyfit(lh, np.log([r[k] for r in g]), 1)[0]
        md.append(f"- p = {p}: fitted order L1 = {fit('L1'):.2f}, L1 away from shock = {fit('L1_away'):.2f}")
    md += [""]
    md += ["## Time-step check, t = 0.5, finest meshes", "",
           "| p | N | L2 (CFL 0.5) | L2 (CFL 0.25) | relative change |", "|---|---|---|---|---|"]
    for r in dtcheck:
        ref = next(s for s in smooth if s["flux"] == "godunov" and s["p"] == r["p"] and s["N"] == r["N"])
        md.append(f"| {r['p']} | {r['N']} | {ref['L2']:.6e} | {r['L2']:.6e} | "
                  f"{abs(r['L2'] - ref['L2']) / ref['L2']:.1e} |")
    with open(os.path.join(RESULTS, "convergence.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    # ----------------------------------------------------------------- plots
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, 4))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, k in zip(axes, norms):
        for p in range(4):
            for flux, ls, mk in (("godunov", "-", "o"), ("llf", "--", "s")):
                g = [r for r in smooth if r["flux"] == flux and r["p"] == p]
                h = np.array([r["h"] for r in g])
                ax.loglog(h, [r[k] for r in g], ls, marker=mk, ms=4, color=colors[p],
                          label=f"p={p} {flux}" if k == "L1" else None)
            # reference slope p + 1 anchored at the coarsest Godunov point
            g = [r for r in smooth if r["flux"] == "godunov" and r["p"] == p]
            h = np.array([r["h"] for r in g])
            ax.loglog(h, 0.5 * g[0][k] * (h / h[0]) ** (p + 1), ":", color="0.5", lw=0.9)
        ax.set_title(f"{k} error, t = 0.5 (smooth)")
        ax.set_xlabel("h")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("error")
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].text(0.98, 0.03, "dotted: slope p+1", transform=axes[-1].transAxes,
                  ha="right", fontsize=8, color="0.4")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "convergence_smooth.png"), dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for p in (1, 2, 3):
        for t, ls, mk, rows in ((0.5, "-", "o", [r for r in smooth if r["flux"] == "godunov"]),
                                (0.9, "--", "^", near)):
            g = [r for r in rows if r["p"] == p]
            a1.loglog([r["h"] for r in g], [r["L2"] for r in g], ls, marker=mk, ms=4,
                      color=colors[p], label=f"p={p}, t={t}")
    a1.set_title("L2 error: smooth (t=0.5) vs near breaking (t=0.9)")
    a1.set_xlabel("h")
    a1.set_ylabel("L2 error")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=7, ncol=2)
    for p in (1, 2):
        g = [r for r in shock if r["p"] == p]
        h = np.array([r["h"] for r in g])
        a2.loglog(h, [r["L1"] for r in g], "-o", ms=4, color=colors[p], label=f"p={p} global")
        a2.loglog(h, [r["L1_away"] for r in g], "--s", ms=4, color=colors[p], label=f"p={p} away from shock")
        a2.loglog(h, 0.5 * g[0]["L1_away"] * (h / h[0]) ** (p + 1), ":", color="0.5", lw=0.9)
    a2.loglog(h, 0.7 * shock[0]["L1"] * (h / h[0]), "-.", color="k", lw=0.9, label="slope 1")
    a2.set_title("L1 error after the shock, t = 2 (TVB limiter)")
    a2.set_xlabel("h")
    a2.grid(True, which="both", alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "convergence_regimes.png"), dpi=150)
    plt.close(fig)

    print("\n".join(md))


if __name__ == "__main__":
    main()
