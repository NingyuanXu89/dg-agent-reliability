"""Convergence study and error diagnostics for DG on periodic Burgers.

1. Smooth h-refinement (no limiter) at t = 0.5 t_s for p = 1, 2, 3:
   L1 / L2 / Linf errors against the characteristic solution, observed orders,
   and a temporal-error check (finest mesh rerun with dt/4).
2. Pointwise error profiles at t = 0.5 t_s.
3. Error growth in time as t -> t_s on fixed meshes.
4. Post-shock L1 convergence (TVB limiter) at t = 1.3 t_s.
5. Conservation: mass drift and energy vs the exact (entropy-dissipating) energy.
"""

import csv
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from common import A, C, T_SHOCK, U_MAX, out, u0
from burgers_dg import DGBurgers, TVBLimiter, exact_solution, gauss_legendre, shock_position

T_SMOOTH = 0.5 * T_SHOCK
T_POST = 1.3 * T_SHOCK
DEGREES = (1, 2, 3)
MESHES = (8, 16, 32, 64, 128, 256)
POST_DEGREES = (1, 2)
POST_MESHES = (32, 64, 128, 256, 512)
CFL = 0.3


def exact_at(t):
    return lambda x: exact_solution(x, t, C, A)


def dt_cap(p, h, h0):
    """SSP-RK3 is third order.  For p = 3 (spatial order 4) shrink dt like
    h^{4/3} so the O(dt^3) temporal error stays below the O(h^4) spatial error."""
    return CFL * h0 / ((2 * p + 1) * U_MAX) * (h / h0) ** max(1.0, (p + 1) / 3.0)


def rates(errs):
    errs = np.asarray(errs)
    return np.concatenate([[np.nan], np.log2(errs[:-1] / errs[1:])])


def exact_energy(t, n_cells=256, n_q=12):
    """(1/2) int u^2 for the exact solution, integrating over a period that starts
    at the shock (or the steepest point, before t_s), so every quadrature cell
    sees a smooth integrand."""
    a = shock_position(t, C)
    r, w = gauss_legendre(n_q)
    edges = a + np.linspace(0.0, 2 * np.pi, n_cells + 1)
    xm, hm = 0.5 * (edges[1:] + edges[:-1]), np.diff(edges)
    xq = xm[:, None] + 0.5 * hm[:, None] * r      # Gauss nodes never touch the shock
    return 0.5 * np.sum(0.5 * hm[:, None] * w * exact_solution(xq, t, C, A) ** 2)


def smooth_study():
    rows, finest = [], {}
    for p in DEGREES:
        h0 = 2 * np.pi / MESHES[0]
        errs = []
        for K in MESHES:
            s = DGBurgers(K, p, cfl=CFL)
            cap = dt_cap(p, s.h, h0)
            nsteps = [0]
            t0 = time.perf_counter()
            c, _ = s.run(s.project(u0), T_SMOOTH, dt_max=cap,
                         on_step=lambda t, c: nsteps.__setitem__(0, nsteps[0] + 1))
            wall = time.perf_counter() - t0
            errs.append(s.errors(c, exact_at(T_SMOOTH)))
            rows.append(dict(p=p, K=K, h=s.h, steps=nsteps[0], wall_s=wall,
                             L1=errs[-1][0], L2=errs[-1][1], Linf=errs[-1][2]))
            if K == MESHES[-1]:
                finest[p] = (s, c, cap)
        e = np.array(errs)
        for j, name in enumerate(("L1", "L2", "Linf")):
            for row, r in zip(rows[-len(MESHES):], rates(e[:, j])):
                row[f"rate_{name}"] = r

    # temporal-error check on the finest mesh: rerun with dt/4
    dt_check = {}
    for p, (s, c, cap) in finest.items():
        s4 = DGBurgers(s.K, p, cfl=CFL / 4)
        c4, _ = s4.run(s4.project(u0), T_SMOOTH, dt_max=cap / 4)
        e_ref = s.errors(c, exact_at(T_SMOOTH))[1]
        e_4 = s4.errors(c4, exact_at(T_SMOOTH))[1]
        dt_check[p] = abs(e_ref - e_4) / e_4
    return rows, dt_check


def write_smooth_tables(rows, dt_check):
    keys = ["p", "K", "h", "steps", "wall_s", "L1", "rate_L1", "L2", "rate_L2", "Linf", "rate_Linf"]
    with open(out("convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    def fmt_rate(v):
        return "—" if np.isnan(v) else f"{v:.2f}"

    lines = [f"Smooth convergence at t = {T_SMOOTH:g} (t_s = {T_SHOCK:g}), "
             f"u0 = {C} + {A} sin(x), Godunov flux, SSP-RK3, no limiter.", "",
             "| p | K | L1 error | rate | L2 error | rate | L∞ error | rate |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['p']} | {r['K']} | {r['L1']:.3e} | {fmt_rate(r['rate_L1'])} | "
                     f"{r['L2']:.3e} | {fmt_rate(r['rate_L2'])} | {r['Linf']:.3e} | {fmt_rate(r['rate_Linf'])} |")
    lines += ["", "Temporal-error check (finest mesh, dt/4): relative change in L2 error  "]
    lines += [f"p = {p}: {v:.2e}  " for p, v in dt_check.items()]
    return lines


def plot_convergence(rows):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    colors = {1: "C0", 2: "C1", 3: "C2"}
    for ax, name, label in zip(axes, ("L1", "L2", "Linf"), ("$L^1$", "$L^2$", r"$L^\infty$")):
        for p in DEGREES:
            sub = [r for r in rows if r["p"] == p]
            h = np.array([r["h"] for r in sub])
            e = np.array([r[name] for r in sub])
            ax.loglog(h, e, "o-", color=colors[p], label=f"p = {p}")
            ref = e[-1] * (h / h[-1]) ** (p + 1)
            ax.loglog(h, ref, ":", color=colors[p], alpha=0.7)
            ax.annotate(f"slope {p + 1}", (h[1], ref[1]), color=colors[p], fontsize=8,
                        xytext=(4, -10), textcoords="offset points")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("cell width h")
        ax.set_title(f"{label} error at t = {T_SMOOTH:g}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("error")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out("convergence.png"), dpi=130)
    plt.close(fig)


def plot_error_profiles(K=64):
    fig, ax = plt.subplots(figsize=(8, 4))
    for p in DEGREES:
        s = DGBurgers(K, p, cfl=CFL)
        c, _ = s.run(s.project(u0), T_SMOOTH, dt_max=dt_cap(p, s.h, 2 * np.pi / MESHES[0]))
        r = np.linspace(-1, 1, 24)
        x = s.x_at(r)
        e = np.abs(s.evaluate(c, r) - exact_at(T_SMOOTH)(x))
        pad = np.full((K, 1), np.nan)
        ax.semilogy(np.hstack([x, pad]).ravel(), np.hstack([e, pad]).ravel(), lw=0.9, label=f"p = {p}")
    ax.axvline(shock_position(T_SMOOTH, C), color="k", ls="--", lw=0.8,
               label="steepest point $\\pi + ct$")
    ax.set_xlabel("x")
    ax.set_ylabel("|u_h - u|")
    ax.set_title(f"Pointwise error, K = {K}, t = {T_SMOOTH:g}")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("error_profile.png"), dpi=130)
    plt.close(fig)


def error_vs_time(p=2, meshes=(32, 64, 128)):
    times = np.linspace(0.0, 0.98 * T_SHOCK, 50)
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    for K in meshes:
        s = DGBurgers(K, p, cfl=CFL)
        _, snaps = s.run(s.project(u0), times[-1], output_times=times)
        e = [s.errors(snaps[t], exact_at(t))[1] for t in times]
        ax.semilogy(times, e, label=f"K = {K}")
    ax.set_ylabel("$L^2$ error")
    ax.set_title(f"Error growth as the gradient steepens (p = {p}, no limiter)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    tt = np.linspace(0.0, 0.98 * T_SHOCK, 200)
    axg.semilogy(tt, A / (1 - A * tt), "k--", lw=1)
    axg.set_ylabel("exact max |u_x|\n= a/(1 − a t)")
    axg.set_xlabel("t")
    axg.grid(True, alpha=0.3)
    for a_ in (ax, axg):
        a_.axvline(T_SHOCK, color="r", lw=0.8)
    fig.tight_layout()
    fig.savefig(out("error_vs_time.png"), dpi=130)
    plt.close(fig)


def post_shock_study():
    rows = []
    for p in POST_DEGREES:
        errs = []
        for K in POST_MESHES:
            s = DGBurgers(K, p, limiter=TVBLimiter(M=1.0), cfl=CFL)
            c, _ = s.run(s.project(u0), T_POST)
            errs.append(s.errors(c, exact_at(T_POST))[0])
            rows.append(dict(p=p, K=K, L1=errs[-1]))
        for row, r in zip(rows[-len(POST_MESHES):], rates(errs)):
            row["rate_L1"] = r
    lines = ["", f"Post-shock L1 convergence at t = {T_POST:g} (TVB limiter, M = 1):", "",
             "| p | K | L1 error | rate |", "|---|---|---|---|"]
    for r in rows:
        rate = "—" if np.isnan(r["rate_L1"]) else f"{r['rate_L1']:.2f}"
        lines.append(f"| {r['p']} | {r['K']} | {r['L1']:.3e} | {rate} |")
    return rows, lines


def conservation(p=2, K=128):
    s = DGBurgers(K, p, limiter=TVBLimiter(M=1.0), cfl=CFL)
    c = s.project(u0)
    m0, E0 = s.mass(c), s.energy(c)
    hist = {"t": [0.0], "mass": [m0], "energy": [E0]}

    def rec(t, c):
        hist["t"].append(t)
        hist["mass"].append(s.mass(c))
        hist["energy"].append(s.energy(c))

    s.run(c, T_POST, on_step=rec)
    t = np.array(hist["t"])
    drift = np.abs(np.array(hist["mass"]) - m0) / abs(m0)
    tE = np.linspace(0, T_POST, 80)
    E_exact = np.array([exact_energy(ti) for ti in tE])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.semilogy(t[1:], np.maximum(drift[1:], 1e-17), lw=0.8)
    a1.axvline(T_SHOCK, color="r", lw=0.8)
    a1.set_xlabel("t")
    a1.set_ylabel("|M(t) - M(0)| / M(0)")
    a1.set_title(f"Mass conservation (p = {p}, K = {K}, limited)")
    a1.grid(True, alpha=0.3)
    a2.plot(t, hist["energy"], label="DG  $\\frac{1}{2}\\int u_h^2$")
    a2.plot(tE, E_exact, "k--", lw=1, label="exact entropy solution")
    a2.axvline(T_SHOCK, color="r", lw=0.8, label="$t_s$")
    a2.set_xlabel("t")
    a2.set_ylabel("energy")
    a2.set_title("Energy: conserved while smooth, dissipated by the shock")
    a2.legend(fontsize=8)
    a2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("conservation.png"), dpi=130)
    plt.close(fig)

    i_ts = np.searchsorted(t, T_SHOCK)
    return {
        "max_rel_mass_drift": float(drift.max()),
        "exact_energy_t0": float(E_exact[0]),
        "dg_rel_energy_loss_before_ts": float((E0 - hist["energy"][i_ts]) / E0),
        "dg_energy_t_post": float(hist["energy"][-1]),
        "exact_energy_t_post": float(exact_energy(T_POST)),
    }


def main():
    t0 = time.perf_counter()
    rows, dt_check = smooth_study()
    md = ["# Convergence study", ""] + write_smooth_tables(rows, dt_check)
    plot_convergence(rows)
    plot_error_profiles()
    error_vs_time()
    post_rows, post_md = post_shock_study()
    md += post_md
    cons = conservation()
    md += ["", "Conservation (p = 2, K = 128, limited, to t = 1.3):  ",
           f"max relative mass drift = {cons['max_rel_mass_drift']:.2e}  ",
           f"DG energy loss before t_s = {cons['dg_rel_energy_loss_before_ts']:.2e} (exact: 0)  ",
           f"energy at t = {T_POST:g}: DG {cons['dg_energy_t_post']:.6f}, exact {cons['exact_energy_t_post']:.6f}"]
    with open(out("convergence.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    finest = {p: next(r for r in rows if r["p"] == p and r["K"] == MESHES[-1]) for p in DEGREES}
    summary = {
        "smooth_t": T_SMOOTH,
        "smooth_final_rates_L2": {p: finest[p]["rate_L2"] for p in DEGREES},
        "smooth_finest_L2": {p: finest[p]["L2"] for p in DEGREES},
        "dt_quarter_rel_change_L2": dt_check,
        "post_shock_final_rate_L1": {p: [r for r in post_rows if r["p"] == p][-1]["rate_L1"] for p in POST_DEGREES},
        **cons,
        "wall_time_s": time.perf_counter() - t0,
    }
    with open(out("convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    return summary


if __name__ == "__main__":
    main()
