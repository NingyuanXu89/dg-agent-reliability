"""Error diagnosis over time for DG on Burgers, through shock formation.

Runs p = 2 Godunov DG on u0 = 0.5 + sin(x) from t = 0 to t = 3 (t_b = 1), both
unlimited and with the TVB minmod limiter, and tracks:
  * L1 error vs the exact entropy solution (global and away from the shock),
  * mass drift (should be round-off),
  * energy int u^2/2 vs the exact energy (conserved until t_b, then dissipated),
  * min/max of u_h vs the exact bounds (overshoots / Gibbs oscillations),
  * steepest slope vs the exact -A/(1 - A t) (resolution limit near t_b),
  * shock-sensor activity (fraction of cells the limiter flags, max smoothness indicator).
It also plots pointwise error profiles at selected times and the error at
t = 2 as a function of distance from the shock.

Outputs (in results/): diagnostics_history.png, diagnostics_profiles.png,
diagnostics_summary.json
"""

import argparse
import json
import os

import _common  # noqa: F401
from _common import A, C, RESULTS, TVB_M

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dgburgers import DGBurgers, exact_solution, initial_condition, shock_position
from dgburgers.diagnostics import (error_norms, energy, extrema, interface_jumps, mass,
                                   piecewise_curve, smoothness_indicator, steepest_slope)
from dgburgers.exact import breaking_time, exact_energy


def history(n_cells, degree, t_end, limiter, n_samples):
    s = DGBurgers(n_cells, degree, limiter=limiter, tvb_M=TVB_M)
    U0 = s.project(lambda x: initial_condition(x, C, A))
    times = np.linspace(0.0, t_end, n_samples)
    wanted = {float(t) for t in times}
    flags = {}

    def grab_flags(t, U):
        # For a limited run the stored state is already limited (a fixed point
        # of the limiter), so record which cells the last stage actually limited.
        if t in wanted:
            flags[t] = s.last_flagged.copy() if limiter else s.troubled_cells(U)

    out = s.solve(U0, t_end, cfl=0.5, integrator="ssprk3", save_times=times, callback=grab_flags)
    out["flags"] = flags
    tb = breaking_time(A)
    rec = {k: [] for k in ("t", "L1", "L1_away", "mass", "energy", "energy_exact", "umin", "umax",
                           "umin_exact", "umax_exact", "slope", "slope_exact", "flagged",
                           "sensor", "max_jump")}
    xs_fine = np.linspace(0, 2 * np.pi, 20001)
    for t in times:
        U = out["snapshots"][float(t)]
        ex = lambda x, t=t: exact_solution(x, t, C, A)
        shocked = t > tb
        xs = shock_position(t, C)
        rec["t"].append(t)
        rec["L1"].append(error_norms(s, U, ex, discontinuities=(xs,) if shocked else ())["L1"])
        rec["L1_away"].append(error_norms(s, U, ex, exclude=(xs, 0.5))["L1"] if shocked else np.nan)
        rec["mass"].append(mass(s, U))
        rec["energy"].append(energy(s, U))
        rec["energy_exact"].append(exact_energy(t, C, A))
        lo, hi = extrema(s, U)
        rec["umin"].append(lo)
        rec["umax"].append(hi)
        ue = ex(xs_fine)
        rec["umin_exact"].append(ue.min())
        rec["umax_exact"].append(ue.max())
        rec["slope"].append(steepest_slope(s, U))
        rec["slope_exact"].append(-A / (1.0 - A * t) if t < tb else -np.inf)
        rec["flagged"].append(flags[float(t)].mean())
        rec["sensor"].append(smoothness_indicator(s, U).max())
        rec["max_jump"].append(interface_jumps(s, U).max())
    return s, {k: np.array(v) for k, v in rec.items()}, out


def first_time(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, default=64)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--t-end", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=301)
    args = ap.parse_args()
    tb = breaking_time(A)

    runs = {}
    for name, lim in (("unlimited", None), ("TVB limited", "minmod")):
        runs[name] = history(args.cells, args.degree, args.t_end, lim, args.samples)
    style = {"unlimited": dict(color="tab:orange"), "TVB limited": dict(color="tab:blue")}

    # ----------------------------------------------------------- time histories
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()
    for name, (s, r, _) in runs.items():
        st = style[name]
        ax[0].semilogy(r["t"], r["L1"], label=f"{name}: global", **st)
        ax[0].semilogy(r["t"], r["L1_away"], "--", label=f"{name}: |x - x_s| > 0.5", **st)
        drift = np.abs(r["mass"] - r["mass"][0]) / abs(r["mass"][0])
        ax[1].semilogy(r["t"], np.maximum(drift, 1e-17), label=name, **st)
        ax[2].plot(r["t"], r["energy"], label=name, **st)
        ax[3].plot(r["t"], r["umax"], label=f"{name}: max", **st)
        ax[3].plot(r["t"], r["umin"], ":", label=f"{name}: min", **st)
        ax[4].plot(r["t"], r["slope"], label=name, **st)
        ax[5].semilogy(r["t"], np.maximum(r["sensor"], 1e-16), label=f"{name}: max smoothness indicator", **st)
        if name == "TVB limited":
            ax5b = ax[5].twinx()
            ax5b.plot(r["t"], 100 * r["flagged"], "k-", lw=0.8, label="limited run: % cells flagged")
            ax5b.set_ylabel("% cells flagged by limiter")
            ax5b.legend(loc="upper left", fontsize=7)
    r = runs["TVB limited"][1]
    ax[2].plot(r["t"], r["energy_exact"], "k--", lw=1, label="exact")
    ax[3].plot(r["t"], r["umax_exact"], "k--", lw=1, label="exact max / min")
    ax[3].plot(r["t"], r["umin_exact"], "k--", lw=1)
    pre = r["t"] < tb
    ax[4].plot(r["t"][pre], np.maximum(r["slope_exact"][pre], -60), "k--", lw=1, label="exact  -A/(1 - At)")
    ax[4].set_ylim(-60, 1)
    titles = ["L1 error vs exact entropy solution", "relative mass drift |M(t) - M(0)| / M(0)",
              "energy ∫u²/2 dx", "solution bounds (overshoot diagnosis)",
              "steepest slope min du_h/dx", "shock sensors"]
    for a, ttl in zip(ax, titles):
        a.axvline(tb, color="crimson", lw=0.8, alpha=0.7)
        a.set_title(ttl, fontsize=10)
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="lower left" if a is ax[5] else "best")
    ax[0].text(tb, ax[0].get_ylim()[1], " t_b", color="crimson", va="top", fontsize=8)
    fig.suptitle(f"DG p={args.degree}, N={args.cells}, Godunov flux, SSP-RK3, "
                 f"u0 = {C} + {A} sin x  (shock forms at t_b = {tb:g}, red line)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "diagnostics_history.png"), dpi=140)
    plt.close(fig)

    # ------------------------------------------------------- spatial profiles
    show_t = [0.5, 0.9, 1.0, 1.5, 2.5]
    fig, ax = plt.subplots(3, len(show_t), figsize=(4 * len(show_t), 9), sharex=True)
    xf = np.linspace(0, 2 * np.pi, 4001)
    for k, t in enumerate(show_t):
        ex = lambda x, t=t: exact_solution(x, t, C, A)
        ax[0, k].plot(xf, ex(xf), "k-", lw=1.2, label="exact")
        for name, (s, r, out) in runs.items():
            j = int(np.argmin(np.abs(r["t"] - t)))
            U = out["snapshots"][float(r["t"][j])]
            xc, uc = piecewise_curve(s, U, 16)
            ax[0, k].plot(xc, uc, lw=1, label=name, **style[name])
            err = np.abs(uc - ex(xc))
            ax[1, k].semilogy(xc, np.maximum(err, 1e-16), lw=0.8, **style[name])
            if name == "TVB limited":
                ax[2, k].semilogy(s.centers, np.maximum(smoothness_indicator(s, U), 1e-16), ".",
                                  ms=4, label="smoothness indicator", **style[name])
                flagged = out["flags"][float(r["t"][j])]
                ax[2, k].plot(s.centers[flagged], np.full(flagged.sum(), 1.0), "rv", ms=5,
                              label="flagged by limiter")
        ax[0, k].set_title(f"t = {t:g}  (t/t_b = {t / tb:.2f})", fontsize=10)
        ax[1, k].set_ylim(1e-12, 3)
        ax[2, k].set_ylim(1e-14, 3)
        ax[2, k].set_xlabel("x")
        if t > tb:
            for a in ax[:, k]:
                a.axvline(shock_position(t, C), color="crimson", lw=0.8, alpha=0.6)
        for a in ax[:, k]:
            a.grid(True, alpha=0.3)
    ax[0, 0].set_ylabel("u")
    ax[1, 0].set_ylabel("|u_h - u_exact|")
    ax[2, 0].set_ylabel("p-mode energy fraction")
    ax[0, 0].legend(fontsize=7)
    ax[2, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Pointwise error and shock sensors, p={args.degree}, N={args.cells} "
                 "(red line: exact shock position)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "diagnostics_profiles.png"), dpi=140)
    plt.close(fig)

    # --------------------------------------------------------------- summary
    summary = {"cells": args.cells, "degree": args.degree, "t_end": args.t_end,
               "breaking_time_exact": tb, "runs": {}}
    for name, (s, r, out) in runs.items():
        pre = r["t"] <= tb
        pre_slope = r["t"] < tb
        rel_slope_err = np.abs(r["slope"][pre_slope] / r["slope_exact"][pre_slope] - 1.0)
        summary["runs"][name] = {
            "steps": out["n_steps"],
            "max_relative_mass_drift": float(np.max(np.abs(r["mass"] - r["mass"][0]) / abs(r["mass"][0]))),
            "energy_rel_change_before_tb": float((r["energy"][pre][-1] - r["energy"][0]) / r["energy"][0]),
            "energy_rel_change_exact_before_tb": float((r["energy_exact"][pre][-1] - r["energy_exact"][0])
                                                       / r["energy_exact"][0]),
            "energy_final": float(r["energy"][-1]),
            "energy_final_exact": float(r["energy_exact"][-1]),
            "energy_increase_events": int(np.sum(np.diff(r["energy"]) > 1e-12 * r["energy"][0])),
            "max_overshoot_above_exact_max": float(np.max(r["umax"] - r["umax_exact"])),
            "max_undershoot_below_exact_min": float(np.max(r["umin_exact"] - r["umin"])),
            "L1_error_at": {f"{t:g}": float(np.interp(t, r["t"], r["L1"])) for t in (0.5, 0.9, 1.0, 2.0, 3.0)
                            if t <= args.t_end},
            "slope_error_exceeds_10pct_at_t": first_time(r["t"][pre_slope], rel_slope_err > 0.1),
            "first_t_limiter_flags_any_cell": first_time(r["t"], r["flagged"] > 0),
            "first_t_max_interface_jump_gt_0.1": first_time(r["t"], r["max_jump"] > 0.1),
        }
    with open(os.path.join(RESULTS, "diagnostics_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
