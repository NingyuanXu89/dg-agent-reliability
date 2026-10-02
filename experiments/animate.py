"""Animate DG-Pp for Burgers from smooth sine data through shock formation.

The solution steepens until the breaking time t_s = 1/(2 pi), when a shock forms
at x_s = 1/2 + t_s/2. The run continues briefly past t_s (default 1.5 t_s) so the
formed shock is visible; the TVB limiter keeps the post-shock solution free of
Gibbs oscillations. Each frame shows

  * u_h against the exact entropy solution (characteristics / Hopf-Lax), with
    the cells flagged by the limiter marked along the bottom,
  * max |u_x| of u_h against the exact blow-up a k / (1 - a k t),
  * the L1 error against the entropy solution and the number of limited cells.

Outputs: results/burgers_shock.gif (and .mp4 when ffmpeg is available),
results/shock_snapshots.png, results/animation_summary.json.

Usage:  python -m experiments.animate [--cells 100] [--degree 2] [--t-end 1.5] [--frames 151]
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from burgers_dg import BurgersDG, SineInitialData, error_norms, exact_solution
from burgers_dg.diagnostics import energy, max_abs_gradient, total_mass

XI_PLOT = np.linspace(-1.0, 1.0, 10)


def dg_polyline(solver, C):
    """Piecewise-polynomial plot data, with NaN breaks so jumps between cells stay visible."""
    x = np.hstack([solver.x_physical(XI_PLOT), np.full((solver.n_cells, 1), np.nan)]).ravel()
    u = np.hstack([solver.evaluate(C, XI_PLOT), np.full((solver.n_cells, 1), np.nan)]).ravel()
    return x, u


def simulate(solver, data, times):
    C = solver.project(data.u0)
    frames = []
    t_prev = 0.0
    for t in times:
        C, _ = solver.advance(C, t_prev, t)
        t_prev = t
        frames.append({
            "t": t,
            "C": C.copy(),
            "limited": solver.last_limited.copy() if t > 0 else np.zeros(solver.n_cells, bool),
            "max_grad": max_abs_gradient(solver, C),
            "L1": error_norms(solver, C, lambda x, t=t: exact_solution(x, t, data))["L1"],
            "mass": total_mass(solver, C),
            "energy": energy(solver, C),
        })
    return frames


def estimate_shock_time(frames, data, fit_fraction=0.6):
    """Extrapolate 1 / max|u_x| (exactly linear in t before breaking) to zero."""
    t = np.array([f["t"] for f in frames])
    g = np.array([f["max_grad"] for f in frames])
    mask = t <= fit_fraction * data.shock_time
    slope, intercept = np.polyfit(t[mask], 1.0 / g[mask], 1)
    return -intercept / slope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--t-end", type=float, default=1.5, help="final time in units of t_s")
    parser.add_argument("--frames", type=int, default=151)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    data = SineInitialData()
    ts = data.shock_time
    # TVB constant ~ (2/3) max|u0''| (Cockburn & Shu) so smooth extrema are not clipped.
    tvb_M = 2.0 / 3.0 * data.amplitude * data.wavenumber ** 2
    solver = BurgersDG(args.cells, args.degree, domain=(0.0, data.length), limiter="tvb", tvb_M=tvb_M)
    times = np.union1d(np.linspace(0.0, args.t_end * ts, args.frames), [0.5 * ts, 0.9 * ts, ts])
    print(f"P{args.degree}, N={args.cells}, TVB M={tvb_M:.2f}: simulating {times.size} frames "
          f"to t = {times[-1]:.4f} ({args.t_end} t_s)")
    frames = simulate(solver, data, times)

    ts_est = estimate_shock_time(frames, data)
    first_limited = next((f["t"] for f in frames if f["limited"].any()), None)
    masses = np.array([f["mass"] for f in frames])
    summary = {
        "degree": args.degree,
        "cells": args.cells,
        "tvb_M": tvb_M,
        "shock_time_exact": ts,
        "shock_time_estimated_from_gradient": ts_est,
        "shock_time_relative_error": abs(ts_est - ts) / ts,
        "shock_location_exact": data.shock_location,
        "first_limiter_activation_time": first_limited,
        "max_mass_drift": float(np.abs(masses - masses[0]).max()),
        "energy_initial": frames[0]["energy"],
        "energy_final": frames[-1]["energy"],
        "L1_error_at": {f"{k} t_s": next(f["L1"] for f in frames if np.isclose(f["t"], k * ts))
                        for k in (0.5, 0.9, 1.0)},
        "L1_error_final": frames[-1]["L1"],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "animation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot_snapshots(args.outdir / "shock_snapshots.png", solver, data, frames)
    anim, fig = build_animation(solver, data, frames)
    gif = args.outdir / "burgers_shock.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=args.fps), dpi=90)
    print(f"wrote {gif}")
    if shutil.which("ffmpeg"):
        mp4 = args.outdir / "burgers_shock.mp4"
        anim.save(mp4, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400), dpi=130)
        print(f"wrote {mp4}")
    plt.close(fig)
    return summary


def build_animation(solver, data, frames):
    ts = data.shock_time
    x_fine = np.linspace(0.0, data.length, 801)
    exact_frames = [exact_solution(x_fine, f["t"], data) for f in frames]
    t = np.array([f["t"] for f in frames])
    grad = np.array([f["max_grad"] for f in frames])
    l1 = np.array([f["L1"] for f in frames])
    n_limited = np.array([f["limited"].sum() for f in frames])

    fig = plt.figure(figsize=(10, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], hspace=0.35, wspace=0.3)
    ax_u = fig.add_subplot(grid[0, :])
    ax_g = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])

    (line_exact,) = ax_u.plot([], [], "k--", lw=1.2, label="exact entropy solution")
    (line_dg,) = ax_u.plot([], [], color="tab:blue", lw=1.8, label=f"DG-P{solver.degree}, N={solver.n_cells}")
    (marks,) = ax_u.plot([], [], "|", color="tab:red", ms=10, mew=1.5, label="limited cells")
    ax_u.axvline(data.shock_location, color="0.75", lw=0.8, zorder=0, label="shock formation point $x_s$")
    ax_u.set_xlim(0.0, data.length)
    ax_u.set_ylim(data.u_min - 0.25, data.u_max + 0.25)
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u")
    ax_u.legend(loc="upper right", fontsize=8)
    title = ax_u.set_title("")

    t_exact = np.linspace(0.0, 0.985 * ts, 300)
    ax_g.semilogy(t_exact / ts, [data.max_gradient(s) for s in t_exact], "k--", lw=1.2, label="exact")
    (line_g,) = ax_g.semilogy([], [], color="tab:blue", lw=1.8, label="DG")
    (dot_g,) = ax_g.semilogy([], [], "o", color="tab:blue")
    ax_g.axvline(1.0, color="tab:red", lw=0.8, ls=":")
    ax_g.set_xlim(0.0, t[-1] / ts)
    ax_g.set_ylim(0.8 * grad.min(), 3.0 * grad.max())
    ax_g.set_xlabel("t / t_s")
    ax_g.set_ylabel("max |u_x|")
    ax_g.legend(fontsize=8, loc="upper left")

    (line_e,) = ax_e.semilogy([], [], color="tab:purple", lw=1.8, label="L1 error")
    ax_e.set_xlim(0.0, t[-1] / ts)
    positive = l1[l1 > 0]
    ax_e.set_ylim(0.5 * positive.min(), 2.0 * positive.max())
    ax_e.set_xlabel("t / t_s")
    ax_e.set_ylabel("L1 error", color="tab:purple")
    ax_e.axvline(1.0, color="tab:red", lw=0.8, ls=":")
    ax_n = ax_e.twinx()
    (line_n,) = ax_n.plot([], [], color="tab:red", lw=1.0, alpha=0.7)
    ax_n.set_ylim(0, max(4, 1.3 * n_limited.max()))
    ax_n.set_ylabel("limited cells", color="tab:red")

    y_marks = data.u_min - 0.18

    def update(i):
        f = frames[i]
        line_exact.set_data(x_fine, exact_frames[i])
        line_dg.set_data(*dg_polyline(solver, f["C"]))
        marks.set_data(solver.centers[f["limited"]], np.full(f["limited"].sum(), y_marks))
        phase = "before shock" if f["t"] < ts else ("shock forms" if np.isclose(f["t"], ts) else "after shock")
        title.set_text(f"Burgers, u0 = 0.5 + sin(2πx):  t = {f['t']:.4f} = {f['t'] / ts:.2f} t_s  ({phase})")
        line_g.set_data(t[: i + 1] / ts, grad[: i + 1])
        dot_g.set_data([t[i] / ts], [grad[i]])
        line_e.set_data(t[1: i + 1] / ts, l1[1: i + 1])
        line_n.set_data(t[: i + 1] / ts, n_limited[: i + 1])
        return line_exact, line_dg, marks, title, line_g, dot_g, line_e, line_n

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    return anim, fig


def plot_snapshots(path, solver, data, frames):
    ts = data.shock_time
    picks = [0.0, 0.5, 0.9, 1.0, frames[-1]["t"] / ts]
    x_fine = np.linspace(0.0, data.length, 1201)
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.1 * len(picks)), sharex=True)
    for ax, k in zip(axes, picks):
        f = min(frames, key=lambda fr: abs(fr["t"] - k * ts))
        ax.plot(x_fine, exact_solution(x_fine, f["t"], data), "k--", lw=1.1, label="exact")
        ax.plot(*dg_polyline(solver, f["C"]), color="tab:blue", lw=1.5, label=f"DG-P{solver.degree}")
        ax.set_ylabel("u")
        ax.set_title(f"t = {f['t'] / ts:.2f} t_s   (L1 error {f['L1']:.2e})", fontsize=9)
        ax.set_ylim(data.u_min - 0.2, data.u_max + 0.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
