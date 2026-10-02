"""Movie of DG Burgers steepening into a shock.

u0 = 0.5 + sin(x) on [0, 2 pi): the wave steepens, the characteristics cross
at t_b = 1 (at x = pi + 0.5), and a shock forms and travels at speed 0.5.

Top panel: exact entropy solution, the multivalued "characteristics" curve
(what you would get without the entropy condition, once t > t_b), and the DG
solutions (p = 2, Godunov flux) with and without the TVB limiter, drawn as
the piecewise polynomials they are.
Bottom left: pointwise error |u_h - u|.
Bottom right: steepest slope vs the exact -1/(1 - t), which diverges at t_b.

Output: results/burgers_shock.mp4 (or .gif if ffmpeg is unavailable) and a
strip of snapshots results/burgers_snapshots.png.
"""

import argparse
import os

import _common  # noqa: F401
from _common import A, C, RESULTS, TVB_M

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from dgburgers import DGBurgers, exact_solution, initial_condition, shock_position
from dgburgers.diagnostics import piecewise_curve, steepest_slope
from dgburgers.exact import breaking_time, characteristic_curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=int, default=48)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--t-end", type=float, default=1.4, help="final time (t_b = 1)")
    ap.add_argument("--frames", type=int, default=141)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true", help="write a GIF instead of MP4")
    args = ap.parse_args()
    tb = breaking_time(A)
    times = np.linspace(0.0, args.t_end, args.frames)

    runs = {}
    for name, lim in (("DG, TVB limited", "minmod"), ("DG, unlimited", None)):
        s = DGBurgers(args.cells, args.degree, limiter=lim, tvb_M=TVB_M)
        U0 = s.project(lambda x: initial_condition(x, C, A))
        out = s.solve(U0, args.t_end, cfl=0.5, integrator="ssprk3", save_times=times)
        runs[name] = (s, [out["snapshots"][float(t)] for t in times])
    style = {"DG, TVB limited": dict(color="tab:blue", lw=2.0),
             "DG, unlimited": dict(color="tab:orange", lw=1.0, alpha=0.9)}
    slopes = {n: np.array([steepest_slope(s, U) for U in Us]) for n, (s, Us) in runs.items()}

    xf = np.linspace(0.0, 2 * np.pi, 3001)
    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1.0], hspace=0.32, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_sl = fig.add_subplot(gs[1, 1])

    (l_char,) = ax.plot([], [], color="0.55", ls="--", lw=1.0, label="characteristics (multivalued)")
    (l_exact,) = ax.plot([], [], "k-", lw=1.3, label="exact entropy solution")
    l_dg = {n: ax.plot([], [], label=n, **style[n])[0] for n in runs}
    v_shock = ax.axvline(np.nan, color="crimson", lw=1.0, alpha=0.7)
    title = ax.set_title("")
    banner = ax.text(0.02, 0.06, "", transform=ax.transAxes, color="crimson", fontsize=12,
                     fontweight="bold")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.7, 2.9)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    e_dg = {n: ax_err.semilogy([], [], **{**style[n], "lw": 1.0})[0] for n in runs}
    ax_err.set_xlim(0, 2 * np.pi)
    ax_err.set_ylim(1e-9, 3)
    ax_err.set_xlabel("x")
    ax_err.set_title("pointwise error |u_h - u_exact|", fontsize=10)
    ax_err.grid(True, alpha=0.3)

    pre = times < tb
    ax_sl.plot(times[pre], -A / (1 - A * times[pre]), "k--", lw=1, label="exact  -1/(1 - t)")
    s_dg = {n: ax_sl.plot([], [], label=n, **{**style[n], "lw": 1.3})[0] for n in runs}
    marker = ax_sl.axvline(0.0, color="0.3", lw=0.8)
    ax_sl.axvline(tb, color="crimson", lw=0.8, alpha=0.7)
    ax_sl.text(tb, -2, " t_b", color="crimson", fontsize=9)
    ax_sl.set_xlim(0, args.t_end)
    ax_sl.set_ylim(-60, 2)
    ax_sl.set_xlabel("t")
    ax_sl.set_title("steepest slope  min du/dx", fontsize=10)
    ax_sl.grid(True, alpha=0.3)
    ax_sl.legend(fontsize=8, loc="lower left")

    fig.suptitle(f"Burgers' equation  u_t + (u²/2)_x = 0,  u₀ = {C} + sin x  —  "
                 f"DG p = {args.degree}, N = {args.cells}, Godunov flux, SSP-RK3", fontsize=11)

    def draw(k):
        t = times[k]
        xc, uc = characteristic_curve(t, C, A)
        l_char.set_data(xc, uc)
        l_char.set_visible(t > tb)
        l_exact.set_data(xf, exact_solution(xf, t, C, A))
        for n, (s, Us) in runs.items():
            x, u = piecewise_curve(s, Us[k], 16)
            l_dg[n].set_data(x, u)
            e_dg[n].set_data(x, np.maximum(np.abs(u - exact_solution(x, t, C, A)), 1e-16))
            s_dg[n].set_data(times[: k + 1], slopes[n][: k + 1])
        v_shock.set_xdata([shock_position(t, C)] * 2 if t >= tb else [np.nan] * 2)
        marker.set_xdata([t, t])
        title.set_text(f"t = {t:.3f}    t / t_b = {t / tb:.2f}")
        if t < tb:
            banner.set_text(f"steepening: max|u_x| = {A / (1 - A * t):.1f}")
        else:
            banner.set_text(f"shock formed at t_b = {tb:g}, x = π + {C * tb:g};  now at x_s = {shock_position(t, C):.3f}")
        return []

    # Hold the last frame for ~1.5 s.
    order = list(range(len(times))) + [len(times) - 1] * int(1.5 * args.fps)
    anim = animation.FuncAnimation(fig, draw, frames=order, blit=False)
    use_gif = args.gif or not animation.writers.is_available("ffmpeg")
    path = os.path.join(RESULTS, "burgers_shock." + ("gif" if use_gif else "mp4"))
    if use_gif:
        writer = animation.PillowWriter(fps=args.fps)
    else:
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400,
                                        extra_args=["-pix_fmt", "yuv420p"])
    anim.save(path, writer=writer, dpi=110)
    plt.close(fig)
    print("wrote", path)

    # ------------------------------------------------------- snapshot strip
    snap_t = [0.0, 0.5, 0.8, 0.95, tb, args.t_end]
    fig, axs = plt.subplots(1, len(snap_t), figsize=(3.2 * len(snap_t), 3.3), sharey=True)
    for a, t in zip(axs, snap_t):
        k = int(np.argmin(np.abs(times - t)))
        t = times[k]
        if t > tb:
            a.plot(*characteristic_curve(t, C, A), color="0.55", ls="--", lw=0.9)
        a.plot(xf, exact_solution(xf, t, C, A), "k-", lw=1.2)
        for n, (s, Us) in runs.items():
            a.plot(*piecewise_curve(s, Us[k], 16), **style[n])
        a.set_title(f"t = {t:.2f}", fontsize=10)
        a.set_xlim(0, 2 * np.pi)
        a.set_ylim(-1.7, 2.9)
        a.grid(True, alpha=0.3)
        a.set_xlabel("x")
    axs[0].set_ylabel("u")
    axs[0].legend([plt.Line2D([], [], color="k"), plt.Line2D([], [], **style["DG, TVB limited"]),
                   plt.Line2D([], [], **style["DG, unlimited"])],
                  ["exact", "DG limited", "DG unlimited"], fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "burgers_snapshots.png"), dpi=140)
    plt.close(fig)
    print("wrote", os.path.join(RESULTS, "burgers_snapshots.png"))


if __name__ == "__main__":
    main()
