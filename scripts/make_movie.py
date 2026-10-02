"""Movie of the DG Burgers solution steepening into a shock.

Default: p = 2, N = 100, TVB-limited, Godunov flux, t in [0, 1.2 t_s].
Writes results/burgers_shock.mp4 (ffmpeg) or .gif (Pillow fallback), plus
results/shock_snapshots.png with a few key frames.
"""

import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dg_burgers import DGBurgers, diagnostics as d, exact  # noqa: E402

TS = exact.breaking_time()


def simulate(p, N, times, M, flux):
    s = DGBurgers(N, p, flux=flux, limiter=True, tvb_M=M)
    t_end = times[-1]
    frames = []

    def cb(t, U):
        frames.append((t, U.copy(), d.max_gradient(s, U), d.max_jump(s, U)))

    s.run(s.project(exact.u0), t_end, output_times=list(times), callback=cb)
    return s, frames


def make_figure(s, frames, t_end):
    xf = np.linspace(0, exact.L_DOMAIN, 1500)
    fig = plt.figure(figsize=(11, 7.6))  # 1210x836 px at dpi 110 (x264 needs even sizes)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[1.6, 1], hspace=0.32, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    axz = fig.add_subplot(gs[1, 0])
    axg = fig.add_subplot(gs[1, 1])

    ax.plot(xf, exact.u0(xf), color="0.75", lw=1, ls=":", label="u0")
    ex_line, = ax.plot([], [], "k-", lw=1.0, label="exact (characteristics / entropy)")
    dg_line, = ax.plot([], [], color="#d62728", lw=1.8, label=f"DG p={s.p}, N={s.N}")
    shock_line = ax.axvline(np.nan, color="#1f77b4", ls="--", lw=1)
    ax.set_xlim(0, exact.L_DOMAIN)
    ax.set_ylim(-0.75, 1.75)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    title = ax.set_title("")
    status = ax.text(0.99, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=11,
                     bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # Zoom that follows the steepening region / shock (moves with speed U_MEAN).
    zx_line, = axz.plot([], [], "k-", lw=1.0)
    zd_line, = axz.plot([], [], color="#d62728", lw=1.8)
    z_edges = axz.vlines([], 0, 1, colors="0.85", lw=0.6)
    zshock = axz.axvline(np.nan, color="#1f77b4", ls="--", lw=1)
    axz.set_ylim(-0.75, 1.75)
    axz.set_xlabel("x  (zoom around the steepest point; grey = cell edges)")
    axz.set_ylabel("u")

    t_all = np.array([f[0] for f in frames])
    g_all = np.array([f[2] for f in frames])
    j_all = np.array([f[3] for f in frames])
    tt = np.linspace(0, 0.995 * TS, 400)
    axg.semilogy(tt / TS, exact.max_gradient_exact(tt), "k--", lw=1, label="exact 1/(1 - t/t_s)")
    g_line, = axg.semilogy([], [], color="#d62728", lw=1.5, label="DG max|∂u/∂x| in cells")
    axg.axvline(1.0, color="0.5", ls=":", lw=1)
    axg.set_xlim(0, t_end / TS)
    axg.set_ylim(0.8, 2.0 * max(g_all.max(), 50))
    axg.set_xlabel("t / t_s")
    axg.set_ylabel("max |∂u/∂x|")
    axg.grid(True, which="both", alpha=0.3)
    axj = axg.twinx()
    j_line, = axj.plot([], [], color="#1f77b4", lw=1, alpha=0.8, label="max interface jump")
    axj.set_ylim(0, 2.2)
    axj.set_ylabel("max interface jump", color="#1f77b4")
    lines = [axg.lines[0], g_line, j_line]
    axg.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7)

    def draw(k):
        nonlocal z_edges
        t, U, _, _ = frames[k]
        x, u = d.dense_profile(s, U, npts=16)
        dg_line.set_data(x, u)
        ex_line.set_data(xf, exact.entropy_solution(xf, t))
        center = exact.shock_position(t)   # also the steepest point before t_s
        if t >= TS:
            shock_line.set_xdata([center, center])
            zshock.set_xdata([center, center])
            status.set_text(f"t/t_s = {t / TS:.3f}\nshock formed, x_s = π + 0.5 t = {center:.3f}")
            status.get_bbox_patch().set_edgecolor("#1f77b4")
        else:
            shock_line.set_xdata([np.nan, np.nan])
            zshock.set_xdata([np.nan, np.nan])
            status.set_text(f"t/t_s = {t / TS:.3f}\nsteepening (max|u_x| exact = {exact.max_gradient_exact(t):.1f})")
            status.get_bbox_patch().set_edgecolor("0.7")
        title.set_text("Inviscid Burgers  u_t + (u²/2)_x = 0,  u0 = 0.5 + sin x  —  DG (Legendre, Godunov flux, SSP-RK3, TVB limiter)")
        title.set_fontsize(10)

        half = 0.6
        lo, hi = center - half, center + half
        zxf = np.linspace(lo, hi, 600)
        zx_line.set_data(zxf, exact.entropy_solution(zxf, t))
        # Shift DG data by a period if the window wraps.
        xs = np.concatenate([x - exact.L_DOMAIN, x, x + exact.L_DOMAIN])
        us = np.concatenate([u, u, u])
        zd_line.set_data(xs, us)
        axz.set_xlim(lo, hi)
        e = np.concatenate([s.edges - exact.L_DOMAIN, s.edges, s.edges + exact.L_DOMAIN])
        e = e[(e >= lo) & (e <= hi)]
        z_edges.remove()
        z_edges = axz.vlines(e, -0.75, 1.75, colors="0.85", lw=0.6, zorder=0)

        g_line.set_data(t_all[:k + 1] / TS, g_all[:k + 1])
        j_line.set_data(t_all[:k + 1] / TS, j_all[:k + 1])
        return dg_line, ex_line, shock_line, zx_line, zd_line, g_line, j_line, status

    return fig, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--t-end-factor", type=float, default=1.2, help="end time in units of t_s")
    ap.add_argument("--frames", type=int, default=181)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tvb-M", type=float, default=5.0)
    ap.add_argument("--flux", default="godunov", choices=["godunov", "rusanov"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "burgers_shock.mp4"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    t_end = args.t_end_factor * TS
    times = np.linspace(0.0, t_end, args.frames)
    if t_end >= TS:
        times = np.union1d(times, [TS])   # always show the breaking time itself
    s, frames = simulate(args.p, args.N, times, args.tvb_M, args.flux)
    print(f"simulated {len(frames)} frames, mass drift "
          f"{abs(d.total_mass(s, frames[-1][1]) - d.total_mass(s, frames[0][1])):.1e}")

    fig, draw = make_figure(s, frames, t_end)
    out = args.out
    if animation.FFMpegWriter.isAvailable() or shutil.which("ffmpeg"):
        writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264",
                                        extra_args=["-pix_fmt", "yuv420p"])
    else:
        out = os.path.splitext(out)[0] + ".gif"
        writer = animation.PillowWriter(fps=args.fps)
    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"movie -> {os.path.abspath(out)}")

    # Key-frame snapshot figure.
    key = [0.0, 0.5, 0.9, 1.0, t_end / TS]
    fig, axes = plt.subplots(1, len(key), figsize=(4 * len(key), 3.4), sharey=True)
    tf = np.array([f[0] for f in frames])
    xf = np.linspace(0, exact.L_DOMAIN, 1500)
    for ax, kt in zip(axes, key):
        k = int(np.argmin(np.abs(tf - kt * TS)))
        t, U = frames[k][0], frames[k][1]
        x, u = d.dense_profile(s, U, npts=16)
        ax.plot(xf, exact.entropy_solution(xf, t), "k-", lw=1)
        ax.plot(x, u, color="#d62728", lw=1.5)
        if t >= TS:
            ax.axvline(exact.shock_position(t), color="#1f77b4", ls="--", lw=1)
        ax.set_title(f"t/t_s = {t / TS:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("u")
    fig.suptitle(f"DG p={s.p}, N={s.N} (red) vs exact (black); dashed = shock")
    fig.tight_layout()
    snap = os.path.join(os.path.dirname(os.path.abspath(out)), "shock_snapshots.png")
    fig.savefig(snap, dpi=130)
    plt.close(fig)
    print(f"snapshots -> {snap}")


if __name__ == "__main__":
    main()
