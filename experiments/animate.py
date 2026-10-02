"""Animate the DG solution of Burgers from smooth data through shock formation.

p = 2, K = 128, Godunov flux, SSP-RK3, TVB limiter.  Runs to 1.3 t_s so the
shock is visibly established.  Top panel: DG solution (per cell, so inter-cell
jumps are visible) against the exact entropy solution, with troubled
(limited) cells marked.  Bottom panel: numerical max |u_x| inside cells
against the exact blow-up a/(1 - a t).

Outputs: results/burgers_shock.mp4 (if ffmpeg is available), burgers_shock.gif,
snapshots.png, gradient_blowup.png, animation_summary.json.
"""

import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from common import A, C, T_SHOCK, out, u0
from burgers_dg import DGBurgers, TVBLimiter, exact_solution, shock_position

P, K = 2, 128
T_END = 1.3 * T_SHOCK
N_FRAMES = 131                      # dt_frame = 0.01
X_FINE = np.linspace(0.0, 2 * np.pi, 2001)


def simulate():
    s = DGBurgers(K, P, limiter=TVBLimiter(M=1.0))
    c = s.project(u0)
    frame_t = np.linspace(0.0, T_END, N_FRAMES)
    frames = [(0.0, c.copy(), s.troubled.copy())]
    hist = {"t": [0.0], "grad": [s.max_abs_gradient(c)], "n_troubled": [int(s.troubled.sum())]}

    def rec(t, c):
        hist["t"].append(t)
        hist["grad"].append(s.max_abs_gradient(c))
        hist["n_troubled"].append(int(s.troubled.sum()))

    for t0, t1 in zip(frame_t[:-1], frame_t[1:]):
        c, _ = s.run(c, t1, t0=t0, on_step=rec)
        frames.append((t1, c.copy(), s.troubled.copy()))
    return s, frames, {k: np.array(v) for k, v in hist.items()}


def detect_shock(hist):
    """Two resolution-dependent indicators of shock formation on this mesh."""
    t, g, n = hist["t"], hist["grad"], hist["n_troubled"]
    first_limited = float(t[np.argmax(n > 0)]) if np.any(n > 0) else None
    pre = t < T_SHOCK
    g_exact = A / (1 - A * t[pre])
    lag = np.nonzero(g[pre] < 0.9 * g_exact)[0]
    first_unresolved = float(t[pre][lag[0]]) if lag.size else None
    return first_limited, first_unresolved


def plot_frame(ax, s, t, c, troubled):
    x, u = s.sample(c, 10)
    ax.plot(X_FINE, exact_solution(X_FINE, t, C, A), color="0.45", ls="--", lw=1.2,
            label="exact (characteristics + RH shock)")
    ax.plot(x, u, color="C0", lw=1.6, label=f"DG  p = {P}, K = {K}")
    if troubled.any():
        xc = s.centers[troubled]
        ax.plot(xc, np.full_like(xc, -0.72), "v", color="C3", ms=5, label="limited cells")
    ax.axvline(shock_position(t, C), color="C3", lw=0.7, alpha=0.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-0.8, 1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("u")


def make_animation(s, frames, hist, first_limited):
    fig, (ax, axg) = plt.subplots(2, 1, figsize=(8.5, 6.6),
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    tt = np.linspace(0, 0.995 * T_SHOCK, 400)

    def draw(i):
        t, c, troubled = frames[i]
        ax.cla()
        plot_frame(ax, s, t, c, troubled)
        state = "smooth: characteristics steepening" if t < T_SHOCK else "SHOCK FORMED  (t ≥ t_s = 1)"
        ax.set_title(f"Inviscid Burgers  $u_t + (u^2/2)_x = 0$,  $u_0 = {C} + \\sin x$     t = {t:.2f}\n{state}",
                     fontsize=10, color="k" if t < T_SHOCK else "C3")
        ax.legend(loc="upper right", fontsize=7.5)
        axg.cla()
        axg.semilogy(tt, A / (1 - A * tt), "k--", lw=1, label="exact  a/(1 − a t)")
        m = hist["t"] <= t + 1e-12
        axg.semilogy(hist["t"][m], hist["grad"][m], color="C0", label="DG max |u_x| in cells")
        axg.axvline(T_SHOCK, color="C3", lw=0.8)
        if first_limited is not None and t >= first_limited:
            axg.axvline(first_limited, color="C3", ls=":", lw=0.8, label="limiter first active")
        axg.set_xlim(0, T_END)
        axg.set_ylim(0.8, 300)
        axg.set_xlabel("t")
        axg.set_ylabel("max |u_x|")
        axg.legend(loc="upper left", fontsize=7.5)
        axg.grid(True, alpha=0.3)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=60)
    written = []
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        plt.rcParams["animation.ffmpeg_path"] = ffmpeg
        anim.save(out("burgers_shock.mp4"), writer=FFMpegWriter(fps=20, bitrate=1800), dpi=110)
        written.append("burgers_shock.mp4")
    anim.save(out("burgers_shock.gif"), writer=PillowWriter(fps=15), dpi=60)
    written.append("burgers_shock.gif")
    plt.close(fig)
    return written


def plot_snapshots(s, frames):
    want = [0.0, 0.5, 0.8, 0.95, 1.0, 1.3]
    by_t = {round(t, 6): (t, c, tr) for t, c, tr in frames}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, tw in zip(axes.ravel(), want):
        t, c, tr = by_t[round(tw, 6)]
        plot_frame(ax, s, t, c, tr)
        ax.set_title(f"t = {t:.2f}" + ("  (shock)" if t >= T_SHOCK else ""))
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out("snapshots.png"), dpi=120)
    plt.close(fig)


def plot_gradient(hist, first_limited, first_unresolved):
    fig, ax = plt.subplots(figsize=(8, 4))
    tt = np.linspace(0, 0.995 * T_SHOCK, 400)
    ax.semilogy(tt, A / (1 - A * tt), "k--", label="exact max |u_x| = a/(1 − a t)")
    ax.semilogy(hist["t"], hist["grad"], label=f"DG max |u_x| in cells (p = {P}, K = {K})")
    ax.axvline(T_SHOCK, color="C3", lw=0.8, label="$t_s = 1$")
    if first_limited is not None:
        ax.axvline(first_limited, color="C3", ls=":", label=f"limiter first active, t = {first_limited:.3f}")
    if first_unresolved is not None:
        ax.axvline(first_unresolved, color="C2", ls=":", label=f"DG < 90% of exact, t = {first_unresolved:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("max |u_x|")
    ax.set_title("Gradient blow-up at shock formation")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out("gradient_blowup.png"), dpi=130)
    plt.close(fig)


def main():
    s, frames, hist = simulate()
    first_limited, first_unresolved = detect_shock(hist)
    plot_snapshots(s, frames)
    plot_gradient(hist, first_limited, first_unresolved)
    written = make_animation(s, frames, hist, first_limited)
    t_end, c_end, _ = frames[-1]
    summary = {
        "p": P, "K": K, "t_end": t_end, "frames": len(frames), "steps": int(len(hist["t"]) - 1),
        "t_shock_exact": T_SHOCK,
        "t_first_limiter_activation": first_limited,
        "t_gradient_below_90pct_exact": first_unresolved,
        "max_grad_numerical": float(hist["grad"].max()),
        "L1_error_t_end": float(s.errors(c_end, lambda x: exact_solution(x, t_end, C, A))[0]),
        "files": written,
    }
    with open(out("animation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
