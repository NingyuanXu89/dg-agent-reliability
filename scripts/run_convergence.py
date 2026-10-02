"""Convergence study and error diagnosis for the DG Burgers solver.

Writes to results/:
  convergence.csv, convergence_report.md       smooth (pre-shock) error tables
  convergence.png                              L1/L2/Linf vs h with reference slopes
  error_profile.png                            pointwise and per-cell error structure
  error_vs_time.png                            error growth approaching breaking
  invariants.png                               mass and energy history through the shock
  post_shock_convergence.png                   L1 error after the shock (full / away from shock)
"""

import argparse
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dg_burgers import DGBurgers, diagnostics as d, exact  # noqa: E402

TS = exact.breaking_time()
COLORS = {0: "#7f7f7f", 1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}


def dt_factor(p, h, h0):
    """SSP-RK3 is 3rd order in time: shrink dt like h^((p+1)/3) for p >= 3 so
    the temporal error does not mask the spatial order p + 1."""
    return min(1.0, (h / h0) ** max(0.0, (p + 1) / 3.0 - 1.0))


def smooth_study(degrees, Ns, t_eval, flux="godunov"):
    rows = []
    ref = lambda x: exact.exact_solution(x, t_eval)
    h0 = exact.L_DOMAIN / Ns[0]
    for p in degrees:
        for N in Ns:
            s = DGBurgers(N, p, flux=flux)
            t0 = time.perf_counter()
            U, nsteps = s.run(s.project(exact.u0), t_eval, dt_factor=dt_factor(p, s.h, h0))
            wall = time.perf_counter() - t0
            e = d.error_norms(s, U, ref)
            rows.append(dict(p=p, N=N, h=s.h, steps=nsteps, wall_s=wall,
                             mass_drift=d.total_mass(s, U) - exact.U_MEAN * exact.L_DOMAIN, **e))
    for p in degrees:
        sub = [r for r in rows if r["p"] == p]
        for norm in ("L1", "L2", "Linf"):
            orders = d.observed_orders([r["h"] for r in sub], [r[norm] for r in sub])
            sub[0][f"order_{norm}"] = np.nan
            for r, o in zip(sub[1:], orders):
                r[f"order_{norm}"] = o
    return rows


def post_shock_study(degrees, Ns, t_eval, M, window=0.25):
    xs = exact.shock_position(t_eval)
    ref = lambda x: exact.entropy_solution(x, t_eval)
    near = lambda x: np.abs(np.mod(x - xs + np.pi, 2 * np.pi) - np.pi) < window
    jump = exact.entropy_solution(xs - 1e-12, t_eval) - exact.entropy_solution(xs + 1e-12, t_eval)
    rows = []
    for p in degrees:
        for N in Ns:
            s = DGBurgers(N, p, limiter=True, tvb_M=M)
            U, _ = s.run(s.project(exact.u0), t_eval)
            full = d.error_norms(s, U, ref, nq=p + 16)["L1"]
            away = d.error_norms(s, U, ref, nq=p + 16, exclude=near)["L1"]
            rows.append(dict(p=p, N=N, h=s.h, L1=full, L1_scaled=full / (jump * s.h), L1_away=away))
    for p in degrees:
        sub = [r for r in rows if r["p"] == p]
        for key in ("L1", "L1_away"):
            orders = d.observed_orders([r["h"] for r in sub], [r[key] for r in sub])
            sub[0][f"order_{key}"] = np.nan
            for r, o in zip(sub[1:], orders):
                r[f"order_{key}"] = o
    return rows, jump


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


# --------------------------------------------------------------------- plots
def plot_convergence(rows, degrees, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, norm in zip(axes, ("L1", "L2", "Linf")):
        for p in degrees:
            sub = [r for r in rows if r["p"] == p]
            h = np.array([r["h"] for r in sub])
            e = np.array([r[norm] for r in sub])
            ax.loglog(h, e, "o-", color=COLORS[p], label=f"p={p}")
            ax.loglog(h, e[-1] * (h / h[-1]) ** (p + 1), ":", color=COLORS[p], lw=1)
        ax.set_title(f"{norm} error at t = {rows[0]['t']:.2f} t_s")
        ax.set_xlabel("h")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("error  (dotted: slope p+1)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_error_profile(t_eval, out, p=2, Ns=(20, 40, 80)):
    ref = lambda x: exact.exact_solution(x, t_eval)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    s = DGBurgers(Ns[0], p)
    U, _ = s.run(s.project(exact.u0), t_eval)
    x, u = d.dense_profile(s, U, npts=20)
    xf = np.linspace(0, exact.L_DOMAIN, 1000)
    axes[0].plot(xf, ref(xf), "k-", lw=1, label="exact (characteristics)")
    axes[0].plot(x, u, color=COLORS[p], lw=1.5, label=f"DG p={p}, N={Ns[0]}")
    axes[0].plot(xf, exact.u0(xf), "k:", lw=0.8, label="u0")
    axes[0].set_ylabel("u")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title(f"Solution and error structure at t = {t_eval:.2f} t_s")
    for N, ls in zip(Ns, ("-", "--", ":")):
        s = DGBurgers(N, p)
        U, _ = s.run(s.project(exact.u0), t_eval)
        x, u = d.dense_profile(s, U, npts=20)
        axes[1].semilogy(x, np.abs(u - ref(x)), ls, lw=1, label=f"N={N}")
        axes[2].semilogy(s.centers, d.cell_l2_errors(s, U, ref), "o" + ls, ms=3, lw=1, label=f"N={N}")
        axes[2].semilogy(s.edges[1:], np.abs(d.interface_jumps(s, U)) + 1e-18, "x", ms=3,
                         color=axes[2].lines[-1].get_color(), alpha=0.5)
    axes[1].set_ylabel("|u_h - u|")
    axes[1].legend(fontsize=8)
    axes[2].set_ylabel("cell L2 error (o)\n|interface jump| (x)")
    axes[2].set_xlabel("x")
    axes[2].legend(fontsize=8)
    ax2 = axes[1].twinx()
    ax2.plot(xf, np.abs(np.gradient(ref(xf), xf)), color="gray", alpha=0.4, lw=1)
    ax2.set_ylabel("|u_x| exact", color="gray")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_error_vs_time(out, p=2, Ns=(20, 40, 80, 160)):
    times = np.linspace(0, 0.98, 50) * TS
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for N in Ns:
        s = DGBurgers(N, p)
        errs = []
        s.run(s.project(exact.u0), times[-1], output_times=list(times),
              callback=lambda t, U: errs.append(
                  d.error_norms(s, U, lambda x: exact.exact_solution(x, t))["L2"]))
        ax.semilogy(times / TS, errs, label=f"N={N}")
    ax2 = ax.twinx()
    ax2.semilogy(times / TS, exact.max_gradient_exact(times), "k--", lw=1)
    ax2.set_ylabel("max |u_x| exact = 1/(1 - t/t_s)  (dashed)")
    ax.set_xlabel("t / t_s")
    ax.set_ylabel(f"L2 error (p={p})")
    ax.set_title("Error growth as the gradient steepens toward breaking")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_invariants(out, p=2, N=200, M=5.0, t_end=2.0):
    s = DGBurgers(N, p, limiter=True, tvb_M=M)
    times = np.linspace(0, t_end, 201)
    hist = {"t": [], "mass": [], "energy": [], "jump": []}

    def cb(t, U):
        hist["t"].append(t)
        hist["mass"].append(d.total_mass(s, U))
        hist["energy"].append(d.total_energy(s, U))
        hist["jump"].append(d.max_jump(s, U))

    s.run(s.project(exact.u0), t_end, output_times=list(times), callback=cb)
    t = np.array(hist["t"])
    xf = np.linspace(0, exact.L_DOMAIN, 200001)[:-1]
    E_ex = np.array([np.mean(exact.entropy_solution(xf, tt) ** 2 / 2) * exact.L_DOMAIN for tt in t[::5]])
    m0 = exact.U_MEAN * exact.L_DOMAIN
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(t / TS, np.array(hist["mass"]) - m0)
    axes[0].set_ylabel("mass - mass(0)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].plot(t / TS, hist["energy"], label=f"DG p={p}, N={N}, TVB M={M:g}")
    axes[1].plot(t[::5] / TS, E_ex, "k.", ms=4, label="exact entropy solution")
    axes[1].set_ylabel("energy  ∫u²/2 dx")
    axes[1].legend(fontsize=8)
    axes[2].plot(t / TS, hist["jump"])
    axes[2].set_ylabel("max interface jump\n(varies as shock crosses cells)")
    axes[2].set_xlabel("t / t_s")
    for ax in axes:
        ax.axvline(1.0, color="r", ls=":", lw=1)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Invariants: mass conserved; energy conserved until the shock, then dissipated")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return np.max(np.abs(np.array(hist["mass"]) - m0))


def plot_post_shock(rows, degrees, t_eval, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in degrees:
        sub = [r for r in rows if r["p"] == p]
        h = np.array([r["h"] for r in sub])
        ax.loglog(h, [r["L1"] for r in sub], "o-", color=COLORS[p], label=f"p={p} full domain")
        ax.loglog(h, [r["L1_away"] for r in sub], "s--", color=COLORS[p], label=f"p={p} |x - x_s| > 0.25")
    h = np.array([r["h"] for r in rows if r["p"] == degrees[0]])
    ax.loglog(h, 0.25 * h, "k:", lw=1, label="O(h)")
    ax.loglog(h, 0.05 * h ** 2, "k-.", lw=1, label="O(h²)")
    ax.set_xlabel("h")
    ax.set_ylabel("L1 error")
    ax.set_title(f"Post-shock L1 error at t = {t_eval:.1f} t_s (TVB-limited)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- report
def write_report(path, smooth, flux_cmp, post, jump, mass_drift, args):
    L = []
    L.append("# DG Burgers convergence and error report\n")
    L.append("Problem: u_t + (u²/2)_x = 0 on [0, 2π), periodic, u0 = 0.5 + sin x, "
             f"breaking time t_s = {TS:g}, shock forms at x = π + 0.5.\n")
    L.append(f"## Smooth regime: t = {args.t_smooth:g} t_s (no limiter, Godunov flux, SSP-RK3)\n")
    L.append("Reference: exact solution via characteristics (Newton/bisection to 1e-14). "
             "Norms use a (p+6)-point Gauss rule per cell; Linf also samples cell ends. "
             "For p ≥ 3, dt is reduced like h^((p+1)/3) so the 3rd-order time error does not "
             "hide the spatial order.\n")
    for p in args.degrees:
        L.append(f"\n**p = {p}** (expected order {p + 1})\n")
        L.append("| N | h | L1 | order | L2 | order | Linf | order | steps | wall [s] |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in [r for r in smooth if r["p"] == p]:
            L.append(f"| {r['N']} | {r['h']:.4f} | {r['L1']:.3e} | {fmt(r['order_L1'])} | "
                     f"{r['L2']:.3e} | {fmt(r['order_L2'])} | {r['Linf']:.3e} | {fmt(r['order_Linf'])} | "
                     f"{r['steps']} | {r['wall_s']:.3f} |")
    L.append(f"\nMax |mass drift| over all smooth runs: "
             f"{max(abs(r['mass_drift']) for r in smooth):.2e}\n")
    L.append("\n## Flux comparison (smooth, L2 error)\n")
    L.append("| p | N | Godunov | Rusanov | ratio R/G |")
    L.append("|---|---|---|---|---|")
    for g, r in flux_cmp:
        L.append(f"| {g['p']} | {g['N']} | {g['L2']:.3e} | {r['L2']:.3e} | {r['L2'] / g['L2']:.2f} |")
    L.append(f"\n## Post-shock regime: t = {args.t_shock:g} t_s (TVB minmod limiter, M = {args.tvb_M:g})\n")
    L.append(f"Reference: exact entropy solution (shock at x = π + 0.5t, jump [u] = {jump:.4f}). "
             "With a captured discontinuity the full-domain L1 error is O(h) with a constant that "
             "depends on where the shock sits inside its cell, so pairwise orders fluctuate; "
             "L1/([u] h) staying bounded is the robust check. 'Away' excludes |x - x_s| < 0.25.\n")
    L.append("| p | N | L1 full | order | L1/([u]h) | L1 away | order |")
    L.append("|---|---|---|---|---|---|---|")
    for r in post:
        L.append(f"| {r['p']} | {r['N']} | {r['L1']:.3e} | {fmt(r['order_L1'])} | {r['L1_scaled']:.3f} | "
                 f"{r['L1_away']:.3e} | {fmt(r['order_L1_away'])} |")
    L.append(f"\nMass drift through the shock (p=2, N=200, limited, t ≤ 2 t_s): max |Δmass| = {mass_drift:.2e}\n")
    L.append("\n## Figures\n")
    for f in ["convergence.png", "error_profile.png", "error_vs_time.png",
              "invariants.png", "post_shock_convergence.png"]:
        L.append(f"- `{f}`")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--t-smooth", type=float, default=0.5, help="evaluation time / t_s (must be < 1)")
    ap.add_argument("--t-shock", type=float, default=1.5, help="post-shock evaluation time / t_s")
    ap.add_argument("--tvb-M", type=float, default=5.0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_smooth = args.t_smooth * TS
    t_shock = args.t_shock * TS

    print("smooth convergence study ...")
    smooth = smooth_study(args.degrees, args.Ns, t_smooth)
    for r in smooth:
        r["t"] = args.t_smooth
    keys = ["p", "N", "h", "L1", "order_L1", "L2", "order_L2", "Linf", "order_Linf",
            "steps", "wall_s", "mass_drift"]
    with open(os.path.join(args.out, "convergence.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(smooth)
    for p in args.degrees:
        o = [r["order_L2"] for r in smooth if r["p"] == p][-1]
        print(f"  p={p}: finest-pair L2 order {o:.2f} (expected {p + 1})")

    print("flux comparison ...")
    flux_cmp = []
    for p in (1, 2):
        g = smooth_study([p], [40], t_smooth, "godunov")[0]
        r = smooth_study([p], [40], t_smooth, "rusanov")[0]
        flux_cmp.append((g, r))

    print("post-shock study ...")
    post, jump = post_shock_study([1, 2], [50, 100, 200, 400, 800], t_shock, args.tvb_M)

    print("plots ...")
    plot_convergence(smooth, args.degrees, os.path.join(args.out, "convergence.png"))
    plot_error_profile(t_smooth, os.path.join(args.out, "error_profile.png"))
    plot_error_vs_time(os.path.join(args.out, "error_vs_time.png"))
    mass_drift = plot_invariants(os.path.join(args.out, "invariants.png"), M=args.tvb_M)
    plot_post_shock(post, [1, 2], args.t_shock, os.path.join(args.out, "post_shock_convergence.png"))
    write_report(os.path.join(args.out, "convergence_report.md"), smooth, flux_cmp, post, jump,
                 mass_drift, args)
    print(f"done -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
