"""Headless diagnostic plots and MP4/GIF export; preserve DG discontinuities."""
import os
from pathlib import Path
import shutil
import subprocess
import json
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dg-burgers-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np
from numpy.polynomial.legendre import legvander
from .core import DG, LENGTH
from .reference import exact_solution, shock_position, shock_foot


def sample_cells(dg, coefficients, points=18):
    xi = np.linspace(-1, 1, points)
    x = dg.centers[:, None]+dg.h/2*xi
    u = coefficients @ legvander(xi, dg.p).T
    # NaN separators prevent plotting an artificial line across cell jumps.
    xx = np.column_stack((x, np.full(len(x), np.nan))).ravel()
    uu = np.column_stack((u, np.full(len(u), np.nan))).ravel()
    return xx, uu


def reference_curve(t, origin=0., points=1000):
    """Sample in characteristic space (fast), keeping the entropy jump separate."""
    if t <= 1:
        xi = np.linspace(0, LENGTH, points)
        x = (xi+t+t*np.sin(xi)-origin) % LENGTH + origin
        u = 1+np.sin(xi)
        order = np.argsort(x)
        return x[order], u[order]
    q = shock_foot(float(t))
    pieces = []
    for a, b in ((0., q), (LENGTH-q, LENGTH)):
        xi = np.linspace(a, b, points//2)
        x = (xi+t+t*np.sin(xi)-origin) % LENGTH + origin
        u = 1+np.sin(xi)
        # Split again where this branch crosses the periodic boundary.
        cuts = np.flatnonzero(np.diff(x) < -LENGTH/2)+1
        for ids in np.split(np.arange(len(x)), cuts):
            if len(ids):
                pieces.append((x[ids], u[ids]))
    return (np.concatenate([np.r_[x, np.nan] for x,u in pieces]),
            np.concatenate([np.r_[u, np.nan] for x,u in pieces]))


def plot_diagnostics(result, rows, directory):
    directory = Path(directory)
    t = result.times
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    fields = [("mass_drift", "Mass drift"), ("entropy", "Quadratic entropy"),
              ("max_interface_jump", "Largest interface jump"),
              ("limiter_fraction", "Fraction of limited cell stages")]
    for ax,(key,label) in zip(axes.ravel(),fields):
        ax.plot(t, [r[key] for r in rows])
        ax.axvline(1, color="gray", ls="--", lw=1)
        ax.set(xlabel="Time", ylabel=label)
        ax.grid(alpha=.2)
    fig.savefig(directory/"diagnostics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for key in ("l1", "l2", "linf_sampled"):
        axes[0].semilogy(t, [max(r[key], 1e-16) for r in rows], label=key)
    axes[0].set(xlabel="Time", ylabel="Error", title="Independent entropy reference")
    axes[0].legend()
    axes[1].plot(t, [r["minimum"] for r in rows], label="Minimum")
    axes[1].plot(t, [r["maximum"] for r in rows], label="Maximum")
    axes[1].axhline(0,color="gray",ls=":")
    axes[1].axhline(2,color="gray",ls=":")
    axes[1].set(xlabel="Time", ylabel="u", title="Polynomial extrema")
    axes[1].legend()
    for ax in axes:
        ax.axvline(1,color="gray",ls="--",lw=1)
        ax.grid(alpha=.2)
    fig.savefig(directory/"errors_and_bounds.png", dpi=160)
    plt.close(fig)
    dg = DG(result.config)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax,target in zip(axes,(0.,1.,result.config.final_time)):
        i = int(np.argmin(abs(t-target)))
        ax.plot(*sample_cells(dg,result.coefficients[i]),color="tab:blue",label="DG")
        ax.plot(*reference_curve(float(t[i]),result.config.origin),color="black",ls="--",label="Reference")
        ax.set(xlabel="x",ylabel="u",title=f"t = {t[i]:.3f}",ylim=(-.15,2.15))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(directory/"solution_snapshots.png",dpi=160)
    plt.close(fig)


def plot_convergence(smooth, shock, directory):
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,key,label in zip(axes,("l1","l2","linf_sampled"),
                            (r"$L^1$ error",r"$L^2$ error",r"Sampled $L^\infty$ error")):
        for p in sorted(set(r["degree"] for r in smooth)):
            g = [r for r in smooth if r["degree"]==p]
            h,e = np.array([r["h"] for r in g]),np.array([r[key] for r in g])
            line, = ax.loglog(h,e,"o-",label=f"p={p}")
            ax.loglog(h,e[-1]*(h/h[-1])**(p+1),":",color=line.get_color(),alpha=.6)
        ax.set(xlabel="Cell width h",ylabel=label,title="Smooth solution at t=0.5")
        ticks=sorted(set(r["h"] for r in smooth))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{v:.3f}" for v in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="both",alpha=.2)
        ax.legend()
    fig.savefig(Path(directory)/"smooth_convergence.png",dpi=160)
    plt.close(fig)
    if shock:
        fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
        for aligned in (False,True):
            g=[r for r in shock if r["aligned"]==aligned]
            ax.loglog([2*np.pi/r["cells"] for r in g],[r["l1"] for r in g],"o-",
                      label="Shock aligned at final time" if aligned else "Shock inside cell at final time")
        ax.set(xlabel="Cell width h",ylabel="Integrated L1 error",title="Limited p=2 DG at t=1.5")
        ax.legend()
        ticks=sorted(set(2*np.pi/r["cells"] for r in shock))
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{v:.3f}" for v in ticks]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="both",alpha=.2)
        fig.savefig(Path(directory)/"postshock_convergence.png",dpi=160)
        plt.close(fig)


def render_movie(result, directory, fps=20):
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    if not Path(ffmpeg).is_file():
        raise RuntimeError("MP4 export requires FFmpeg on PATH")
    plt.rcParams["animation.ffmpeg_path"]=ffmpeg
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    dg=DG(result.config)
    fig,(ax,history)=plt.subplots(2,1,figsize=(10,6),height_ratios=(3,1),layout="constrained")
    numerical,=ax.plot([],[],color="tab:blue",lw=1.4,label=f"DG p={dg.p}, N={result.config.cells}")
    reference,=ax.plot([],[],color="black",ls="--",lw=1.2,label="Entropy reference")
    marker=ax.axvline(np.pi,color="tab:red",ls=":",lw=1.2,label="Predicted shock position (t >= 1)")
    annotation=ax.text(.02,.05,"",transform=ax.transAxes)
    ax.set(xlim=(dg.edges[0],dg.edges[-1]),ylim=(-.15,2.15),xlabel="x",ylabel="u(x,t)")
    ax.legend(loc="upper right",fontsize=9)
    ax.grid(alpha=.2)
    history.plot(result.times,[d["entropy"] for d in result.diagnostics],color="tab:purple")
    history.axvline(1,color="gray",ls="--",label="Shock formation t=1")
    cursor=history.axvline(0,color="tab:red")
    history.set(xlabel="Time",ylabel="Quadratic entropy",xlim=(0,result.config.final_time))
    history.grid(alpha=.2)
    history.legend(loc="lower left",fontsize=8)
    def update(i):
        t=float(result.times[i])
        numerical.set_data(*sample_cells(dg,result.coefficients[i]))
        reference.set_data(*reference_curve(t,result.config.origin))
        marker.set_xdata([shock_position(t,result.config.origin)]*2)
        marker.set_visible(t>=1)
        cursor.set_xdata([t,t])
        d=result.diagnostics[i]
        annotation.set_text(f"t = {t:.3f}    shock time = 1.000    mass drift = {d['mass_drift']:.2e}")
        ax.set_title("Inviscid Burgers: wave steepening and moving shock")
    mp4=directory/"burgers_shock.mp4"
    gif=directory/"burgers_shock.gif"
    writer=FFMpegWriter(fps=fps,codec="libx264",extra_args=["-pix_fmt","yuv420p","-crf","20"])
    with writer.saving(fig,str(mp4),dpi=120):
        for i in range(len(result.times)):
            update(i)
            writer.grab_frame()
    print(f"Wrote {mp4}",flush=True)
    writer=PillowWriter(fps=fps)
    with writer.saving(fig,str(gif),dpi=80):
        for i in range(len(result.times)):
            update(i)
            writer.grab_frame()
    for target in (0.,.75,1.,1.25,result.config.final_time):
        i=int(np.argmin(abs(result.times-target)))
        update(i)
        fig.savefig(directory/f"frame_t{result.times[i]:.2f}.png",dpi=120)
    plt.close(fig)
    ffprobe=shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    metadata=json.loads(subprocess.check_output([ffprobe,"-v","error","-select_streams","v:0",
        "-show_entries","stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
        "-of","json",str(mp4)],text=True))
    stream=metadata["streams"][0]
    if stream["codec_name"]!="h264" or int(stream["nb_frames"])!=len(result.times):
        raise RuntimeError("movie verification failed")
    (directory/"movie_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Verified H.264 movie: {stream['nb_frames']} frames, {metadata['format']['duration']} seconds",flush=True)
    return mp4,gif
