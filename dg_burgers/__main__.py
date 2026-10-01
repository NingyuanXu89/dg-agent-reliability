"""Run with python -m dg_burgers {solve,convergence,movie,demo}."""
import argparse
import json
from pathlib import Path
import numpy as np
from .core import Config, simulate
from .diagnostics import save_result
from .experiments import convergence, postshock, acceptance
from .plotting import plot_diagnostics, plot_convergence, render_movie


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("solve","convergence","movie","demo"):
        p=sub.add_parser(command)
        p.add_argument("--output",type=Path,default=Path("outputs"))
        if command!="convergence":
            p.add_argument("--cells",type=int,default=128)
            p.add_argument("--degree",type=int,choices=(1,2,3),default=2)
            p.add_argument("--final-time",type=float,default=1.5)
            p.add_argument("--cfl",type=float,default=.15)
            p.add_argument("--tvb-m",type=float,default=1.)
            p.add_argument("--no-limiter",action="store_true")
            p.add_argument("--frames",type=int,default=151)
            p.add_argument("--fps",type=int,default=20)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command in ("convergence","demo"):
        smooth=convergence(args.output)
        shock=postshock(args.output)
        report=acceptance(smooth,shock,args.output)
        plot_convergence(smooth,shock,args.output)
        print(json.dumps(report,indent=2),flush=True)
    if args.command in ("solve","movie","demo"):
        if args.frames<2 or args.fps<1:
            parser.error("frames must be >= 2 and fps >= 1")
        cfg=Config(cells=args.cells,degree=args.degree,final_time=args.final_time,
                   cfl=args.cfl,tvb_m=args.tvb_m,limiter=not args.no_limiter)
        times=np.linspace(0,cfg.final_time,args.frames) if cfg.final_time else np.array([0.])
        result=simulate(cfg,times=times)
        directory=args.output/"solution"
        rows=save_result(result,directory)
        plot_diagnostics(result,rows,directory)
        if args.command in ("movie","demo"):
            if cfg.final_time==0:
                parser.error("movie requires positive final-time")
            render_movie(result,directory,args.fps)
        print(f"Solved with {result.steps} steps; max mass drift = "
              f"{max(abs(d['mass_drift']) for d in rows):.3e}",flush=True)


if __name__=="__main__":
    main()
