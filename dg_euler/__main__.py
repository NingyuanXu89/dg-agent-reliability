import argparse,json,os
from pathlib import Path
from .core import EulerConfig,simulate
from .experiments import validate,khi_suite,report
from .plotting import movie


def main():
    parser=argparse.ArgumentParser(description='From-scratch periodic 2D Euler DG and tanh Kelvin–Helmholtz test')
    parser.add_argument('command',choices=['khi','validate','movie','demo'])
    parser.add_argument('--nx',type=int,default=64);parser.add_argument('--ny',type=int)
    parser.add_argument('--degree',type=int,default=2);parser.add_argument('--gamma',type=float,default=5/3)
    parser.add_argument('--final-time',type=float,default=3);parser.add_argument('--cfl',type=float,default=.15)
    parser.add_argument('--layer-width',type=float,default=.025);parser.add_argument('--perturbation',type=float,default=.01)
    parser.add_argument('--flux',choices=['hllc','llf'],default='hllc');parser.add_argument('--tvb',type=float,default=50)
    parser.add_argument('--output',default='outputs/khi');args=parser.parse_args()
    from .native import build
    try:
        if os.environ.get('DG_EULER_NATIVE','1')!='0':build()
    except (OSError,RuntimeError):pass
    if args.command=='demo' and args.final_time!=3:parser.error('demo uses final-time 3; use khi for other final times')
    cfg=EulerConfig(nx=args.nx,ny=args.ny or args.nx,degree=args.degree,gamma=args.gamma,final_time=args.final_time,cfl=args.cfl,layer_width=args.layer_width,perturbation=args.perturbation,flux=args.flux,tvb=args.tvb)
    if args.command=='khi':simulate(cfg,output=args.output,movie_fields=True,progress=True)
    elif args.command=='validate':print(json.dumps(validate(Path(args.output)/'validation'),indent=2))
    elif args.command=='movie':print(json.dumps(movie(args.output),indent=2))
    else:
        validate(Path(args.output)/'validation');khi_suite(args.output,cfg);movie(Path(args.output)/'n64')
        print(f'Complete deliverables: {args.output}/report.md')

if __name__=='__main__':main()
