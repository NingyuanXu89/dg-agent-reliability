"""Optional local C accelerator via stdlib ctypes; NumPy remains the fallback.

Compilation is explicit (build), needs an existing C compiler, and only creates
ignored local artifacts. No Python dependency or external solver is installed.
"""
import ctypes as ct
import hashlib,os,shutil,subprocess
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).with_name('kernels.c')
CACHE=SOURCE.parent.parent/'outputs'/'khi'/'native'

def library_path():
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]
    return CACHE/f'euler_{digest}.dylib'

def build():
    compiler=shutil.which('cc') or shutil.which('clang')
    if compiler is None:return False
    target=library_path()
    if not target.exists():
        CACHE.mkdir(parents=True,exist_ok=True)
        try:
            subprocess.run([compiler,'-O3','-march=native','-ffp-contract=off','-shared','-fPIC',str(SOURCE),'-o',str(target)],check=True,capture_output=True)
        except (OSError,subprocess.CalledProcessError):return False
    return True

class NativeKernel:
    def __init__(self,solver):
        self.s=solver;self.lib=ct.CDLL(str(library_path()));p=solver.config.degree;m=p+1;q=p+2
        self.vb=np.einsum('aj,bi->abji',solver.b,solver.b).reshape(q*q,m*m)
        factor=solver.massfactor
        self.vx=(np.einsum('aj,bi->abji',solver.bw,solver.dw)*factor[None,None]/(2*solver.dx)).reshape(q*q,m*m)
        self.vy=(np.einsum('aj,bi->abji',solver.dw,solver.bw)*factor[None,None]/(2*solver.dy)).reshape(q*q,m*m)
        one=np.ones(m);sign=solver.parity
        self.fb=np.concatenate([np.einsum('nj,i->nji',solver.b,sign).reshape(q,m*m),np.einsum('nj,i->nji',solver.b,one).reshape(q,m*m),np.einsum('j,ni->nji',sign,solver.b).reshape(q,m*m),np.einsum('j,ni->nji',one,solver.b).reshape(q,m*m)])
        self.fx=(np.einsum('nj,i->nji',solver.bw,one)*factor[None]/(2*solver.dx)).reshape(q,m*m)
        self.fy=(np.einsum('j,ni->nji',one,solver.bw)*factor[None]/(2*solver.dy)).reshape(q,m*m)
        self.tables=[np.ascontiguousarray(v) for v in (self.vb,self.vx,self.vy,self.fb,self.fx,self.fy,solver.check)]
        self.work=np.empty(4*solver.config.nx*solver.config.ny*m*m*4,dtype=np.float64)
        self.speed_function=getattr(self.lib,f"dg_speed{p}");self.speed_function.restype=ct.c_double
        self.step_function=getattr(self.lib,f"dg_step{p}")
    @staticmethod
    def ptr(a):return a.ctypes.data_as(ct.POINTER(ct.c_double))
    def step(self,a,dt):
        from .core import StepRejected
        s=self.s;c=s.config;a=np.ascontiguousarray(a);out=np.empty_like(a)
        counts=np.zeros(3,dtype=np.int64);theta=ct.c_double(s.min_theta)
        code=self.step_function(self.ptr(a),self.ptr(out),self.ptr(self.work),ct.c_int(c.nx),ct.c_int(c.ny),ct.c_int(c.degree),ct.c_double(c.gamma),ct.c_int(c.flux=='llf'),*[self.ptr(v) for v in self.tables],ct.c_int(len(s.check)),ct.c_double(dt),ct.c_double(c.tvb),ct.c_int(c.limiting),ct.c_double(c.floor),counts.ctypes.data_as(ct.POINTER(ct.c_longlong)),ct.byref(theta))
        s.fallbacks+=int(counts[0]);s.limited+=int(counts[1]);s.scaled+=int(counts[2]);s.min_theta=theta.value
        if code:raise StepRejected('native stage has nonfinite values or inadmissible means')
        return out
    def stabilize(self,a):
        from .core import StepRejected
        s=self.s;c=s.config;a=np.ascontiguousarray(a).copy();counts=np.zeros(2,dtype=np.int64);theta=ct.c_double(s.min_theta)
        code=self.lib.dg_stabilize(self.ptr(a),ct.c_int(c.nx),ct.c_int(c.ny),ct.c_int(c.degree),ct.c_double(c.gamma),ct.c_double(c.tvb),ct.c_int(c.limiting),ct.c_double(c.floor),self.ptr(self.tables[-1]),ct.c_int(len(s.check)),counts.ctypes.data_as(ct.POINTER(ct.c_longlong)),counts[1:].ctypes.data_as(ct.POINTER(ct.c_longlong)),ct.byref(theta),self.ptr(self.work))
        s.limited+=int(counts[0]);s.scaled+=int(counts[1]);s.min_theta=theta.value
        if code:raise StepRejected('native stabilization failed')
        return a
    def timestep(self,a):
        s=self.s;c=s.config;a=np.ascontiguousarray(a)
        speed=self.speed_function(self.ptr(a),ct.c_int(c.nx),ct.c_int(c.ny),ct.c_int(c.degree),ct.c_double(c.gamma),self.ptr(self.tables[-1]),ct.c_int(len(s.check)))
        if not np.isfinite(speed) or speed<=0:
            from .core import StepRejected
            raise StepRejected('nonfinite or zero characteristic speed')
        return min(c.dt_cap,c.cfl/((2*c.degree+1)*speed))

def attach(solver):
    if os.environ.get('DG_EULER_NATIVE','1')!='0' and library_path().exists():
        try:kernel=NativeKernel(solver)
        except OSError:return
        solver.step=kernel.step;solver.timestep=kernel.timestep;solver.native=kernel
