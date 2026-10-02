"""Conservative tensor-product modal DG for periodic ideal-gas Euler."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander, legder, legval, legroots


def pressure(u, gamma):
    return (gamma-1)*(u[...,3]-.5*(u[...,1]**2+u[...,2]**2)/u[...,0])


def conservative(rho, vx, vy, p, gamma):
    rho,vx,vy,p=np.broadcast_arrays(rho,vx,vy,p)
    return np.stack((rho,rho*vx,rho*vy,p/(gamma-1)+.5*rho*(vx*vx+vy*vy)),axis=-1)


def primitive(u,gamma):
    return np.stack((u[...,0],u[...,1]/u[...,0],u[...,2]/u[...,0],pressure(u,gamma)),axis=-1)


def physical_flux(u,gamma,axis):
    p=pressure(u,gamma);v=u[...,axis+1]/u[...,0]
    f=u*v[...,None];f[...,axis+1]+=p;f[...,3]+=p*v
    return f


def numerical_flux(left,right,gamma,axis,choice='hllc'):
    """HLLC, with an admissibility/degeneracy fallback to local LF."""
    fl=physical_flux(left,gamma,axis);fr=physical_flux(right,gamma,axis)
    pl=pressure(left,gamma);pr=pressure(right,gamma)
    rl=left[...,0];rr=right[...,0];vl=left[...,axis+1]/rl;vr=right[...,axis+1]/rr
    cl=np.sqrt(gamma*pl/rl);cr=np.sqrt(gamma*pr/rr)
    speed=np.maximum(abs(vl)+cl,abs(vr)+cr)
    lf=.5*(fl+fr)-.5*speed[...,None]*(right-left)
    if choice=='llf':return lf,0
    sl=np.minimum(vl-cl,vr-cr);sr=np.maximum(vl+cl,vr+cr)
    den=rl*(sl-vl)-rr*(sr-vr)
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        sm=(pr-pl+rl*vl*(sl-vl)-rr*vr*(sr-vr))/den
        def star(u,r,v,p,s):
            rs=r*(s-v)/(s-sm);out=u*(rs/r)[...,None]
            out[...,axis+1]=rs*sm
            out[...,3]=rs*(u[...,3]/r+(sm-v)*(sm+p/(r*(s-v))))
            return out
        ls=star(left,rl,vl,pl,sl);rs=star(right,rr,vr,pr,sr)
        result=np.where((sm>=0)[...,None],fl+sl[...,None]*(ls-left),fr+sr[...,None]*(rs-right))
        result=np.where((sl>=0)[...,None],fl,np.where((sr<=0)[...,None],fr,result))
        bad=(abs(den)<1e-14)|(abs(sl-sm)<1e-14)|(abs(sr-sm)<1e-14)
        bad|=~np.all(np.isfinite(ls)&np.isfinite(rs)&np.isfinite(result),axis=-1)
        bad|=(pl+rl*(sl-vl)*(sm-vl)<=0)
        bad|=(ls[...,0]<=0)|(rs[...,0]<=0)|(pressure(ls,gamma)<=0)|(pressure(rs,gamma)<=0)
    return np.where(bad[...,None],lf,result),int(np.count_nonzero(bad))


@dataclass
class EulerConfig:
    nx:int=64
    ny:int=64
    degree:int=2
    gamma:float=5/3
    final_time:float=3.
    cfl:float=.15
    tvb:float=50.
    limiting:bool=True
    floor:float=1e-12
    layer_width:float=.025
    perturbation:float=.01
    flux:str='hllc'
    dt_cap:float=float('inf')
    def __post_init__(self):
        if any(not isinstance(v,(int,np.integer)) for v in (self.nx,self.ny,self.degree)) or self.nx<2 or self.ny<2 or self.degree not in (1,2):raise ValueError('nx,ny >= 2; degree must be 1 or 2')
        if not np.isfinite([self.gamma,self.final_time,self.cfl,self.tvb,self.floor,self.layer_width,self.perturbation]).all():raise ValueError('nonfinite configuration')
        if self.gamma<=1 or self.final_time<0 or not 0<self.cfl<=.5 or self.tvb<0 or self.floor<=0 or self.layer_width<=0 or self.dt_cap<=0 or np.isnan(self.dt_cap):raise ValueError('invalid configuration')
        if self.flux not in ('hllc','llf'):raise ValueError('unknown flux')


class StepRejected(RuntimeError):pass


class EulerDG:
    def __init__(self,config):
        self.config=config;p=config.degree;self.dx=1/config.nx;self.dy=1/config.ny
        self.z,self.w=leggauss(p+2);self.b=legvander(self.z,p)
        self.db=np.column_stack([legval(self.z,legder(np.eye(p+1)[j])) for j in range(p+1)])
        self.bw=self.b*self.w[:,None];self.dw=self.db*self.w[:,None]
        self.parity=(-1.)**np.arange(p+1);self.massfactor=np.outer(2*np.arange(p+1)+1,2*np.arange(p+1)+1)
        lob=np.r_[-1.,legroots(legder(np.eye(p+2)[-1])),1.]
        # Union of tensor Gauss/Lobatto and all face Gauss nodes.
        gy,gx=np.meshgrid(self.z,self.z,indexing='ij');ly,lx=np.meshgrid(lob,lob,indexing='ij')
        xx=np.r_[gx.ravel(),lx.ravel(),self.z,self.z,np.full(p+2,-1),np.full(p+2,1)]
        yy=np.r_[gy.ravel(),ly.ravel(),np.full(p+2,-1),np.full(p+2,1),self.z,self.z]
        self.check=np.einsum('nj,ni->nji',legvander(yy,p),legvander(xx,p)).reshape(len(xx),-1)
        self.fallbacks=0;self.limited=0;self.scaled=0;self.min_theta=1.;self.rejected=0
    def reconstruct(self,a,x=None,y=None):
        bx=self.b if x is None else legvander(x,self.config.degree)
        by=self.b if y is None else legvander(y,self.config.degree)
        return np.einsum('...jiv,aj,bi->...abv',a,by,bx,optimize=True)
    def project(self,initial):
        """Quadrature L² projection; reference-cell mass is 4/[(2i+1)(2j+1)]."""
        x=(np.arange(self.config.nx)[:,None]+.5*(1+self.z))/self.config.nx
        y=(np.arange(self.config.ny)[:,None]+.5*(1+self.z))/self.config.ny
        u=np.broadcast_to(initial(x[None,:,None,:],y[:,None,:,None]),(self.config.ny,self.config.nx,len(self.z),len(self.z),4))
        return np.einsum('yxabv,aj,bi->yxjiv',u,self.bw,self.bw,optimize=True)*self.massfactor[None,None,:,:,None]/4
    def traces(self,a):
        xr=np.einsum('yxjiv,aj->yxav',a,self.b,optimize=True)
        xl=np.einsum('yxjiv,i,aj->yxav',a,self.parity,self.b,optimize=True)
        yr=np.einsum('yxjiv,bi->yxbv',a,self.b,optimize=True)
        yl=np.einsum('yxjiv,j,bi->yxbv',a,self.parity,self.b,optimize=True)
        return xl,xr,yl,yr
    def residual(self,a):
        """Weak volume derivative minus outward shared face flux, divided by mass."""
        u=self.reconstruct(a);f=physical_flux(u,self.config.gamma,0);g=physical_flux(u,self.config.gamma,1)
        rx=np.einsum('yxabv,aj,bi->yxjiv',f,self.bw,self.dw,optimize=True)
        ry=np.einsum('yxabv,aj,bi->yxjiv',g,self.dw,self.bw,optimize=True)
        xl,xr,yl,yr=self.traces(a)
        fx,nx=numerical_flux(xr,np.roll(xl,-1,axis=1),self.config.gamma,0,self.config.flux)
        fy,ny=numerical_flux(yr,np.roll(yl,-1,axis=0),self.config.gamma,1,self.config.flux)
        self.fallbacks+=nx+ny
        ix=np.einsum('yxav,aj->yxjv',fx,self.bw,optimize=True)
        iy=np.einsum('yxbv,bi->yxiv',fy,self.bw,optimize=True)
        rx-=ix[:,:,:,None,:];rx+=np.roll(ix,1,axis=1)[:,:,:,None,:]*self.parity[None,None,None,:,None]
        ry-=iy[:,:,None,:,:];ry+=np.roll(iy,1,axis=0)[:,:,None,:,:]*self.parity[None,None,:,None,None]
        return (rx/(2*self.dx)+ry/(2*self.dy))*self.massfactor[None,None,:,:,None]
    def check_values(self,a):
        return np.matmul(self.check,a.reshape(self.config.ny,self.config.nx,-1,4))
    def admissible_means(self,a):
        m=a[:,:,0,0]
        if not np.isfinite(a).all() or np.any(m[...,0]<=self.config.floor):raise StepRejected('nonfinite coefficients or inadmissible conservative mean')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):p=pressure(m,self.config.gamma)
        if not np.isfinite(p).all() or np.any(p<=self.config.floor):raise StepRejected('nonfinite pressure or inadmissible conservative mean')
    @staticmethod
    def minmod(a,b,c):
        same=(np.sign(a)==np.sign(b))&(np.sign(a)==np.sign(c))
        return np.where(same,np.sign(a)*np.minimum(abs(a),np.minimum(abs(b),abs(c))),0)
    def tvb_limit(self,a):
        """Directional TVB detector and conservative componentwise slope limiting."""
        m=a[:,:,0,0].copy();flag=np.zeros(m.shape[:2],bool);linear=[]
        for axis,h,index in [(1,self.dx,(0,1)),(0,self.dy,(1,0))]:
            dl=m-np.roll(m,1,axis=axis);dr=np.roll(m,-1,axis=axis)-m
            scale=np.maximum(1,np.maximum(abs(m),np.maximum(abs(np.roll(m,1,axis=axis)),abs(np.roll(m,-1,axis=axis)))))
            modes=a[:,:,0,:,:] if axis==1 else a[:,:,:,0,:]
            hi=np.sum(modes,axis=2)-m;lo=m-np.sum(modes*self.parity[None,None,:,None],axis=2)
            for dev in (hi,lo):
                limited=self.minmod(dev,dl,dr)
                flag|=np.any((abs(dev)>self.config.tvb*h*h*scale)&(abs(dev-limited)>1e-12),axis=-1)
            linear.append(self.minmod(a[:,:,index[0],index[1]],dl,dr))
        if flag.any():
            a[flag]=0;a[:,:,0,0][flag]=m[flag]
            a[:,:,0,1][flag]=linear[0][flag];a[:,:,1,0][flag]=linear[1][flag]
        self.limited+=int(flag.sum());return a
    def positivity(self,a):
        """Contract higher modes toward admissible means; never clip conservative means."""
        self.admissible_means(a);mean=a[:,:,0,0].copy()
        with np.errstate(over='ignore',invalid='ignore'):u=self.check_values(a)
        if not np.isfinite(u).all():raise StepRejected('nonfinite reconstructed state')
        rho=u[...,0].min(axis=-1)
        theta=np.where(rho<self.config.floor,np.minimum(1,(mean[...,0]-self.config.floor)/np.maximum(mean[...,0]-rho,1e-300)),1.)
        a*=np.where(theta<1,theta*.999999,theta)[:,:,None,None,None];a[:,:,0,0]=mean
        u=self.check_values(a) if np.any(theta<1) else u
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):pcheck=pressure(u,self.config.gamma)
        if not np.isfinite(pcheck).all():raise StepRejected('nonfinite reconstructed pressure')
        bad=pcheck.min(axis=-1)<self.config.floor
        if bad.any():
            ub=u[bad];mb=mean[bad,None,:];lo=np.zeros(len(ub));hi=np.ones(len(ub))
            for _ in range(42):
                mid=(lo+hi)/2;trial=mb+mid[:,None,None]*(ub-mb)
                ok=np.all(pressure(trial,self.config.gamma)>=self.config.floor,axis=1)
                lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
            a[bad]*=(.999999*lo)[:,None,None,None];a[:,:,0,0]=mean;theta[bad]*=.999999*lo
        self.scaled+=int(np.count_nonzero(theta<1));self.min_theta=min(self.min_theta,float(theta.min()))
        if not np.isfinite(a).all():raise StepRejected('nonfinite stabilized state')
        if np.any(theta<1):
            checked=self.check_values(a);p=pressure(checked,self.config.gamma)
            if np.any(checked[...,0]<self.config.floor) or np.any(p<self.config.floor) or not np.isfinite(p).all():raise StepRejected('sampled positivity failed after conservative scaling')
        return a
    def stabilize(self,a):
        self.admissible_means(a)
        if self.config.limiting:a=self.tvb_limit(a)
        return self.positivity(a)
    def timestep(self,a):
        u=self.check_values(a);rho=u[...,0];c=np.sqrt(self.config.gamma*pressure(u,self.config.gamma)/rho)
        speed=np.max((abs(u[...,1])/rho+c)/self.dx+(abs(u[...,2])/rho+c)/self.dy)
        if not np.isfinite(speed) or speed<=0:raise StepRejected('nonfinite or zero characteristic speed')
        return min(self.config.dt_cap,self.config.cfl/((2*self.config.degree+1)*speed))
    def step(self,a,dt):
        """Three SSPRK3 stages, with stabilization after each forward-Euler combination."""
        b=self.stabilize(a+dt*self.residual(a))
        c=self.stabilize(.75*a+.25*(b+dt*self.residual(b)))
        return self.stabilize(a/3+2*(c+dt*self.residual(c))/3)
    def advance(self,a,dt):
        for attempt in range(9):
            try:return self.step(a,dt),dt
            except StepRejected:
                self.rejected+=1;dt*=.5
        raise StepRejected('eight retries exhausted')
    def fields(self,a,subcells=3):
        z=(np.arange(subcells)+.5)*2/subcells-1;b=legvander(z,self.config.degree)
        db=np.column_stack([legval(z,legder(np.eye(self.config.degree+1)[j])) for j in range(self.config.degree+1)])
        u=self.reconstruct(a,z,z)
        ux=np.einsum('yxjiv,aj,bi->yxabv',a,b,db,optimize=True)*2/self.dx
        uy=np.einsum('yxjiv,aj,bi->yxabv',a,db,b,optimize=True)*2/self.dy
        rho=u[...,0];vort=(ux[...,2]*rho-u[...,2]*ux[...,0]-uy[...,1]*rho+u[...,1]*uy[...,0])/rho**2
        flatten=lambda v:v.transpose(0,2,1,3).reshape(self.config.ny*subcells,self.config.nx*subcells)
        return flatten(rho).astype('float32'),flatten(vort).astype('float32')
    def diagnostics(self,a,time,initial):
        totals=a[:,:,0,0].sum(axis=(0,1))*self.dx*self.dy;u=self.reconstruct(a);checks=self.check_values(a)
        weights=self.w[:,None]*self.w[None,:];x=(np.arange(self.config.nx)[:,None]+.5*(1+self.z))*self.dx
        mode=np.sum(u[...,2]*np.exp(-4j*np.pi*x)[None,:,None,:]*weights[None,None])*self.dx*self.dy/4
        ky=np.sum(.5*u[...,2]**2/u[...,0]*weights[None,None])*self.dx*self.dy/4
        return dict(time=float(time),totals=totals.tolist(),drift=float(np.max(abs(totals-initial)/np.maximum(1,abs(initial)))),min_density=float(checks[...,0].min()),min_pressure=float(pressure(checks,self.config.gamma).min()),transverse_kinetic=float(ky),mode_amplitude=float(2*abs(mode)/totals[0]),limited=self.limited,positivity_scaled=self.scaled,min_theta=self.min_theta,flux_fallbacks=self.fallbacks,rejected_steps=self.rejected)


def khi_initial(config):
    def initial(x,y):
        s=.5*(1+np.tanh(-np.cos(2*np.pi*y)/(2*np.pi*config.layer_width)))
        return conservative(1+s,.5-s,config.perturbation*np.sin(4*np.pi*x),2.5,config.gamma)
    return initial


def simulate(config,initial=None,times=None,output=None,movie_fields=False,progress=False,accelerated=True,resume=False):
    solver=EulerDG(config)
    if accelerated:
        from .native import attach
        attach(solver)
    a=solver.positivity(solver.project(initial or khi_initial(config)))
    times=(np.linspace(0,config.final_time,151) if config.final_time>0 else np.array([0.])) if times is None else np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<1 or not np.isfinite(times).all() or times[0]!=0 or np.any(np.diff(times)<=0) or times[-1]>config.final_time+1e-12:raise ValueError('output times must increase from zero within final_time')
    out=Path(output) if output else None
    if out:
        out.mkdir(parents=True,exist_ok=True);saved_config=asdict(config);saved_config['dt_cap']=None if np.isinf(config.dt_cap) else config.dt_cap
        if resume and (out/'output_times.json').exists():
            previous_times=np.asarray(json.loads((out/'output_times.json').read_text()))
            if previous_times.shape!=times.shape or not np.allclose(previous_times,times,rtol=0,atol=1e-12):raise ValueError('cannot resume with a different output schedule')
        if resume and (out/'config.json').exists():
            previous=json.loads((out/'config.json').read_text());previous.pop('backend',None)
            if previous!=saved_config:raise ValueError('cannot resume with a different configuration')
        saved_config['backend']='native' if hasattr(solver,'native') else 'numpy'
        (out/'config.json').write_text(json.dumps(saved_config,indent=2,allow_nan=False));(out/'frames').mkdir(exist_ok=True)
        (out/'output_times.json').write_text(json.dumps(times.tolist()))
    t=0.;steps=0;history=[];initial_totals=a[:,:,0,0].sum(axis=(0,1))*solver.dx*solver.dy
    start_frame=0
    if resume and out and (out/'diagnostics.json').exists():
        saved=json.loads((out/'diagnostics.json').read_text())
        candidates=sorted(out.glob('checkpoint_*.npz'))
        if candidates:
            with np.load(candidates[-1]) as checkpoint:
                a=checkpoint['coefficients'].copy();t=float(checkpoint['time'])
            if a.shape!=(config.ny,config.nx,config.degree+1,config.degree+1,4):raise ValueError('checkpoint shape/configuration mismatch')
            if not np.any(np.isclose(times,t,rtol=0,atol=1e-12)):raise ValueError('checkpoint time is not in output schedule')
            solver.admissible_means(a)
            history=[d for d in saved if d['time']<=t+1e-12];last=history[-1];steps=last['steps']
            initial_totals=np.array(history[0]['totals']);start_frame=int(np.argmin(abs(times-t)))+1
            solver.fallbacks=last['flux_fallbacks'];solver.limited=last['limited'];solver.scaled=last['positivity_scaled'];solver.min_theta=last['min_theta'];solver.rejected=last['rejected_steps']
            if progress:print(f'Resume {config.nx}² from t={t:.3f}, frame={start_frame-1}',flush=True)
    for frame,target in enumerate(times):
        if frame<start_frame:continue
        while t<target-1e-14:
            dt=None
            try:
                dt=min(solver.timestep(a),target-t)
                if not np.isfinite(dt) or dt<=0:raise StepRejected('invalid time step')
                a,dt=solver.advance(a,dt)
            except StepRejected as exc:
                if out:
                    np.savez_compressed(out/'failure_state.npz',coefficients=a,time=t)
                    (out/'failure.json').write_text(json.dumps(dict(time=t,target=float(target),dt=dt,error=str(exc),steps=steps,rejected_steps=solver.rejected,mean_density_min=float(a[:,:,0,0,0].min()),mean_pressure_min=float(pressure(a[:,:,0,0],config.gamma).min())),indent=2))
                raise
            t+=dt;steps+=1
        diag=solver.diagnostics(a,t,initial_totals);diag['steps']=steps;history.append(diag)
        if out:
            if movie_fields:
                rho,vort=solver.fields(a);np.savez_compressed(out/'frames'/f'{frame:04d}.npz',density=rho,vorticity=vort,time=t,drift=diag['drift'])
            np.savez_compressed(out/f'means_{frame:04d}.npz',means=a[:,:,0,0],time=t)
            if frame in (0,len(times)//6,len(times)//3,2*len(times)//3,len(times)-1):np.savez_compressed(out/f'checkpoint_{frame:04d}.npz',coefficients=a,time=t)
            (out/'diagnostics.json').write_text(json.dumps(history,indent=2))
        if progress and frame%10==0:print(f'{config.nx}x{config.ny} p={config.degree}: t={t:.3f}, steps={steps}, A2={diag["mode_amplitude"]:.5g}, drift={diag["drift"]:.2g}',flush=True)
    if out:np.savez_compressed(out/'final_state.npz',coefficients=a,time=t)
    return dict(config=config,solver=solver,coefficients=a,history=history,times=times)
