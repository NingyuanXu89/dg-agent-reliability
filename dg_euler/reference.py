"""Independent exact planar ideal-gas Riemann solution (shock/rarefaction)."""
import numpy as np
from scipy.optimize import brentq
from .core import conservative


def riemann(x,t,gamma=1.4,left=(1.,0.,1.),right=(.125,0.,.1),origin=.5,axis=0):
    def curve(p,state):
        rho,v,p0=state;c=np.sqrt(gamma*p0/rho)
        if p>p0:
            a=2/((gamma+1)*rho);b=(gamma-1)*p0/(gamma+1)
            return (p-p0)*np.sqrt(a/(p+b))
        return 2*c/(gamma-1)*((p/p0)**((gamma-1)/(2*gamma))-1)
    f=lambda p:curve(p,left)+curve(p,right)+right[1]-left[1]
    hi=max(left[2],right[2])
    while f(hi)<0:hi*=2
    ps=brentq(f,1e-14,hi,xtol=1e-14);vs=.5*(left[1]+right[1]+curve(ps,right)-curve(ps,left))
    x=np.asarray(x);xi=(x-origin)/t if t>0 else np.where(x<origin,-np.inf,np.inf)
    rho=np.empty_like(xi,dtype=float);v=rho.copy();p=rho.copy()
    for isleft,state in [(True,left),(False,right)]:
        r0,v0,p0=state;c0=np.sqrt(gamma*p0/r0);direction=-1 if isleft else 1
        mask=xi<=vs if isleft else xi>vs
        if ps>p0:
            rs=r0*((ps/p0)+(gamma-1)/(gamma+1))/((gamma-1)/(gamma+1)*(ps/p0)+1)
            speed=v0+direction*c0*np.sqrt((gamma+1)/(2*gamma)*ps/p0+(gamma-1)/(2*gamma))
            undist=xi<speed if isleft else xi>speed
            rho[mask]=np.where(undist,r0,rs)[mask];v[mask]=np.where(undist,v0,vs)[mask];p[mask]=np.where(undist,p0,ps)[mask]
        else:
            cs=c0*(ps/p0)**((gamma-1)/(2*gamma));rs=r0*(ps/p0)**(1/gamma)
            head=v0+direction*c0;tail=vs+direction*cs
            undist=xi<head if isleft else xi>head;star=xi>=tail if isleft else xi<=tail
            vf=2/(gamma+1)*(xi-direction*c0+.5*(gamma-1)*v0)
            cf=2/(gamma+1)*(c0-direction*.5*(gamma-1)*(v0-xi))
            rf=r0*np.maximum(cf/c0,0)**(2/(gamma-1));pf=p0*np.maximum(cf/c0,0)**(2*gamma/(gamma-1))
            rho[mask]=np.where(undist,r0,np.where(star,rs,rf))[mask]
            v[mask]=np.where(undist,v0,np.where(star,vs,vf))[mask]
            p[mask]=np.where(undist,p0,np.where(star,ps,pf))[mask]
    return conservative(rho,v if axis==0 else 0,v if axis==1 else 0,p,gamma)
