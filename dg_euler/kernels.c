/* Optional from-scratch Euler DG kernels. No external numerical library.
 * Coefficients: [cell_y,cell_x,mode_y,mode_x,conserved_variable].
 * All integrals, HLLC fluxes, limiters and SSPRK3 stages mirror core.py.
 */
#include <math.h>
#include <float.h>
#include <stddef.h>
#include <string.h>

static double gas_pressure(const double *u,double g) {
    return (g-1)*(u[3]-.5*(u[1]*u[1]+u[2]*u[2])/u[0]);
}
static void flux(const double *u,double g,int axis,double *f) {
    double p=gas_pressure(u,g),v=u[axis+1]/u[0];
    for(int k=0;k<4;k++) f[k]=u[k]*v;
    f[axis+1]+=p;f[3]+=p*v;
}
static void hllc(const double *l,const double *r,double g,int axis,int use_lf,double *f,long long *fallback) {
    double fl[4],fr[4],ls[4],rs[4];flux(l,g,axis,fl);flux(r,g,axis,fr);
    double pl=gas_pressure(l,g),pr=gas_pressure(r,g),vl=l[axis+1]/l[0],vr=r[axis+1]/r[0];
    double cl=sqrt(g*pl/l[0]),cr=sqrt(g*pr/r[0]);
    double sl=fmin(vl-cl,vr-cr),sr=fmax(vl+cl,vr+cr);
    double den=l[0]*(sl-vl)-r[0]*(sr-vr);
    double sm=(pr-pl+l[0]*vl*(sl-vl)-r[0]*vr*(sr-vr))/den;
    int bad=fabs(den)<1e-14 || fabs(sl-sm)<1e-14 || fabs(sr-sm)<1e-14;
    double dl=l[0]*(sl-vl)/(sl-sm),dr=r[0]*(sr-vr)/(sr-sm);
    for(int k=0;k<4;k++){ls[k]=l[k]*(dl/l[0]);rs[k]=r[k]*(dr/r[0]);}
    ls[axis+1]=dl*sm;rs[axis+1]=dr*sm;
    ls[3]=dl*(l[3]/l[0]+(sm-vl)*(sm+pl/(l[0]*(sl-vl))));
    rs[3]=dr*(r[3]/r[0]+(sm-vr)*(sm+pr/(r[0]*(sr-vr))));
    bad=bad || dl<=0 || dr<=0 || gas_pressure(ls,g)<=0 || gas_pressure(rs,g)<=0 || pl+l[0]*(sl-vl)*(sm-vl)<=0;
    for(int k=0;k<4;k++) {
        if(sl>=0) f[k]=fl[k];
        else if(sr<=0) f[k]=fr[k];
        else if(sm>=0) f[k]=fl[k]+sl*(ls[k]-l[k]);
        else f[k]=fr[k]+sr*(rs[k]-r[k]);
        bad=bad || !isfinite(ls[k]) || !isfinite(rs[k]) || !isfinite(f[k]);
    }
    if(bad || use_lf) {
        double speed=fmax(fabs(vl)+cl,fabs(vr)+cr);
        for(int k=0;k<4;k++)f[k]=.5*(fl[k]+fr[k])-.5*speed*(r[k]-l[k]);
        if(bad && !use_lf)(*fallback)++;
    }
}
static inline __attribute__((always_inline)) void eval(const double *a,const double *basis,int modes,double *u) {
    for(int k=0;k<4;k++)u[k]=0;
    for(int m=0;m<modes;m++)for(int k=0;k<4;k++)u[k]+=basis[m]*a[m*4+k];
}
static inline __attribute__((always_inline)) void residual_impl(const double *a,double *rhs,int nx,int ny,int degree,double g,int use_lf,
                 const double *vb,const double *vx,const double *vy,const double *fb,
                 const double *fx,const double *fy,long long *fallback) {
    int m=degree+1,modes=m*m,q=degree+2,stride=modes*4,cells=nx*ny;
    memset(rhs,0,(size_t)cells*stride*sizeof(double));
    for(int c=0;c<cells;c++) {
        const double *ac=a+c*stride;double *rc=rhs+c*stride;
        for(int n=0;n<q*q;n++) {
            double u[4],f[4],h[4];eval(ac,vb+n*modes,modes,u);flux(u,g,0,f);flux(u,g,1,h);
            for(int mode=0;mode<modes;mode++)for(int k=0;k<4;k++)
                rc[mode*4+k]+=vx[n*modes+mode]*f[k]+vy[n*modes+mode]*h[k];
        }
    }
    /* One flux per shared right/top interface, applied to both adjacent cells. */
    for(int y=0;y<ny;y++)for(int x=0;x<nx;x++)for(int axis=0;axis<2;axis++) {
        int c=y*nx+x,neighbor=axis==0 ? y*nx+(x+1)%nx : ((y+1)%ny)*nx+x;
        for(int n=0;n<q;n++) {
            double l[4],r[4],f[4];
            eval(a+c*stride,fb+((axis*2+1)*q+n)*modes,modes,l);
            eval(a+neighbor*stride,fb+((axis*2)*q+n)*modes,modes,r);
            hllc(l,r,g,axis,use_lf,f,fallback);
            const double *weights=axis==0 ? fx+n*modes : fy+n*modes;
            for(int j=0;j<m;j++)for(int i=0;i<m;i++) {
                int mode=j*m+i;double factor=weights[mode],sign=((axis==0?i:j)%2) ? -1:1;
                for(int k=0;k<4;k++) {
                    rhs[c*stride+mode*4+k]-=factor*f[k];
                    rhs[neighbor*stride+mode*4+k]+=sign*factor*f[k];
                }
            }
        }
    }
}
static double minmod(double a,double b,double c) {
    if(a*b<=0 || a*c<=0)return 0;
    return copysign(fmin(fabs(a),fmin(fabs(b),fabs(c))),a);
}
static inline __attribute__((always_inline)) int stabilize(double *a,int nx,int ny,int degree,double g,double tvb,int limiting,double floor,
                     const double *checks,int ncheck,long long *limited,long long *scaled,double *min_theta,double *means) {
    int m=degree+1,modes=m*m,stride=4*modes,cells=nx*ny;
    for(int c=0;c<cells;c++) {
        double *ac=a+c*stride;
        for(int k=0;k<stride;k++)if(!isfinite(ac[k]))return 1;
        if(ac[0]<=floor || !isfinite(gas_pressure(ac,g)) || gas_pressure(ac,g)<=floor)return 1;
        memcpy(means+c*4,ac,4*sizeof(double));
    }
    if(limiting)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++) {
        int cell=y*nx+x;double *ac=a+cell*stride,*mean=means+cell*4;int flag=0;double linear[2][4];
        for(int axis=0;axis<2;axis++) {
            int prev=axis==0 ? y*nx+(x+nx-1)%nx : ((y+ny-1)%ny)*nx+x;
            int next=axis==0 ? y*nx+(x+1)%nx : ((y+1)%ny)*nx+x;
            double h=1./(axis==0?nx:ny);
            for(int k=0;k<4;k++) {
                double dl=mean[k]-means[prev*4+k],dr=means[next*4+k]-mean[k],hi=0,lo=0;
                double scale=fmax(1,fmax(fabs(mean[k]),fmax(fabs(means[prev*4+k]),fabs(means[next*4+k]))));
                for(int j=1;j<m;j++) {
                    double coef=ac[(axis==0 ? j : j*m)*4+k];hi+=coef;lo-=(j%2?-1:1)*coef;
                }
                if((fabs(hi)>tvb*h*h*scale && fabs(hi-minmod(hi,dl,dr))>1e-12) ||
                   (fabs(lo)>tvb*h*h*scale && fabs(lo-minmod(lo,dl,dr))>1e-12))flag=1;
                linear[axis][k]=minmod(ac[(axis==0?1:m)*4+k],dl,dr);
            }
        }
        if(flag) {
            memset(ac+4,0,(stride-4)*sizeof(double));
            for(int k=0;k<4;k++){ac[4+k]=linear[0][k];ac[m*4+k]=linear[1][k];}
            (*limited)++;
        }
    }
    for(int cell=0;cell<cells;cell++) {
        double *ac=a+cell*stride,*mean=means+cell*4,theta=1,minrho=INFINITY,minpressure=INFINITY,u[4];
        /* |P_j(xi)P_i(eta)|<=1: sufficient conservative component bounds.
         * If these already certify all points, theta=1 without reconstruction.
         * Otherwise perform the original Gauss/face/Lobatto checks and scaling. */
        double deviation[4]={0,0,0,0};
        for(int mode=1;mode<modes;mode++)for(int k=0;k<4;k++)deviation[k]+=fabs(ac[mode*4+k]);
        double rho_lower=mean[0]-deviation[0],energy_lower=mean[3]-deviation[3];
        double mx_upper=fabs(mean[1])+deviation[1],my_upper=fabs(mean[2])+deviation[2];
        double pressure_lower=(g-1)*(energy_lower-.5*(mx_upper*mx_upper+my_upper*my_upper)/rho_lower);
        double margin=128*DBL_EPSILON*fmax(1,fabs(mean[3])+deviation[3]);
        int bounds_finite=1;for(int k=0;k<4;k++)if(!isfinite(fabs(mean[k])+deviation[k]))bounds_finite=0;
        if(bounds_finite && rho_lower>floor+margin && pressure_lower>floor+margin)continue;
        for(int n=0;n<ncheck;n++){
            eval(ac,checks+n*modes,modes,u);for(int k=0;k<4;k++)if(!isfinite(u[k]))return 1;
            minrho=fmin(minrho,u[0]);
            if(u[0]>floor){double pc=gas_pressure(u,g);if(!isfinite(pc))return 1;minpressure=fmin(minpressure,pc);}
        }
        if(minrho<floor) {
            theta=fmin(1,(mean[0]-floor)/(mean[0]-minrho));
            for(int k=4;k<stride;k++)ac[k]*=theta*.999999;
        }
        int bad=minpressure<floor;
        if(theta<1){bad=0;for(int n=0;n<ncheck;n++){eval(ac,checks+n*modes,modes,u);if(gas_pressure(u,g)<floor)bad=1;}}
        if(bad) {
            double lo=0,hi=1;
            for(int iter=0;iter<42;iter++) {
                double mid=.5*(lo+hi);int ok=1;
                for(int n=0;n<ncheck;n++) {
                    eval(ac,checks+n*modes,modes,u);
                    for(int k=0;k<4;k++)u[k]=mean[k]+mid*(u[k]-mean[k]);
                    if(gas_pressure(u,g)<floor || !isfinite(gas_pressure(u,g)))ok=0;
                }
                if(ok)lo=mid;else hi=mid;
            }
            double factor=.999999*lo;theta*=factor;
            for(int k=4;k<stride;k++)ac[k]*=factor;
        }
        if(theta<1){
            (*scaled)++;*min_theta=fmin(*min_theta,theta);
            for(int n=0;n<ncheck;n++){eval(ac,checks+n*modes,modes,u);double pcheck=gas_pressure(u,g);if(u[0]<floor || pcheck<floor || !isfinite(pcheck))return 1;}
        }
        for(int k=0;k<stride;k++)if(!isfinite(ac[k]))return 1;
    }
    return 0;
}
static inline __attribute__((always_inline)) double speed_impl(const double *a,int nx,int ny,int degree,double g,const double *checks,int ncheck) {
    int modes=(degree+1)*(degree+1),stride=modes*4;double maximum=0;
    for(int cell=0;cell<nx*ny;cell++)for(int n=0;n<ncheck;n++) {
        double u[4];eval(a+cell*stride,checks+n*modes,modes,u);
        for(int k=0;k<4;k++)if(!isfinite(u[k]))return NAN;
        double p=gas_pressure(u,g);if(u[0]<=0 || !isfinite(p) || p<=0)return NAN;
        double c=sqrt(g*p/u[0]);
        maximum=fmax(maximum,(fabs(u[1])/u[0]+c)*nx+(fabs(u[2])/u[0]+c)*ny);
    }
    return maximum;
}
static inline __attribute__((always_inline)) int step_impl(const double *a,double *out,double *work,int nx,int ny,int degree,double g,int use_lf,
            const double *vb,const double *vx,const double *vy,const double *fb,const double *fx,const double *fy,
            const double *checks,int ncheck,double dt,double tvb,int limiting,double floor,long long *counts,double *min_theta) {
    size_t size=(size_t)nx*ny*(degree+1)*(degree+1)*4;
    double *b=work,*c=work+size,*rhs=work+2*size,*means=work+3*size;
    residual_impl(a,rhs,nx,ny,degree,g,use_lf,vb,vx,vy,fb,fx,fy,counts);
    for(size_t k=0;k<size;k++)b[k]=a[k]+dt*rhs[k];
    if(stabilize(b,nx,ny,degree,g,tvb,limiting,floor,checks,ncheck,counts+1,counts+2,min_theta,means))return 1;
    residual_impl(b,rhs,nx,ny,degree,g,use_lf,vb,vx,vy,fb,fx,fy,counts);
    for(size_t k=0;k<size;k++)c[k]=.75*a[k]+.25*(b[k]+dt*rhs[k]);
    if(stabilize(c,nx,ny,degree,g,tvb,limiting,floor,checks,ncheck,counts+1,counts+2,min_theta,means))return 1;
    residual_impl(c,rhs,nx,ny,degree,g,use_lf,vb,vx,vy,fb,fx,fy,counts);
    for(size_t k=0;k<size;k++)out[k]=a[k]/3+2*(c[k]+dt*rhs[k])/3;
    return stabilize(out,nx,ny,degree,g,tvb,limiting,floor,checks,ncheck,counts+1,counts+2,min_theta,means);
}

/* Compile-time degree specialization preserves the same formulas. */
double dg_speed1(const double *a,int nx,int ny,int degree,double g,const double *checks,int ncheck) {
 return speed_impl(a,nx,ny,1,g,checks,ncheck);
}
int dg_step1(const double *a,double *out,double *work,int nx,int ny,int degree,double g,int use_lf,
 const double *vb,const double *vx,const double *vy,const double *fb,const double *fx,const double *fy,
 const double *checks,int ncheck,double dt,double tvb,int limiting,double floor,long long *counts,double *min_theta) {
 return step_impl(a,out,work,nx,ny,1,g,use_lf,vb,vx,vy,fb,fx,fy,checks,ncheck,dt,tvb,limiting,floor,counts,min_theta);
}

/* Compile-time degree specialization preserves the same formulas. */
double dg_speed2(const double *a,int nx,int ny,int degree,double g,const double *checks,int ncheck) {
 return speed_impl(a,nx,ny,2,g,checks,ncheck);
}
int dg_step2(const double *a,double *out,double *work,int nx,int ny,int degree,double g,int use_lf,
 const double *vb,const double *vx,const double *vy,const double *fb,const double *fx,const double *fy,
 const double *checks,int ncheck,double dt,double tvb,int limiting,double floor,long long *counts,double *min_theta) {
 return step_impl(a,out,work,nx,ny,2,g,use_lf,vb,vx,vy,fb,fx,fy,checks,ncheck,dt,tvb,limiting,floor,counts,min_theta);
}

int dg_stabilize(double *a,int nx,int ny,int degree,double g,double tvb,int limiting,double floor,
 const double *checks,int ncheck,long long *limited,long long *scaled,double *min_theta,double *means) {
 return stabilize(a,nx,ny,degree,g,tvb,limiting,floor,checks,ncheck,limited,scaled,min_theta,means);
}
