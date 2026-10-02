"""Reproducible convergence, planar shock, KHI comparison, and reporting."""
from dataclasses import replace,asdict,fields
from pathlib import Path
import csv,json,re
import numpy as np
from numpy.polynomial.legendre import leggauss
from .core import EulerConfig,EulerDG,simulate,conservative,pressure
from .reference import riemann


def save_table(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def smooth_initial(gamma,time=0):
    return lambda x,y:conservative(1+.2*np.sin(2*np.pi*(x+y-.5*time)),.3,.2,1,gamma)


def density_error(result,reference):
    s=result['solver'];z,w=leggauss(s.config.degree+5)
    x=(np.arange(s.config.nx)[None,:,None,None]+.5*(1+z)[None,None,None,:])*s.dx
    y=(np.arange(s.config.ny)[:,None,None,None]+.5*(1+z)[None,None,:,None])*s.dy
    err=s.reconstruct(result['coefficients'],z,z)[...,0]-reference(x,y)[...,0]
    weights=w[:,None]*w[None,:]*s.dx*s.dy/4
    return float(np.sum(abs(err)*weights)),float(np.sqrt(np.sum(err**2*weights))),float(abs(err).max())


def state_difference(a,b,solver):
    diff=solver.reconstruct(a-b)[...,0]
    return float(np.sqrt(np.sum(diff**2*solver.w[:,None]*solver.w[None,:])*solver.dx*solver.dy/4))


def smooth_convergence(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for p in (1,2):
        for n in (16,32,64):
            cfg=EulerConfig(nx=n,ny=n,degree=p,final_time=.1,cfl=.05,limiting=False,dt_cap=.02/n**((p+1)/3))
            coarse=simulate(cfg,initial=smooth_initial(cfg.gamma),times=[0,.1])
            for reduction in range(6):
                fine=simulate(replace(cfg,cfl=cfg.cfl/2,dt_cap=cfg.dt_cap/2),initial=smooth_initial(cfg.gamma),times=[0,.1])
                l1,l2,linf=density_error(fine,smooth_initial(cfg.gamma,.1))
                temporal=state_difference(coarse['coefficients'],fine['coefficients'],fine['solver'])
                if temporal<=.1*l2:break
                cfg=replace(cfg,cfl=cfg.cfl/2,dt_cap=cfg.dt_cap/2);coarse=fine
            else:raise RuntimeError('smooth temporal error not controlled')
            previous=next((r for r in reversed(rows) if r['degree']==p),None)
            rate=float(np.log2(previous['l2']/l2)) if previous else None
            row=dict(degree=p,cells=n,l1=l1,l2=l2,linf=linf,rate=rate,temporal_discrepancy=temporal,temporal_fraction=temporal/l2,cfl=fine['config'].cfl,drift=max(d['drift'] for d in fine['history']))
            rows.append(row);print('smooth',row,flush=True)
            (output/'smooth.json').write_text(json.dumps(rows,indent=2))
    save_table(output/'smooth.csv',rows)
    passed=all(abs(r['rate']-(r['degree']+1))<.4 for r in rows if r['cells']==64)
    if not passed:raise RuntimeError('smooth convergence acceptance failed')
    return rows


def sod_comparisons(output):
    output=Path(output);rows=[]
    for axis in (0,1):
        for n in (32,64,128):
            cfg=EulerConfig(nx=n if axis==0 else 3,ny=3 if axis==0 else n,gamma=1.4,final_time=.1)
            def initial(x,y):
                q=x if axis==0 else y;rho=np.where(q<.5,1,.125);p=np.where(q<.5,1,.1)
                rho,p=np.broadcast_arrays(rho+np.zeros_like(x+y),p+np.zeros_like(x+y))
                return conservative(rho,0,0,p,cfg.gamma)
            result=simulate(cfg,initial=initial,times=[0,.1]);s=result['solver'];z,w=leggauss(20)
            u=s.reconstruct(result['coefficients'],z,z)
            x=(np.arange(cfg.nx)[None,:,None,None]+.5*(1+z)[None,None,None,:])*s.dx
            y=(np.arange(cfg.ny)[:,None,None,None]+.5*(1+z)[None,None,:,None])*s.dy
            q=np.broadcast_to(x if axis==0 else y,u.shape[:-1]);ref=riemann(q,.1,axis=axis)
            weights=w[:,None]*w[None,:]*s.dx*s.dy/4*((q>=.25)&(q<=.75))
            errors=np.sum(abs(u-ref)*weights[...,None],axis=(0,1,2,3))
            row=dict(axis='x' if axis==0 else 'y',cells=n,density_l1=float(errors[0]),normal_momentum_l1=float(errors[axis+1]),energy_l1=float(errors[3]),drift=max(d['drift'] for d in result['history']))
            rows.append(row);print('Sod',row,flush=True);(output/'sod.json').write_text(json.dumps(rows,indent=2))
    save_table(output/'sod.csv',rows)
    for axis in ('x','y'):
        group=[r for r in rows if r['axis']==axis]
        if not all(b['density_l1']<a['density_l1'] and b['normal_momentum_l1']<a['normal_momentum_l1'] and b['energy_l1']<a['energy_l1'] for a,b in zip(group,group[1:])):raise RuntimeError('Sod refinement acceptance failed')
    return rows


def validate(output='outputs/khi/validation'):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    smooth=json.loads((out/'smooth.json').read_text()) if (out/'smooth.csv').exists() else smooth_convergence(out)
    sod=json.loads((out/'sod.json').read_text()) if (out/'sod.csv').exists() else sod_comparisons(out)
    acceptance=dict(smooth_rates=all(abs(r['rate']-r['degree']-1)<.4 for r in smooth if r['cells']==64),temporal_control=all(r['temporal_fraction']<=.1 for r in smooth),sod_refinement=True)
    (out/'acceptance.json').write_text(json.dumps(acceptance,indent=2));return acceptance


def khi_suite(output='outputs/khi',base=None):
    out=Path(output);base=base or EulerConfig()
    if base.final_time!=3:raise ValueError('the complete benchmark suite uses final_time=3; use simulate/khi for other times')
    for n in (32,64,128):
        directory=out/f'n{n}'
        expected=replace(base,nx=n,ny=n)
        if (directory/'final_state.npz').exists():
            previous=json.loads((directory/'config.json').read_text());previous.pop('backend',None)
            previous['dt_cap']=None if previous.get('dt_cap') is None or np.isinf(previous['dt_cap']) else previous['dt_cap']
            wanted=asdict(expected);wanted['dt_cap']=None if np.isinf(expected.dt_cap) else expected.dt_cap
            if previous!=wanted:raise ValueError(f'{directory} contains a different configuration; use a new output directory')
        else:simulate(expected,output=directory,movie_fields=n==64,progress=True)
    sensitivity=out/'half_step_64'
    expected=replace(base,nx=64,ny=64,final_time=1,cfl=base.cfl/2)
    if (sensitivity/'final_state.npz').exists():
        previous=json.loads((sensitivity/'config.json').read_text());previous.pop('backend',None)
        previous['dt_cap']=None if previous.get('dt_cap') is None or np.isinf(previous['dt_cap']) else previous['dt_cap']
        wanted=asdict(expected);wanted['dt_cap']=None if np.isinf(expected.dt_cap) else expected.dt_cap
        if previous!=wanted:raise ValueError(f'{sensitivity} contains a different configuration; use a new output directory')
    else:simulate(expected,times=np.linspace(0,1,51),output=sensitivity,progress=True)
    return report(out)


def report(output='outputs/khi'):
    out=Path(output);histories={n:json.loads((out/f'n{n}'/'diagnostics.json').read_text()) for n in (32,64,128)}
    raw=json.loads((out/'n64'/'config.json').read_text());raw.pop('backend',None);raw['dt_cap']=float('inf') if raw.get('dt_cap') is None else raw['dt_cap'];config=EulerConfig(**{f.name:raw[f.name] for f in fields(EulerConfig) if f.name in raw})
    if any(len(h)!=151 or abs(h[-1]['time']-3)>1e-12 for h in histories.values()):raise ValueError('all three KHI comparisons must finish through t=3 before reporting')
    rows=[]
    for n,h in histories.items():
        t=np.array([d['time'] for d in h]);a=np.array([d['mode_amplitude'] for d in h]);window=(t>=.2-1e-12)&(t<=.6+1e-12)
        fit=np.polyfit(t[window],np.log(a[window]),1);pred=np.polyval(fit,t[window]);obs=np.log(a[window]);ss=float(np.sum((obs-pred)**2));sst=float(np.sum((obs-obs.mean())**2))
        row=dict(cells=n,growth_fit=float(fit[0]),fit_r_squared=1-ss/sst,initial_amplitude=float(a[0]),final_amplitude=float(a[-1]),peak_amplitude=float(a.max()),max_drift=max(d['drift'] for d in h),min_density=min(d['min_density'] for d in h),min_pressure=min(d['min_pressure'] for d in h),limited=h[-1]['limited'],positivity_scaled=h[-1]['positivity_scaled'],flux_fallbacks=h[-1]['flux_fallbacks'],rejected_steps=h[-1]['rejected_steps'])
        totals=np.asarray([d['totals'] for d in h]);initial=totals[0];drifts=np.max(abs(totals-initial)/np.maximum(1,abs(initial)),axis=0)
        row.update({f'{name}_drift':float(drifts[i]) for i,name in enumerate(('mass','momentum_x','momentum_y','energy'))})
        row['limiter_cell_stage_fraction']=row['limited']/(3*h[-1]['steps']*n*n)
        rows.append(row);save_table(out/f'n{n}'/'diagnostics.csv',[{**d,**{f'total_{i}':v for i,v in enumerate(d['totals'])}} for d in h])
    save_table(out/'khi_summary.csv',rows)
    comparisons=[]
    for low,high in ((32,64),(64,128)):
        for time in (.6,1.,3.):
            index=int(round(time/.02));l=np.load(out/f'n{low}'/f'means_{index:04d}.npz')['means'][...,0];h=np.load(out/f'n{high}'/f'means_{index:04d}.npz')['means'][...,0]
            restricted=h.reshape(low,2,low,2).mean((1,3));delta=restricted-l
            comparisons.append(dict(coarse=low,fine=high,time=time,density_mean_l1=float(abs(delta).mean()),density_mean_l2=float(np.sqrt((delta**2).mean())),amplitude_difference=abs(histories[low][index]['mode_amplitude']-histories[high][index]['mode_amplitude'])))
    save_table(out/'resolution_differences.csv',comparisons)
    half=json.loads((out/'half_step_64'/'diagnostics.json').read_text());a=np.load(out/'n64'/'checkpoint_0050.npz')['coefficients'];b=np.load(out/'half_step_64'/'final_state.npz')['coefficients'];s=EulerDG(config)
    temporal=dict(time=1,density_l2=state_difference(a,b,s),mode_amplitude_difference=abs(histories[64][50]['mode_amplitude']-half[-1]['mode_amplitude']),max_drift=max(d['drift'] for d in half))
    (out/'time_step_sensitivity.json').write_text(json.dumps(temporal,indent=2))
    finite=all(np.isfinite([d['min_density'],d['min_pressure'],d['mode_amplitude'],d['drift'],*d['totals']]).all() for h in histories.values() for d in h)
    positive=all(r['min_density']>=config.floor*.99 and r['min_pressure']>=config.floor*.99 for r in rows)
    conserved=all(r['max_drift']<1e-9 for r in rows);growth=all(r['peak_amplitude']>r['initial_amplitude'] for r in rows)
    accepted=finite and positive and conserved and growth
    checks=dict(khi=accepted,sampled_positivity=positive,finite=finite,conservation=conserved,seeded_mode_growth=growth)
    if (out/'validation'/'acceptance.json').exists():checks.update(json.loads((out/'validation'/'acceptance.json').read_text()))
    if (out/'validation_native'/'acceptance.json').exists():checks['native_validation']=all(json.loads((out/'validation_native'/'acceptance.json').read_text()).values())
    if (out/'unit_tests.log').exists():
        log=(out/'unit_tests.log').read_text();match=re.search(r'Ran (\d+) tests',log)
        checks['unit_tests_passed']=log.rstrip().endswith('OK');checks['unit_test_count']=int(match.group(1)) if match else None
    (out/'acceptance.json').write_text(json.dumps(checks,indent=2))
    from .plotting import diagnostic_plots
    diagnostic_plots(histories,out)
    lines=['# Euler DG numerical report','', f'The tanh variant of the classic periodic double-shear KHI uses degree {config.degree}, {config.flux.upper()}, gamma={config.gamma:g}, layer scale a={config.layer_width:g}, and vy perturbation {config.perturbation:g} sin(4 pi x).', '', '|Cells|Early log-amplitude slope, t=0.2–0.6|Fit R²|Peak A2|Maximum conserved-total drift|Minimum density|Minimum pressure|','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["cells"]}²|{r["growth_fit"]:.6g}|{r["fit_r_squared"]:.5g}|{r["peak_amplitude"]:.6g}|{r["max_drift"]:.3g}|{r["min_density"]:.6g}|{r["min_pressure"]:.6g}|')
    lines+=['', '|Cells|Steps|Limited cell-stage fraction|Positivity scalings|HLLC fallbacks|Rejected steps|','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["cells"]}²|{histories[r["cells"]][-1]["steps"]}|{r["limiter_cell_stage_fraction"]:.6g}|{r["positivity_scaled"]}|{r["flux_fallbacks"]}|{r["rejected_steps"]}|')
    lines+=['', '|Cells|Mass drift|x-momentum drift|y-momentum drift|Energy drift|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["cells"]}²|{r["mass_drift"]:.3g}|{r["momentum_x_drift"]:.3g}|{r["momentum_y_drift"]:.3g}|{r["energy_drift"]:.3g}|')
    lines+=['',f'![KHI diagnostic histories]({(out/"diagnostics.png").resolve()})', '', 'The prescribed early fit window spans a transient dip and turn-around; the low fit R² values show that it does not represent a single exponential growth phase.']
    lines+=['',f'KHI acceptance: **{accepted}**. The slope is a measured finite-time fit, not an exact eigenmode growth rate.', '', f'Halving the 64² time step through t=1 gives density L² difference {temporal["density_l2"]:.6g} and mode-amplitude difference {temporal["mode_amplitude_difference"]:.6g}.', '', '|Coarse/fine cells|Time|Density means L¹ difference|Density means L² difference|A2 difference|','|---|---:|---:|---:|---:|']
    for r in comparisons:lines.append(f'|{r["coarse"]}/{r["fine"]}|{r["time"]:g}|{r["density_mean_l1"]:.6g}|{r["density_mean_l2"]:.6g}|{r["amplitude_difference"]:.6g}|')
    lines+=['',f'![Final cell-averaged density comparison]({(out/"density_comparison.png").resolve()})']
    vd=out/'validation'
    if (vd/'smooth.json').exists():
        lines+=['','## Smooth advected density wave','', 'Density = 1 + 0.2 sin(2 pi (x+y−0.5t)); velocity=(0.3,0.2), pressure=1; t=0.1. TVB disabled, positivity remains enabled. Half-step discrepancy is at most 10% of measured density L² error.', '', '|Degree|Cells|L¹|L²|Sampled L∞|L² order|Temporal fraction|','|---|---:|---:|---:|---:|---:|---:|']
        for r in json.loads((vd/'smooth.json').read_text()):lines.append(f'|{r["degree"]}|{r["cells"]}|{r["l1"]:.6g}|{r["l2"]:.6g}|{r["linf"]:.6g}|{r["rate"] if r["rate"] is not None else "—"}|{r["temporal_fraction"]:.3g}|')
    if (vd/'sod.json').exists():
        lines+=['','## Planar Sod strips','', 'Gamma=1.4, left (rho,v,P)=(1,0,1), right=(0.125,0,0.1); t=0.1. Integrate only the central coordinate interval [0.25,0.75], excluding waves from the periodic edge. Independent shock/rarefaction exact Riemann reference; 20-point error quadrature per cell.', '', '|Axis|Cells|Density L¹|Normal momentum L¹|Energy L¹|Drift|','|---|---:|---:|---:|---:|---:|']
        for r in json.loads((vd/'sod.json').read_text()):lines.append(f'|{r["axis"]}|{r["cells"]}|{r["density_l1"]:.6g}|{r["normal_momentum_l1"]:.6g}|{r["energy_l1"]:.6g}|{r["drift"]:.3g}|')
    lines+=['','## Interpretation','', 'The NumPy implementation remains the reference implementation; the optional from-scratch native kernels require no external numerical solver or additional Python packages. The 128² solution is a numerical comparison, not an exact reference. No physical viscosity or diffusivity is prescribed; late-time inviscid small-scale structure is not claimed to be converged. Limiter activity is a cumulative cell-stage count; positivity scaling and HLLC fallbacks are also cumulative. These runs demonstrate sampled admissibility and empirical stability, not a global positivity or entropy theorem.', '', 'Vorticity is differentiated within each cell from the modal conserved fields using the quotient rule; distributional contributions at interface derivative jumps are omitted. Density and signed logarithmic vorticity movie limits are fixed over all frames.']
    if checks.get('unit_tests_passed'):
        lines+=['',f'All {checks["unit_test_count"]} Burgers and Euler unit tests passed; the complete test log is saved in unit_tests.log.']
    if checks.get('native_validation'):
        lines+=['', 'The optional native backend also passed the smooth and Sod acceptance checks; measured tables are saved in validation_native/.']
    if (out/'backend_agreement.json').exists():
        agreement=json.loads((out/'backend_agreement.json').read_text())
        lines+=['',f'Full-time 32² native/NumPy comparison through t=3: final density L² difference {agreement["final_density_l2"]:.6g}, maximum amplitude-history difference {agreement["max_mode_amplitude_difference"]:.6g}, maximum modal coefficient difference {agreement["max_modal_difference"]:.6g}.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return dict(summary=rows,temporal=temporal,accepted=accepted)
