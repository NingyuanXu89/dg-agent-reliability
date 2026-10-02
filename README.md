# DG Agent Reliability

A workshop repository for exploring the reliability of agentic coding through short, well-scoped numerical PDE tasks, with applications to computational astrophysics.

The initial focus is the **Discontinuous Galerkin (DG) method**: local polynomial approximations coupled across mesh cells by numerical fluxes.

## Proposed first exercise

Implement a small DG solver for periodic one-dimensional advection:

$$
\frac{\partial u}{\partial t}+a\frac{\partial u}{\partial x}=0.
$$

For constant velocity, the exact solution is the initial profile translated by $at$, with periodic wrapping. This provides an independent reference for checking the numerical solution.

Use synthetic inputs and keep the task small enough to repeat from a clean starting state.

## Reliability questions

- **Repeatability:** Does the same prompt produce correct results across fresh runs?
- **Sensitivity:** Do equivalent prompt phrasings change correctness?
- **Collaboration:** Does the agent correctly incorporate documented human edits?
- **Verification:** Do its tests and final report accurately establish what works?

Numerical checks should cover conservation of the cell-integrated quantity, preservation of a constant solution, correct transport direction, periodic boundaries, and convergence toward the exact solution.

Record success for each task and run, together with regressions, runtime, cost where available, and human interventions. Distinguish success in at least one attempt from success in every attempt.

## Burgers' equation experiment

A self-contained DG experiment for the periodic inviscid Burgers equation

$$
\frac{\partial u}{\partial t}+\frac{\partial}{\partial x}\left(\frac{u^2}{2}\right)=0,\qquad u_0(x)=\tfrac12+\sin(2\pi x),\quad x\in[0,1).
$$

The sine profile steepens and breaks at $t_s = 1/(2\pi)\approx 0.159$, forming a shock at $x_s = \tfrac12 + t_s/2$. The nonzero mean makes the profile drift to the right, which also checks the transport direction.

### Method (written from scratch with NumPy only)

| Component | Choice | File |
|---|---|---|
| Basis | Modal Legendre polynomials $P_0,\dots,P_p$ per cell, diagonal mass matrix | [burgers_dg/basis.py](burgers_dg/basis.py) |
| Volume integral | Gauss–Legendre rule exact for the degree-$(3p-1)$ integrand $f(u_h)P_k'$ | [burgers_dg/solver.py](burgers_dg/solver.py) |
| Numerical flux | Exact Godunov flux (default) or local Lax–Friedrichs | [burgers_dg/solver.py](burgers_dg/solver.py) |
| Time stepping | SSP-RK3, $\Delta t = \mathrm{CFL}\,h/((2p+1)\max\lvert u\rvert)$ | [burgers_dg/solver.py](burgers_dg/solver.py) |
| Shock capturing | Cockburn–Shu TVB minmod limiter (shock run only) | [burgers_dg/solver.py](burgers_dg/solver.py) |
| Reference solution | Characteristics (Newton–bisection) before $t_s$; Hopf–Lax entropy solution after | [burgers_dg/exact.py](burgers_dg/exact.py) |

Exact quadrature together with an E-flux gives the Jiang–Shu cell entropy inequality, so $\int u_h^2/2\,dx$ cannot increase. The test suite checks this for random DG states.

### Running

```bash
uv venv .venv && uv pip install --python .venv/bin/python numpy matplotlib pillow pytest
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python -m experiments.convergence
```

```bash
.venv/bin/python -m experiments.animate
```

The two experiment scripts write to [results/](results/). Each takes `--help` for options such as degrees, meshes, flux and final time. The convergence study takes a few seconds and the animation about 30 s.

### Results

**Convergence** ([results/convergence.md](results/convergence.md), [convergence.png](results/convergence.png), [error_profiles.png](results/error_profiles.png)). P0–P3 run on N = 10…320 cells to $t = 0.5\,t_s$ (smooth solution, no limiter). Errors are measured against the exact characteristic solution. Observed orders on the finest pair of meshes:

| p | L1 | L2 | L∞ | expected |
|---|----|----|----|----------|
| 0 | 0.99 | 0.98 | 0.98 | 1 |
| 1 | 1.98 | 1.97 | 1.98 | 2 |
| 2 | 2.99 | 2.96 | 2.92 | 3 |
| 3 | 3.97 | 3.95 | 3.94 | 4 |

For P3 the time step is capped at $\propto h^{4/3}$ so the $O(\Delta t^3)$ RK3 error stays below the $O(h^4)$ spatial error. Halving $\Delta t$ on the finest mesh changes the L2 error by less than $10^{-3}$ (relative) for every $p$, so the reported rates are spatial. The mass drift is at most $1.6\times10^{-14}$, and $\int u^2/2$ is lower at the final time than at the start in every run, by $1.5\times10^{-12}$ for P3 at N = 320. That loss is numerical dissipation, since the exact smooth solution conserves it.

**Shock formation** ([results/burgers_shock.gif](results/burgers_shock.gif), [.mp4](results/burgers_shock.mp4), [shock_snapshots.png](results/shock_snapshots.png), [animation_summary.json](results/animation_summary.json)). This run uses P2, N = 100 and the TVB limiter with $M = \tfrac23\max\lvert u_0''\rvert$. It goes from $t = 0$ to $1.5\,t_s$, a short way past breaking so the formed shock is visible. The animation shows:
- $u_h$ against the exact entropy solution, with the limited cells marked;
- $\max\lvert\partial_x u_h\rvert$ against the exact blow-up $2\pi/(1-2\pi t)$;
- the L1 error and the number of limited cells over time.

Extrapolating $1/\max\lvert\partial_x u_h\rvert$, which is exactly linear in $t$ before breaking, gives $t_s \approx 0.15876$. The exact value is $0.15915$, so the relative error is $0.25\%$. After breaking, the shock is held within about two cells with no Gibbs oscillations, mass stays conserved to $5\times10^{-15}$, and the L1 error at $1.5\,t_s$ is $3.5\times10^{-3}$.

Known limitations:
- The TVB limiter flags a few smooth cells from about $0.55\,t_s$ onward, before the front is under-resolved. This slightly raises the pre-shock L1 error in the animation run (still below $5\times10^{-5}$ at $0.9\,t_s$). The convergence study does not use the limiter.
- Past the shock the scheme is first-order accurate near the discontinuity, as expected for limited DG.

## Project status

The Burgers DG solver, exact reference solutions, tests (`tests/`), convergence study and shock animation are in place. The linear-advection exercise and the multi-run reliability harness described above have not been added yet.

## References

- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
- [MFEM PDE examples](https://mfem.org/examples/)
