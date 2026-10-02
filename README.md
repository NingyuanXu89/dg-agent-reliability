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

## Project status

The first experiment is the nonlinear version of this exercise: a DG solver for the inviscid **Burgers equation**, written from scratch (described below). The linear-advection exercise above and the multi-run reliability harness have not been added yet.

## Experiment: DG for inviscid Burgers through shock formation

$$
\frac{\partial u}{\partial t}+\frac{\partial}{\partial x}\left(\frac{u^2}{2}\right)=0,\qquad u_0(x)=0.5+\sin x,\quad x\in[0,2\pi)\ \text{periodic}.
$$

Characteristics, $u=u_0(\xi)$ with $\xi+t\,u_0(\xi)=x$, give the exact solution. The gradient blows up as $\max|u_x|=1/(1-t)$, so a shock forms at $t_s=1$. Because $u-0.5$ is odd about the steepest point, the shock then travels at the mean speed, $x_s(t)=\pi+0.5t$, which is also the Rankine–Hugoniot speed. Taking the appropriate characteristic branch on each side of $x_s$ gives an **exact entropy solution after the shock too** ([burgers_dg/exact.py](burgers_dg/exact.py)). Every error below is measured against this independent reference. The nonzero mean also checks transport direction and periodic wrapping.

### Method (all code in [burgers_dg/](burgers_dg/), NumPy only)

- Modal Legendre basis of degree $p$ on uniform cells, which makes the mass matrix diagonal. The volume integral uses Gauss–Legendre quadrature exact for $f(u_h)P_m'$, so there is no aliasing.
- The numerical flux is the exact Godunov flux for $u^2/2$, including the sonic-point case. Rusanov is also available.
- Time stepping is SSP-RK3 (Shu–Osher) with $\Delta t=0.3\,h/((2p+1)\max|u|)$. In the convergence study $p=3$ uses $\Delta t\propto h^{4/3}$ so the temporal error stays below the $O(h^4)$ spatial error. A check on the finest meshes with $\Delta t/4$ changes the $L^2$ error by at most $1.2\times10^{-6}$ relative.
- The shock-capturing limiter is a TVB minmod limiter (Cockburn–Shu, $M=1$), applied after every RK stage. It changes only troubled cells (reducing them to limited linears) and never changes a cell mean, so mass is conserved exactly.

### Results (regenerate with `experiments/run_all.py`, about 30 s)

**Smooth convergence** at $t=0.5$, with no limiter. The full table, including $L^1$ and $L^\infty$, is in [results/convergence.md](results/convergence.md). The ideal rate is $p+1$.

| p | K = 32 L² error | K = 256 L² error | final L² rate | final L∞ rate |
|---|---|---|---|---|
| 1 | 6.08e-03 | 1.06e-04 | 1.96 | 1.98 |
| 2 | 2.67e-04 | 6.13e-07 | 2.96 | 2.92 |
| 3 | 1.56e-05 | 4.50e-09 | 3.94 | 3.94 |

**Through the shock** (p = 2, K = 128, TVB limiter, run to $t=1.3$):

- Relative mass drift stays at or below $2\times10^{-14}$ (round-off).
- Energy $\tfrac12\int u^2$ stays constant to plotting accuracy while the solution is smooth, then decays after $t_s$. At $t=1.3$ the DG value is 2.2489 and the exact entropy solution gives 2.2657, so DG dissipates slightly more, mostly in the shock cell.
- No new extrema form: the point values stay within $[-0.5, 1.5]$ to within $2\times10^{-6}$. The largest cell-mean jump is $0.24h$ from the exact $x_s(1.3)$.
- On this mesh, shock formation is detected at t ≈ 0.93–0.95. The DG gradient falls below 90% of the exact $1/(1-t)$ blow-up at $t=0.934$, and the limiter first activates at $t=0.954$ (exact $t_s=1$). Both indicators depend on resolution: they move closer to 1 as K increases (the limiter fires at 0.68 / 0.92 / 0.95 / 0.97 / 0.98 for K = 32 to 512 with p = 2).
- Post-shock $L^1$ error at $t=1.3$ is about first order ($L^1$ rates 0.96–1.73 for K = 32 to 512, p = 1, 2), which is expected for discontinuous solutions.

| file | content |
|---|---|
| [results/burgers_shock.mp4](results/burgers_shock.mp4), [.gif](results/burgers_shock.gif) | animation from $t=0$ to $1.3$: DG vs exact, limited cells, gradient blow-up |
| [results/snapshots.png](results/snapshots.png) | the same at $t = 0, 0.5, 0.8, 0.95, 1, 1.3$ |
| [results/convergence.png](results/convergence.png) | $L^1/L^2/L^\infty$ error vs $h$ with slope-$(p+1)$ references |
| [results/error_profile.png](results/error_profile.png) | pointwise error, which peaks at the steepening front |
| [results/error_vs_time.png](results/error_vs_time.png) | error growth as $t\to t_s$ |
| [results/gradient_blowup.png](results/gradient_blowup.png) | numerical max $\lvert u_x\rvert$ vs exact $1/(1-t)$, with shock-detection times |
| [results/conservation.png](results/conservation.png) | mass drift and DG vs exact energy |
| `results/*.json`, `results/convergence.csv` | machine-readable numbers |

**Limitations.** The TVB parameter $M$ is set by hand. Smooth extrema are protected only to the extent $M h^2$ allows. Post-shock accuracy is first order in $L^1$, as for any shock-capturing scheme. The temporal-order safeguard for $p=3$ is a step-size restriction, not a fourth-order integrator. Only uniform meshes and the scalar case are implemented.

### How to run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r pyproject.toml --extra dev
.venv/bin/python -m pytest -q           # 36 tests: basis, flux, exact solution, solver checks
.venv/bin/python experiments/run_all.py # convergence study + diagnostics + animation → results/
```

The tests cover preservation of a constant state, mass conservation (limited and unlimited), energy, transport direction, periodic wrap, observed order, exact landing on output times, limiter behaviour, and shock location. They also cross-check the exact solution against SciPy's `brentq`. MP4 output needs `ffmpeg` on `PATH`; the GIF is always written.

## References

- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
- [MFEM PDE examples](https://mfem.org/examples/)
