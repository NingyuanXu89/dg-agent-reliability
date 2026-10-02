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

A DG solver for the **inviscid Burgers equation** has been added. It was written from scratch with numpy only, and covers the run through shock formation. The advection exercise above and the experiment harness have not been added yet.

### Burgers DG solver (`dgburgers/`)

The solver handles $u_t + (u^2/2)_x = 0$ on a periodic interval.

- **Discretisation:** modal Legendre basis of degree $p$. The volume integral uses Gauss quadrature that is exact for $f(u_h)P_l'$, which gives the semi-discrete scheme the cell entropy inequality.
- **Fluxes:** Godunov (exact Riemann) or local Lax–Friedrichs.
- **Limiter:** optional TVB minmod slope limiter (Cockburn–Shu).
- **Time stepping:** SSP-RK3 or SSPRK(10,4), with the limiter applied after every stage.
- **Exact reference:** `dgburgers/exact.py` gives the exact entropy solution for $u_0 = c + A\sin x$, valid both before and after the shock. The shock forms at $t_b = 1/A$, at $x = \pi + c\,t_b$, and then moves at speed $c$.

All results use $u_0 = 0.5 + \sin x$ on $[0, 2\pi)$, so $t_b = 1$.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy matplotlib pytest
.venv/bin/python -m pytest              # 81 tests, < 1 s
.venv/bin/python scripts/convergence.py # ~15 s  -> results/convergence.{md,csv}, convergence_*.png
.venv/bin/python scripts/diagnostics.py # ~8 s   -> results/diagnostics_*.png, diagnostics_summary.json
.venv/bin/python scripts/make_movie.py  # ~15 s  -> results/burgers_shock.mp4, burgers_snapshots.png
```

### Key results

The full tables are in `results/convergence.md`.

- **Smooth regime ($t = 0.5$):** with no limiter, the L1, L2 and L∞ errors converge at the optimal order $p+1$ for $p = 0$–$3$, with both fluxes. At $N = 640$ the observed orders are 1.00, 1.98, 2.98 and 3.97. Halving the time step changes the error by $\le 10^{-5}$ relative, so time-stepping error does not pollute these rates.
- **Near breaking ($t = 0.9$):** the steepest slope is $\max|u_x| = 10$. Rates are erratic until the front is resolved, then reach about $p+1$ on the finest meshes.
- **After the shock ($t = 2$), TVB-limited:**
  - The global L1 error is first order (least-squares fit 1.1). The order between successive meshes oscillates because the error depends on where the shock falls within its cell.
  - Away from the shock the order is $p+1$ again (fits 2.2 for $p = 1$ and 3.4 for $p = 2$).
- **Diagnostics over time ($p = 2$, $N = 64$):**
  - Mass is conserved to about $2\times10^{-14}$.
  - The energy $\int u^2/2$ never increases, follows the exact dissipation after $t_b$, and is conserved before it.
  - The unlimited solution overshoots by about 1.2 (Gibbs oscillations). The TVB limiter reduces this to about $2\times10^{-4}$.
  - The limiter first activates at $t = 0.94$, and the computed steepest slope departs from the exact $-1/(1-t)$ at $t \approx 0.93$. This is the mesh's resolution limit just before $t_b$.
- **Movie:** `results/burgers_shock.mp4` shows DG ($p = 2$, $N = 48$) with and without the limiter against the exact solution, from $t = 0$ to $1.4$. It overlays the multivalued characteristic curve and tracks the pointwise error and the steepest slope.

## References

- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
- [MFEM PDE examples](https://mfem.org/examples/)
