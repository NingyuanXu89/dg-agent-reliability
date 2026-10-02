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

A from-scratch DG solver for periodic 1D **inviscid Burgers' equation** $u_t + (u^2/2)_x = 0$ is in `dg_burgers/` (numpy only; no PDE/DG libraries). The experiment harness for the reliability questions above has not been added yet.

- **Method:** modal Legendre basis of degree p, exact Godunov flux (Rusanov optional), SSP-RK3, and an optional TVB minmod slope limiter (Cockburn–Shu).
- **Test problem:** $u_0 = 0.5 + \sin x$ on $[0, 2\pi)$. The breaking time is $t_s = 1$, and the shock forms at $x = \pi + 0.5$, then moves at speed 0.5.
- **Exact references:** characteristics give the exact solution for $t < t_s$. For any $t$, the exact entropy solution comes from symmetry in the frame moving at speed 0.5.

| Path | Contents |
|---|---|
| `dg_burgers/basis.py` | Legendre basis, Gauss quadrature |
| `dg_burgers/solver.py` | `DGBurgers`: projection, DG right-hand side, fluxes, limiter, SSP-RK3 |
| `dg_burgers/exact.py` | characteristic and entropy exact solutions, breaking time |
| `dg_burgers/diagnostics.py` | error norms, observed orders, mass and energy |
| `scripts/run_convergence.py` | convergence tables and error-diagnosis plots → `results/` |
| `scripts/make_movie.py` | movie of the solution steepening into a shock → `results/burgers_shock.mp4` |
| `tests/test_dg_burgers.py` | pytest suite |

## How to run

```bash
uv venv .venv && uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_convergence.py
.venv/bin/python scripts/make_movie.py
```

The movie needs `ffmpeg` on PATH. Without it, the script falls back to an animated GIF.

## References

- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
- [MFEM PDE examples](https://mfem.org/examples/)
