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

This repository currently contains this README only. Solver code, tests, and an experiment harness have not been added.

## References

- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
- [MFEM PDE examples](https://mfem.org/examples/)
