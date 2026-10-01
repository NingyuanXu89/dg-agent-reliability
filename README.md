# DG Burgers: solver, convergence, and shock movie

A from-scratch numerical PDE exercise for an agentic coding workshop. The solver uses **modal Discontinuous Galerkin (DG)**, a local Lax–Friedrichs interface flux, and explicitly implemented SSP Runge–Kutta time stepping. No PDE solver or finite-element library is used.

## Problem and expected behavior

We solve the nondimensional, periodic, inviscid Burgers equation

$$
u_t + \partial_x(u^2/2)=0,\qquad x\in[0,2\pi],\qquad u(x,0)=1+\sin x.
$$

Characteristics satisfy $x=\xi+t[1+\sin\xi]$. Their first crossing occurs at **$t_s=1$**. After that, the entropy solution contains a moving shock at $x_s=(\pi+t)\bmod2\pi$, traveling with speed 1. The default movie ends at $t=1.5$.

The solver supports degrees **1, 2, and 3** on a uniform periodic mesh. The independent reference solution is specific to this initial wave; it is not a general reference for arbitrary initial data passed to the Python API.

## Run

Dependencies: Python 3.10+, NumPy, SciPy, Matplotlib, and Pillow. FFmpeg with the `libx264` encoder is needed for MP4 output; FFprobe verifies its metadata.

From the repository root, using an environment with these packages:

```sh
python -m unittest discover -s tests -v
python -m dg_burgers demo
```

On the workshop machine, the existing environment can be used without changing it:

```sh
/Users/ningyuanxu/mamba/envs/nlevp-py/bin/python -m unittest discover -s tests -v
/Users/ningyuanxu/mamba/envs/nlevp-py/bin/python -m dg_burgers demo
```

For a separate installation, create a virtual environment and install `requirements.txt`. Output and Matplotlib caches are not committed.

Individual commands:

```sh
python -m dg_burgers solve --cells 128 --degree 2 --final-time 1.5
python -m dg_burgers convergence
python -m dg_burgers movie --cells 128 --degree 2 --frames 151 --fps 20
python -m dg_burgers solve --no-limiter --final-time 0.5
```

`demo` performs both convergence studies, checks numerical acceptance criteria, solves the default evolution, and creates plots, MP4, and GIF. `solve` writes snapshots, diagnostics, and plots. `movie` also exports the animation. Use `--output` to select the output directory; the default is `outputs/`.

Other evolution options are `--cfl` (default 0.15) and `--tvb-m` (default 1). An unlimited solution is intended for smooth pre-shock experiments and can become unstable after shock formation.

## DG formulation

On cell $K$, map $x$ to $\eta\in[-1,1]$ and write

$$
u_h(x,t)|_K=\sum_{j=0}^p U_{K,j}(t)P_j(\eta).
$$

Legendre orthogonality gives diagonal physical mass entries $M_j=h/(2j+1)$. Multiplying the conservation law by $P_j$ and integrating by parts yields

$$
M_j\dot U_{K,j}
=\int_{-1}^1 f(u_h)P_j'(\eta)\,d\eta
-\widehat f_{K+1/2}P_j(1)
+\widehat f_{K-1/2}P_j(-1),\qquad f(u)=u^2/2.
$$

The interface flux is shared by the two adjacent cells:

$$
\widehat f(u_L,u_R)=\frac{f(u_L)+f(u_R)}2
-\frac{\max(|u_L|,|u_R|)}2(u_R-u_L).
$$

Consequently, the cell-mean balance telescopes across the periodic mesh. Projection and nonlinear volume terms use $2p+3$ Gauss–Legendre points, sufficient to integrate the polynomial volume terms without aliasing.

The three SSPRK stages are implemented directly:

$$
U^{(1)}=U^n+\Delta t R(U^n),\qquad
U^{(2)}=\tfrac34U^n+\tfrac14[U^{(1)}+\Delta t R(U^{(1)})],
$$

$$
U^{n+1}=\tfrac13U^n+\tfrac23[U^{(2)}+\Delta t R(U^{(2)})].
$$

The adaptive CFL bound is $\Delta t\leq\mathrm{CFL}\,h/[(2p+1)\max|u_h|]$. Speeds include exact extrema of the degree-1 through degree-3 cell polynomials, not just nodal samples. Steps end exactly at requested snapshot times. Identically zero states are handled without division by zero.

### Shock limiting

The TVB detector compares both endpoint deviations from the cell mean with neighboring cell-mean differences. Deviations below $Mh^2$ are exempt. In flagged cells, the linear coefficient is minmod-limited and higher modes are removed. The cell mean stays unchanged. This limiter is applied after each RK stage.

Limiting trades local high-order accuracy for controlled shock oscillations; it is disabled in smooth convergence studies. It does not force exact interface continuity. The implementation does not claim a fully discrete entropy inequality or a universal maximum principle.

## Independent reference and diagnosis

In moving coordinates $y=(x-t)\bmod2\pi$, the reference inverts $y=\xi+t\sin\xi$ using bracketed SciPy root finding. For $t\leq1$ the solution is single-valued. For $t>1$, the shock foot $q$ solves $q+t\sin q=\pi$ on the branch below $\arccos(-1/t)$. Surviving characteristics have $\xi\in[0,q]$ or $\xi\in[2\pi-q,2\pi]$. The exact shock point is assigned $u=1$; this arbitrary point value does not affect integral errors.

Smooth studies use $t=0.5$, degrees 1–3, and 16, 32, 64, 128 cells. Expected asymptotic spatial order is $p+1$. They use CFL 0.05 with an additional $h^{\max(0,(p+1)/3-1)}$ time-step factor. Each run is compared to one with half-sized steps and refined further until the temporal difference is below 10% of the spatial error. The accepted solution is the finer run.

Post-shock studies use $t=1.5$, degree 2, and 64, 128, 256 cells. Two mesh families place the final shock inside a cell or on a cell face. A periodic mesh-origin shift produces the aligned family; the initial physical wave is unchanged. This alignment applies only at final time: the shock crosses interfaces during evolution. Error quadrature splits at the physical shock. Smooth high-order rates are not required after shock formation.

Diagnostics include:

- Integrated $L^1$ and $L^2$ errors against the independent entropy solution.
- **Sampled** $L^\infty$: maximum error at the split-cell quadrature nodes, not a rigorous supremum. It need not decrease across a moving discontinuity.
- Mass drift relative to the projected initial state; expected initial mass is $2\pi$.
- Quadratic entropy $\frac12\int u_h^2dx$; expected initial value is $3\pi/2$. It is conserved by the smooth exact solution and dissipated after shock formation. Numerical behavior is measured rather than guaranteed.
- Exact polynomial minimum and maximum, largest interface jump, and fraction of limited cell stages.

Acceptance requires decreasing smooth $L^2$ errors, finest-grid rates within 0.4 of $p+1$, mass drift below $10^{-10}$, controlled temporal differences, and decreasing post-shock $L^1$ errors for both mesh families. The default shock runs also check finite solutions and extrema within $[-0.05,2.05]$; this is an empirical acceptance tolerance.

## Generated outputs

All generated artifacts live in the ignored `outputs/` directory:

| Artifact | Contents |
|---|---|
| `smooth_convergence.csv`, `smooth_convergence.png` | Errors, rates, and temporal checks for each degree and mesh |
| `postshock_convergence.csv`, `postshock_convergence.png` | Aligned and unaligned shock refinement |
| `acceptance.json`, `report.md` | Machine-readable checks and a numerical readout |
| `solution/config.json`, `solution/snapshots.npz` | Configuration, timing, grid, times, modal coefficients |
| `solution/diagnostics.csv` | Time histories of errors, conservation, entropy, and limiting |
| `solution/diagnostics.png`, `solution/errors_and_bounds.png` | Diagnostic plots |
| `solution/solution_snapshots.png` | Initial, shock-formation, and final states |
| `solution/burgers_shock.mp4`, `solution/burgers_shock.gif` | 151-frame default evolution at 20 fps |
| `solution/frame_t*.png`, `solution/movie_metadata.json` | Representative frames and verified MP4 metadata |

Each convergence run also saves its configuration, initial/final snapshots, and diagnostics. Numerical cell curves and post-shock reference branches are separated in plots to avoid drawing artificial connections across jumps.

## Workshop reliability experiments

Repeat the same numerical task from clean starting states; paraphrase its requirements; introduce documented human edits; compare claimed correctness with the independent tests. Keep initial data, mesh, polynomial degree, time-step settings, and the evaluator fixed when comparing runs. Record both success in at least one attempt and success in every attempt.

## References

The numerical code here is original and uses only basic numerical/plotting packages. Background references:

- [Cockburn and Shu: TVB Runge–Kutta DG framework](https://doi.org/10.1090/S0025-5718-1989-0983311-4)
- [DG advection lecture](https://www.geophysik.uni-muenchen.de/~igel/Lectures/Book/SS19/08_DiscontinuousGalerkinMethod/dg.pdf)
