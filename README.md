# From-scratch DG: Burgers and 2D compressible hydrodynamics

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


## 2D ideal compressible Euler solver

The separate `dg_euler` package solves the nondimensional ideal-gas Euler system on a periodic unit square:

$$
U_t+F(U)_x+G(U)_y=0,\qquad U=(\rho,m_x,m_y,E),
$$
$$
P=(\gamma-1)\left(E-\frac{m_x^2+m_y^2}{2\rho}\right),\quad
F=(m_x,m_xv_x+P,m_yv_x,(E+P)v_x),
$$
$$
G=(m_y,m_xv_y,m_yv_y+P,(E+P)v_y).
$$

Coefficients have shape `(ny,nx,mode_y,mode_x,4)`; mode `(0,0)` stores each conservative cell mean. The polynomial space is tensor-product Legendre of degree 1 or 2 in each coordinate. On a cell of widths $\Delta x,\Delta y$, the diagonal mass entries are $\Delta x\Delta y/[(2i+1)(2j+1)]$. Integration by parts gives volume terms $\int F\partial_x\phi+G\partial_y\phi$ minus the outward numerical-flux boundary integral. Each periodic face uses one shared flux, so cell-mean residuals telescope for mass, both momenta, and energy.

Volume and face integrals use `p+2` Gauss–Legendre points per coordinate. This is overintegration; rational Euler fluxes are not integrated exactly. HLLC includes tangential momentum and uses outer wave bounds from both states. Degenerate or inadmissible intermediate states, including nonpositive contact pressure, fall back to local Lax–Friedrichs; fallback counts are recorded. See [Athena's Riemann solver guidance](https://princetonuniversity.github.io/Athena-Cversion/AthenaDocsUGRiemann.html). The alternative `--flux llf` is available.

The unsplit residual advances with explicitly implemented SSPRK3. The time step is

$$
\Delta t\le\frac{\mathrm{CFL}}{(2p+1)\max[(|v_x|+c)/\Delta x+(|v_y|+c)/\Delta y]},\qquad c=\sqrt{\gamma P/\rho}.
$$

Speeds are checked at volume and face quadrature nodes, with additional Lobatto sampling. Output times truncate the last step. Default CFL is 0.15.

### Stabilization

After each RK stage, directional TVB minmod detects troubled cells from face-averaged endpoint deviations and differences of neighboring conservative means. The TVB exemption is $Mh^2$ times a local component scale `max(1,abs(neighboring means))`, with default $M=50$. A flagged cell retains minmod-limited linear x/y modes and discards mixed and higher modes. Every conservative mean is preserved.

Conservative positivity scaling then contracts nonconstant modes toward the unchanged mean. Density is scaled analytically; pressure uses bisection. The optional native kernel first uses conservative component bounds based on $|P_iP_j|\le1$ to skip nodal reconstruction in cells whose full polynomial is already safely admissible; otherwise it performs the same nodal checks and scaling. Density and pressure floors are $10^{-12}$, sampled at volume/face Gauss nodes and tensor-product Gauss–Lobatto nodes. An inadmissible mean is never repaired by clipping. Failed stages retry the previous valid state with a halved step, at most eight retries. A terminal failure writes `failure_state.npz` and `failure.json` when an output directory is supplied. These checks demonstrate sampled positivity and empirical stability; no global positivity or discrete entropy theorem is asserted.

### Tanh Kelvin–Helmholtz setup

This is a tanh variant of the classic double-shear test, with an exactly periodic profile:

$$
S(y)=\tfrac12\left[1+\tanh\left(\frac{-\cos(2\pi y)}{2\pi a}\right)\right],\qquad a=0.025,
$$
$$
\rho=1+S,\quad v_x=0.5-S,\quad v_y=0.01\sin(4\pi x),\quad P=2.5,\quad\gamma=5/3.
$$

Interfaces are centered at $y=0.25,0.75$; locally $a$ is the tanh scale. Total energy is constructed from these primitive fields. The main run is 64² cells with degree 2 through $t=3$; comparison runs use 32² and 128² cells with identical physical parameters.

```sh
python -m dg_euler khi --nx 64 --degree 2 --final-time 3 --output outputs/khi/n64
python -m dg_euler validate
python -m dg_euler movie --output outputs/khi/n64
python -m dg_euler demo
```

`demo` generates the smooth convergence and Sod validations, three KHI resolutions, the 64² half-step comparison through $t=1$, a numerical report, plots, MP4, and GIF. Completed run directories are reused. For a fresh run or different parameters, use a new `--output` directory. The 128² DG run is computationally substantial. The CLI optionally compiles the included from-scratch C kernels with an existing C compiler and loads them using standard-library `ctypes`; it installs no packages and changes no environment. A missing compiler or unavailable local kernel falls back to NumPy. Set `DG_EULER_NATIVE=0` to select NumPy explicitly. Native/NumPy agreement is tested, including stabilization. Compiled files live under ignored `outputs/khi/native/`. The Python API uses a previously built kernel when available; `simulate(..., accelerated=False)` selects NumPy. `simulate(..., resume=True)` continues from selected modal checkpoints when the configuration and output schedule match. Options include `--nx`, `--ny`, `--degree`, `--gamma`, `--final-time`, `--cfl`, `--tvb`, `--layer-width`, `--perturbation`, and `--flux`.

Python API:

```python
from dg_euler import EulerConfig, EulerDG, simulate
config = EulerConfig(nx=32, ny=32, final_time=0.1)
result = simulate(config, output="outputs/khi/short", movie_fields=True)
modal_state = result["coefficients"]
```

### Diagnostics and outputs

All generated files are ignored beneath `outputs/khi/`:

- `report.md`, `acceptance.json`, `khi_summary.csv`, `resolution_differences.csv`, `time_step_sensitivity.json`, and `diagnostics.png` contain measured results.
- Each resolution directory contains `config.json`, `diagnostics.json`/CSV, conservative cell means at output times, selected modal checkpoints, and the final modal state. Full modal snapshots are not accumulated in memory.
- The main `n64/frames/` directory stores 151 reconstructed density/vorticity fields, from $t=0$ to 3. `khi.mp4` and `khi.gif` use 20 fps (7.55 s); representative frame PNGs and `movie_metadata.json` support inspection.
- `validation/` contains smooth errors and refinement rates, half-step temporal discrepancies, and x/y Sod-strip errors against an independently implemented exact 1D Euler Riemann solution. Smooth tests advect density $1+0.2\sin[2\pi(x+y-0.5t)]$ with velocity `(0.3,0.2)` and pressure 1 through $t=0.1$, with TVB disabled. Finest-grid L² rates must be within 0.4 of $p+1$. Sod comparisons integrate the central interval `[0.25,0.75]` at $t=0.1$, before periodic-edge waves enter it.

Conservation drift divides each total difference by `max(1,abs(initial_total))`; acceptance requires its maximum below $10^{-9}$. The seeded amplitude is

$$
A_2(t)=\frac{2|\int \rho v_y e^{-4\pi i x}\,dx\,dy|}{\int \rho\,dx\,dy}.
$$

The report fits log-amplitude over $t\in[0.2,0.6]$ and compares full amplitude histories. This interval can contain an initial transient; its fitted slope is not automatically a linear instability growth rate. Transverse kinetic energy, sampled minimum density/pressure, cumulative limiter cell-stage counts, positivity scaling, HLLC fallbacks, and rejected steps are also recorded. Fine-grid density means are conservatively restricted before resolution comparisons.

Vorticity uses analytical within-cell modal derivatives and the quotient rule to convert conservative fields to velocity derivatives, $\omega=\partial_x v_y-\partial_y v_x$. Contributions at interface derivative jumps are omitted. The movie uses one density color range and one symmetric logarithmic vorticity scale derived from all frames, so changing colors do not masquerade as physical growth.

The 128² run is a comparison, not an exact solution. Without specified physical viscosity/diffusion, late-time inviscid small-scale structure is not claimed to be converged; see [McNally, Lyra & Passy (2012)](https://arxiv.org/abs/1111.1764). Resolution and time-step differences measure sensitivity of this particular numerical experiment.
