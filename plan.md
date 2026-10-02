# From-scratch DG Burgers solver and diagnostics

## Summary

Build on the local `workshop` branch in `/Users/ningyuanxu/Projects/sandbox/dg-agent-reliability`.

Solve periodic, one-dimensional inviscid Burgers:

\[
u_t+\partial_x(u^2/2)=0,\qquad
x\in[0,2\pi],\qquad u(x,0)=1+\sin x.
\]

Support polynomial degrees 1–3. Produce numerical error and convergence reports, plus a movie through \(t=1.5\). The initial wave forms a shock at \(t=1\); its subsequent position is \(x_s(t)=(\pi+t)\bmod 2\pi\).

## Numerical implementation

- Implement a modal DG method using Legendre polynomials on uniform cells. Write projection, interface traces, volume integration, and the DG residual directly. Use NumPy polynomial utilities and Gauss–Legendre quadrature; use no existing PDE solver.
- Use the conservative weak formulation with diagonal cell mass entries \(h/(2j+1)\), a single shared numerical flux at every interface, and periodic neighbor indexing.
- Use local Lax–Friedrichs flux:
  \[
  \widehat f(u_L,u_R)
  =\tfrac12[f(u_L)+f(u_R)]
  -\tfrac12\max(|u_L|,|u_R|)(u_R-u_L).
  \]
  Use \(2p+3\) quadrature points for projection and nonlinear volume integration.
- Implement three-stage SSP Runge–Kutta directly. Choose adaptive steps from polynomial extrema and
  \[
  \Delta t\leq \frac{\mathrm{CFL}\,h}{(2p+1)\max|u_h|}.
  \]
  Default CFL is 0.15. Clip steps to requested output times and handle the identically zero state explicitly.
- For shock evolution, apply a TVB minmod limiter after each RK stage, with default \(M=1\). Detect troubled cells using both endpoint deviations from the cell mean and neighboring mean differences, with the \(Mh^2\) exemption. In flagged cells, minmod-limit the linear coefficient and remove higher modes while preserving the cell mean. Disable limiting for smooth convergence studies.
- Reject invalid configurations and stop with a diagnostic if states become nonfinite.

## Reference solution and error diagnosis

- Implement an independent characteristic reference with SciPy root finding. In the moving coordinate \(y=(x-t)\bmod2\pi\), solve
  \[
  y=\xi+t\sin\xi,\qquad u=1+\sin\xi.
  \]
  Before shock formation, use the unique root on \([0,2\pi]\).
- After \(t=1\), select the entropy branches. Find the shock foot \(q\) from \(q+t\sin q=\pi\), bracketed below the characteristic turning point. Use \(\xi\in[0,q]\) left of the shock and \(\xi\in[2\pi-q,2\pi]\) right of it. Document the arbitrary point value at the discontinuity.
- Run smooth convergence studies at \(t=0.5\), with \(N=16,32,64,128\) and \(p=1,2,3\). Report integrated \(L^1\), \(L^2\), sampled \(L^\infty\), and observed refinement rates.
- Start convergence runs with CFL 0.05 and a degree-dependent time-step cap. Compare against half-sized steps; reduce further if the temporal discrepancy exceeds 10% of the spatial \(L^2\) error.
- Track mass drift, quadratic entropy, extrema, interface jumps, and limiter activation. Add post-shock \(L^1\) comparisons at \(t=1.5\) on both aligned and unaligned meshes. Split error integration at the exact shock; do not claim smooth-solution convergence rates after shock formation.
- Save configuration, diagnostics, snapshots, CSV tables, and static plots under an ignored `outputs/` directory.

## Movie and usage

- Add a small `dg_burgers` package with commands `solve`, `convergence`, `movie`, and `demo`; `demo` generates the complete default deliverables.
- Default movie: 128 cells, degree 2, 151 frames from \(t=0\) through \(1.5\), at 20 fps.
- Plot the numerical cell polynomials and independent reference, with time, shock time, and shock location annotated. Separate discontinuous curve segments to avoid misleading connecting lines.
- Export H.264 MP4 using the installed FFmpeg, plus a GIF preview. Verify the movie metadata and inspect representative frames before delivery.
- Update the README with the derivation, commands, limiter behavior, output descriptions, and interpretation of diagnostics.

## Validation and defaults

- Test constant preservation, flux consistency, periodic conservation, projection accuracy, limiter mean preservation, and characteristic-reference residuals.
- Exercise fluxes with positive, negative, and transonic states.
- Require mass drift below \(10^{-10}\), decreasing smooth errors, and finest-grid \(L^2\) rates within 0.4 of \(p+1\), once temporal error is controlled.
- Check finite, stable post-shock evolution, decreasing post-shock \(L^1\) error with refinement, and sensitivity to halving the time step. Report entropy and overshoot measurements without asserting an unproved discrete entropy guarantee.
- Use NumPy, SciPy, Matplotlib, Pillow, and standard-library `unittest`. The existing Python 3.11 environment has the numerical and plotting dependencies; leave that environment unchanged.
- Keep implementation and generated deliverables local. Do not commit or push unless requested.
