"""
Discontinuous Galerkin solver for the 1D inviscid Burgers equation
    ∂u/∂t + ∂(u²/2)/∂x = 0

Built entirely from scratch — no DG libraries.

Mathematical structure
----------------------
1.  MESH          : uniform partition of [x_L, x_R] into N elements K_e
2.  LOCAL SPACE   : polynomials of degree ≤ k on each element,
                    represented in the Legendre modal basis {P_0, P_1, …, P_k}
3.  WEAK FORM     : integrate-by-parts on each element K_e, giving
                       M_e dû_e/dt = V(û_e)  -  F_right P(+1)  +  F_left P(-1)
                    where M_e is the (diagonal) local mass matrix,
                    V is the volume integral, F_right/left are numerical fluxes
4.  NUMERICAL FLUX: Lax-Friedrichs  or  Godunov (exact for Burgers)
5.  TIME STEPPING : explicit SSP-RK3 (Shu–Osher)
6.  LIMITER       : TVB-corrected minmod slope limiter on the k=1 coefficient

Reference elements
------------------
Physical element  K_e = [x_{e-1/2}, x_{e+1/2}],  centre x_e,  width h_e
Reference element K̂   = [-1, 1],  coordinate ξ
Map:  x = x_e + (h_e/2) ξ   →   dx = (h_e/2) dξ
Gradient: ∂/∂x = (2/h_e) ∂/∂ξ

Jacobians cancel in the volume integral:
    ∫_K f(u) ∂P_i/∂x dx  =  ∫_{-1}^{1} f(u) P'_i(ξ) dξ     (no h factor!)

Mass matrix (diagonal for Legendre basis):
    M_{ij} = (h_e/2) ∫_{-1}^{1} P_i P_j dξ = h_e/(2j+1) δ_{ij}
    → M_inv_{jj} = (2j+1)/h_e
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# 1.  LEGENDRE POLYNOMIALS AND DERIVATIVES
# =============================================================================

def legendre(n: int, xi: np.ndarray) -> np.ndarray:
    """Evaluate Legendre polynomial P_n at xi via the three-term recurrence.

    P_0 = 1,  P_1 = ξ,  (n+1)P_{n+1} = (2n+1)ξ P_n - n P_{n-1}
    """
    xi = np.asarray(xi, dtype=float)
    if n == 0:
        return np.ones_like(xi)
    if n == 1:
        return xi.copy()
    P_prev, P_curr = np.ones_like(xi), xi.copy()
    for m in range(1, n):
        P_next = ((2*m + 1) * xi * P_curr - m * P_prev) / (m + 1)
        P_prev, P_curr = P_curr, P_next
    return P_curr


def legendre_deriv(n: int, xi: np.ndarray) -> np.ndarray:
    """Derivative P'_n(ξ) via recurrence (1-ξ²)P'_n = n(P_{n-1} - ξ P_n).

    Endpoints ξ = ±1 are handled analytically:
        P'_n(+1) = n(n+1)/2,   P'_n(-1) = (-1)^{n+1} n(n+1)/2
    """
    xi = np.asarray(xi, dtype=float)
    if n == 0:
        return np.zeros_like(xi)
    if n == 1:
        return np.ones_like(xi)
    Pn   = legendre(n,   xi)
    Pnm1 = legendre(n-1, xi)
    interior = np.abs(xi) < 1.0 - 1e-14
    dP = np.empty_like(xi)
    dP[ interior] = n * (Pnm1[interior] - xi[interior] * Pn[interior]) \
                      / (1.0 - xi[interior]**2)
    dP[~interior & (xi > 0)] = n * (n+1) / 2.0
    dP[~interior & (xi < 0)] = (-1)**(n+1) * n * (n+1) / 2.0
    return dP


def gauss_legendre(n_pts: int):
    """Gauss-Legendre quadrature nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(n_pts)


# =============================================================================
# 2.  DISCONTINUOUS GALERKIN DISCRETISATION CLASS
# =============================================================================

class DG1D:
    """1D DG spatial discretisation for conservation laws.

    Parameters
    ----------
    x_left, x_right : float
        Domain boundaries.
    N_elem : int
        Number of elements.
    k_order : int
        Polynomial degree (k=1 → piecewise linear, k=2 → quadratic, …).
    bc : 'periodic' or 'outflow'
        Boundary conditions.
    """

    def __init__(self, x_left, x_right, N_elem, k_order, bc='periodic'):
        self.xl  = x_left
        self.xr  = x_right
        self.N   = N_elem
        self.k   = k_order
        self.Np  = k_order + 1          # basis functions per element
        self.bc  = bc

        # ------------------------------------------------------------------
        # Uniform mesh
        # ------------------------------------------------------------------
        self.x_edges   = np.linspace(x_left, x_right, N_elem + 1)
        self.h         = self.x_edges[1:] - self.x_edges[:-1]      # widths
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

        # ------------------------------------------------------------------
        # Gauss-Legendre quadrature
        # We need to integrate f(u) P'_i exactly; f(u) ~ degree 2k,
        # P'_i ~ degree k-1 → integrand degree 3k-1.
        # Gauss with n pts integrates degree 2n-1 exactly → n = ceil(3k/2)+1.
        # ------------------------------------------------------------------
        n_q = k_order + 2
        self.xi_q, self.w_q = gauss_legendre(n_q)

        # ------------------------------------------------------------------
        # Precompute basis tables (shape: [Np, n_q])
        # phi_q[j, q]  = P_j(ξ_q)   — used for reconstruction at quad pts
        # dphi_q[j, q] = P'_j(ξ_q)  — used for volume integral
        # ------------------------------------------------------------------
        self.phi_q  = np.array([legendre(j,       self.xi_q) for j in range(self.Np)])
        self.dphi_q = np.array([legendre_deriv(j, self.xi_q) for j in range(self.Np)])

        # Basis at element endpoints (for face terms)
        # P_j(+1) = 1 for all j;  P_j(-1) = (-1)^j
        self.phi_right = np.array([legendre(j, np.array([1.0]))[0]  for j in range(self.Np)])
        self.phi_left  = np.array([legendre(j, np.array([-1.0]))[0] for j in range(self.Np)])

        # ------------------------------------------------------------------
        # Local mass matrix (diagonal): M_inv[e, j] = (2j+1)/h[e]
        # ------------------------------------------------------------------
        j_arr = np.arange(self.Np, dtype=float)
        # M[e,j] = h[e]/(2j+1)  →  M_inv[e,j] = (2j+1)/h[e]
        self.M_inv = (2.0 * j_arr + 1.0) / self.h[:, None]   # shape (N, Np)

    # ------------------------------------------------------------------
    # L2 projection of initial condition
    # ------------------------------------------------------------------
    def project(self, u_func) -> np.ndarray:
        """L2-project u_func onto the DG space.

        û_j^e = (2j+1)/2 * ∫_{-1}^{1} u(x(ξ)) P_j(ξ) dξ    (by orthogonality)
        """
        u_hat = np.zeros((self.N, self.Np))
        j_coeff = (2.0 * np.arange(self.Np) + 1.0) / 2.0
        for e in range(self.N):
            x_q = self.x_centers[e] + 0.5 * self.h[e] * self.xi_q
            u_q = u_func(x_q)
            for j in range(self.Np):
                u_hat[e, j] = j_coeff[j] * np.dot(self.w_q, u_q * self.phi_q[j])
        return u_hat

    # ------------------------------------------------------------------
    # Reconstruct solution for plotting
    # ------------------------------------------------------------------
    def reconstruct(self, u_hat, n_pts=6) -> tuple:
        """Return (x, u) arrays for plotting, n_pts sub-points per element."""
        xi_p = np.linspace(-1.0, 1.0, n_pts)
        phi_p = np.array([legendre(j, xi_p) for j in range(self.Np)])   # (Np, n_pts)
        u_rec = u_hat @ phi_p        # (N, n_pts)
        x_rec = self.x_centers[:, None] + 0.5 * self.h[:, None] * xi_p  # (N, n_pts)
        return x_rec.ravel(), u_rec.ravel()

    # ------------------------------------------------------------------
    # Interior and boundary traces
    # ------------------------------------------------------------------
    def traces(self, u_hat):
        """Return left and right boundary values of u on each element.

        u_R[e] = u|_e(ξ=+1)  — right face of element e (interior trace)
        u_L[e] = u|_e(ξ=-1)  — left  face of element e (interior trace)
        """
        u_R = u_hat @ self.phi_right   # (N,)
        u_L = u_hat @ self.phi_left    # (N,)
        return u_L, u_R

    # ------------------------------------------------------------------
    # Numerical fluxes
    # ------------------------------------------------------------------
    @staticmethod
    def flux_lax_friedrichs(u_m, u_p, f, df):
        """Local Lax-Friedrichs (Rusanov) flux.

        h(u⁻, u⁺) = ½[f(u⁻)+f(u⁺)] - ½ α (u⁺-u⁻),   α = max|f'|
        """
        alpha = np.maximum(np.abs(df(u_m)), np.abs(df(u_p)))
        return 0.5 * (f(u_m) + f(u_p)) - 0.5 * alpha * (u_p - u_m)

    @staticmethod
    def flux_godunov_burgers(u_m, u_p):
        """Exact Godunov flux for Burgers  f(u) = u²/2.

        The Riemann problem has:
          • shock    (u⁻ > u⁺):  upwind on shock speed s = (u⁻+u⁺)/2
          • rarefaction (u⁻ < u⁺):  characteristics diverge; use min/max
        """
        f_m = 0.5 * u_m**2
        f_p = 0.5 * u_p**2
        # Shock branch: pick upwind side based on Rankine-Hugoniot speed
        s         = 0.5 * (u_m + u_p)
        f_shock   = np.where(s >= 0.0, f_m, f_p)
        # Rarefaction branch
        f_rarefac = np.where(u_m >= 0.0, f_m,
                    np.where(u_p <= 0.0, f_p, 0.0))   # sonic point → 0
        return np.where(u_m > u_p, f_shock, f_rarefac)

    # ------------------------------------------------------------------
    # RHS:  M_e^{-1} [ volume  −  face_right  +  face_left ]
    # ------------------------------------------------------------------
    def rhs(self, u_hat, f_func, df_func, flux='LF') -> np.ndarray:
        """Compute dû/dt for all elements.

        Weak form on element K_e, for each basis function P_i (i=0…k):

            (h_e/(2i+1)) dû_i/dt
              =  ∫_{-1}^{1} f(u(ξ)) P'_i(ξ) dξ       [volume — Jacobians cancel]
                 − f̂_R · P_i(+1)                       [right face]
                 + f̂_L · P_i(−1)                       [left  face]
        """
        N, Np = self.N, self.Np

        # ---- volume integral ------------------------------------------------
        # u at quad pts:  u_q[e,q] = Σ_j û_j P_j(ξ_q)
        u_q = u_hat @ self.phi_q      # (N, n_q)
        f_q = f_func(u_q)             # (N, n_q)
        # res_vol[e,i] = Σ_q w_q f_q[e,q] P'_i(ξ_q)
        res = f_q @ (self.w_q[:, None] * self.dphi_q.T)   # (N, Np)

        # ---- assemble face states -------------------------------------------
        u_L, u_R = self.traces(u_hat)

        u_m = np.empty(N + 1)   # left  state at face i+1/2
        u_p = np.empty(N + 1)   # right state at face i+1/2

        u_m[1:N] = u_R[:N-1]
        u_p[1:N] = u_L[1:N]

        if self.bc == 'periodic':
            u_m[0] = u_R[N-1];  u_p[0] = u_L[0]
            u_m[N] = u_R[N-1];  u_p[N] = u_L[0]
        else:   # outflow / zero-gradient
            u_m[0] = u_L[0];    u_p[0] = u_L[0]
            u_m[N] = u_R[N-1];  u_p[N] = u_R[N-1]

        # ---- numerical flux at all N+1 faces --------------------------------
        if flux == 'LF':
            fhat = self.flux_lax_friedrichs(u_m, u_p, f_func, df_func)
        else:
            fhat = self.flux_godunov_burgers(u_m, u_p)

        # ---- subtract face contributions ------------------------------------
        # right face of element e is face e+1:  subtract fhat[e+1] * P_i(+1)
        # left  face of element e is face e  :  add    fhat[e]   * P_i(-1)
        res -= fhat[1:N+1, None] * self.phi_right[None, :]
        res += fhat[0:N,   None] * self.phi_left [None, :]

        return self.M_inv * res   # (N, Np)

    # ------------------------------------------------------------------
    # Minmod slope limiter (TVB-corrected, works for k ≥ 1)
    # ------------------------------------------------------------------
    @staticmethod
    def _minmod3(a, b, c):
        """Componentwise minmod of three arrays."""
        s = np.sign(a)
        return np.where(
            (s == np.sign(b)) & (s == np.sign(c)),
            s * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))),
            0.0,
        )

    def limit(self, u_hat, M_tvb=0.0) -> np.ndarray:
        """TVB-corrected minmod slope limiter.

        For the k=1 (and higher) case:
          • cell average û_0 is never touched
          • slope coefficient û_1 is limited to
              û_1^lim = minmod(û_1,  Δ+,  Δ-)
            where Δ± are differences of neighbouring cell averages
          • higher coefficients (k>1) are zeroed in troubled cells

        The TVB parameter M_tvb suppresses limiting in smooth extrema:
        cells where |û_1| ≤ M_tvb h² skip the limiter entirely.
        """
        if self.k == 0:
            return u_hat

        u_lim = u_hat.copy()
        u0 = u_hat[:, 0]   # cell averages

        if self.bc == 'periodic':
            u0_ext = np.concatenate([[u0[-1]], u0, [u0[0]]])
        else:
            u0_ext = np.concatenate([[u0[0]], u0, [u0[-1]]])

        delta_p = u0_ext[2:]    - u0_ext[1:-1]   # forward  difference
        delta_m = u0_ext[1:-1]  - u0_ext[:-2]    # backward difference

        slope     = u_hat[:, 1]
        slope_lim = self._minmod3(slope, delta_p, delta_m)

        # TVB: skip limiting where the slope is naturally small
        troubled = np.abs(slope - slope_lim) > M_tvb * self.h**2

        u_lim[troubled, 1]  = slope_lim[troubled]
        if self.k > 1:
            u_lim[troubled, 2:] = 0.0

        return u_lim


# =============================================================================
# 3.  TIME INTEGRATOR: SSP-RK3 (Shu–Osher, 1988)
# =============================================================================

def ssp_rk3(u_hat, dt, dg, f_func, df_func,
            flux='LF', use_limiter=True, M_tvb=0.0) -> np.ndarray:
    """Third-order Strong Stability Preserving Runge-Kutta.

    u¹ =        u^n + Δt L(u^n)
    u² = 3/4 u^n + 1/4 (u¹ + Δt L(u¹))
    u^{n+1} = 1/3 u^n + 2/3 (u² + Δt L(u²))

    Each stage optionally applies the slope limiter so that SSP
    properties (total-variation stability) are preserved.
    """
    def L(u):
        return dg.rhs(u, f_func, df_func, flux=flux)

    def lim(u):
        return dg.limit(u, M_tvb) if use_limiter else u

    u1 = lim(u_hat + dt * L(u_hat))
    u2 = lim(0.75 * u_hat + 0.25 * (u1 + dt * L(u1)))
    return   lim((1/3) * u_hat + (2/3) * (u2 + dt * L(u2)))


def compute_dt(dg, u_hat, CFL=0.1) -> float:
    """CFL-based time step.

    For degree-k DG:  Δt ≤ CFL * h_min / (λ_max * (2k+1))
    The (2k+1) factor reflects the tighter stability bound of high-order DG
    compared to first-order finite volume.
    """
    u_L, u_R = dg.traces(u_hat)
    lam_max = max(np.max(np.abs(u_L)), np.max(np.abs(u_R)), 1e-12)
    return CFL * np.min(dg.h) / (lam_max * (2 * dg.k + 1))


# =============================================================================
# 4.  TEST PROBLEMS FOR BURGERS' EQUATION
# =============================================================================

def burgers_f(u):  return 0.5 * u**2
def burgers_df(u): return u            # characteristic speed


def exact_burgers_sine(x, t, n_iter=50):
    """Exact solution of  ∂u/∂t + u∂u/∂x = 0,  u_0 = sin(πx)  on [0,2] periodic.

    Solved by Newton iteration on the implicit equation  u = sin(π(x - u·t)).
    Valid for t < 1/π ≈ 0.318 (before shock formation).
    """
    u = np.sin(np.pi * x)
    for _ in range(n_iter):
        residual  = u - np.sin(np.pi * (x - u * t))
        jacobian  = 1.0 + t * np.pi * np.cos(np.pi * (x - u * t))
        u -= residual / jacobian
    return u


def l2_error(dg, u_hat, u_exact, n_sub=10):
    """Compute L2 error against an exact solution function."""
    xi_sub = np.linspace(-1.0, 1.0, n_sub)
    phi_sub = np.array([legendre(j, xi_sub) for j in range(dg.Np)])
    err2 = 0.0
    for e in range(dg.N):
        x_sub  = dg.x_centers[e] + 0.5 * dg.h[e] * xi_sub
        u_num  = (u_hat[e] @ phi_sub)
        u_ref  = u_exact(x_sub)
        err2  += np.trapezoid((u_num - u_ref)**2, x_sub)
    return np.sqrt(err2)


# =============================================================================
# 5.  SIMULATION RUNNER
# =============================================================================

def run(problem='sine', N=64, k=1, t_final=0.8,
        CFL=0.1, flux='LF', use_limiter=True, M_tvb=0.0,
        snap_times=(0.0, 0.15, 0.30, 0.60, 0.80)):
    """Run DG simulation and return (dg, snapshots) where each snapshot is (t, u_hat)."""
    if problem == 'sine':
        xl, xr, bc = 0.0, 2.0, 'periodic'
        u0 = lambda x: np.sin(np.pi * x)
    elif problem == 'tophat':
        xl, xr, bc = 0.0, 1.0, 'periodic'
        u0 = lambda x: np.where(np.abs(x - 0.5) < 0.25, 1.0, 0.0)
    elif problem == 'shock':           # stationary shock at x=0
        xl, xr, bc = -1.0, 1.0, 'outflow'
        u0 = lambda x: np.where(x < 0.0, 1.0, -1.0)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    dg    = DG1D(xl, xr, N, k, bc=bc)
    u_hat = dg.project(u0)

    snaps      = [(0.0, u_hat.copy())]
    snap_set   = sorted(s for s in snap_times if s > 0.0)
    snap_idx   = 0
    t          = 0.0

    while t < t_final - 1e-14:
        dt = min(compute_dt(dg, u_hat, CFL), t_final - t)
        u_hat = ssp_rk3(u_hat, dt, dg, burgers_f, burgers_df,
                        flux=flux, use_limiter=use_limiter, M_tvb=M_tvb)
        t += dt
        if snap_idx < len(snap_set) and t >= snap_set[snap_idx] - 1e-10:
            snaps.append((t, u_hat.copy()))
            snap_idx += 1

    return dg, snaps


# =============================================================================
# 6.  CONVERGENCE STUDY  (pre-shock smooth regime)
# =============================================================================

def convergence_study(k_list=(1, 2, 3), N_list=(16, 32, 64, 128),
                      t_test=0.20, CFL=0.05):
    """Measure L2 convergence rate for the sine-wave problem before shock.

    Expected rate for degree-k DG: O(h^{k+1}).
    """
    print("\n=== Convergence study: Burgers sine wave, t = {:.2f} ===\n".format(t_test))
    results = {}
    for k in k_list:
        print(f"  Polynomial degree k = {k}  (expected rate ≈ {k+1})")
        print(f"  {'N':>6}  {'h':>10}  {'L2 error':>14}  {'rate':>8}")
        errors = []
        prev_err = None
        for N in N_list:
            dg    = DG1D(0.0, 2.0, N, k, bc='periodic')
            u_hat = dg.project(lambda x: np.sin(np.pi * x))
            t     = 0.0
            while t < t_test - 1e-14:
                dt    = min(compute_dt(dg, u_hat, CFL), t_test - t)
                u_hat = ssp_rk3(u_hat, dt, dg, burgers_f, burgers_df,
                                flux='LF', use_limiter=False)
                t += dt

            err  = l2_error(dg, u_hat, lambda x: exact_burgers_sine(x, t_test))
            rate = np.log2(prev_err / err) if prev_err else float('nan')
            print(f"  {N:>6}  {2.0/N:>10.4f}  {err:>14.4e}  {rate:>8.2f}")
            errors.append(err)
            prev_err = err
        results[k] = (N_list, errors)
        print()
    return results


# =============================================================================
# 7.  PLOTTING
# =============================================================================

def plot_simulation(dg, snaps, problem='sine', save_path=None):
    """Plot solution snapshots on a single figure."""
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(snaps)))
    fig, ax = plt.subplots(figsize=(9, 5))

    for (t, u_hat), c in zip(snaps, colors):
        x, u = dg.reconstruct(u_hat, n_pts=8)
        ax.plot(x, u, color=c, lw=1.6, label=f't = {t:.2f}')

    titles = {'sine': 'Burgers: sine wave → shock',
              'tophat': 'Burgers: top-hat (steepening + rarefaction)',
              'shock': 'Burgers: stationary shock'}
    ax.set_title(titles.get(problem, problem), fontsize=13)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('u(x, t)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_convergence(results, save_path=None):
    """Log-log convergence plot for multiple polynomial degrees."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ['o', 's', '^', 'D']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (k, (N_list, errors)) in enumerate(sorted(results.items())):
        h_list = 2.0 / np.array(N_list)
        ax.loglog(h_list, errors, marker=markers[idx], color=colors[idx],
                  lw=1.8, ms=7, label=f'k={k}  (meas. ≈ {np.polyfit(np.log(h_list[-3:]),np.log(errors[-3:]),1)[0]:.1f})')
        # Reference slope
        ref_h   = h_list[-2:]
        ref_err = errors[-2] * (ref_h / ref_h[0])**(k+1)
        ax.loglog(ref_h, ref_err, '--', color=colors[idx], lw=1, alpha=0.6)

    ax.set_xlabel('h  (element width)', fontsize=11)
    ax.set_ylabel('L² error', fontsize=11)
    ax.set_title('DG convergence for Burgers equation (pre-shock)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_flux_comparison(N=64, k=1, save_path=None):
    """Compare LF vs Godunov flux on the sine-wave problem at t=0.50 (post-shock)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    t_plot = 0.50

    for ax, flux_name in zip(axes, ['LF', 'Godunov']):
        dg, snaps = run(problem='sine', N=N, k=k, t_final=t_plot,
                        flux=flux_name, use_limiter=True, M_tvb=0.0,
                        snap_times=(t_plot,))
        t_snap, u_hat = snaps[-1]
        x, u = dg.reconstruct(u_hat, n_pts=8)
        ax.plot(x, u, lw=1.8, color='#1f77b4')
        ax.set_title(f'Flux: {flux_name}  (t = {t_snap:.2f})', fontsize=12)
        ax.set_xlabel('x', fontsize=11)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel('u(x, t)', fontsize=11)
    fig.suptitle(f'LF vs Godunov flux comparison  (N={N}, k={k})', fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# =============================================================================
# 8.  MAIN
# =============================================================================

if __name__ == '__main__':
    import os
    out = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out, exist_ok=True)

    # ---- Problem 1: sine wave, k=1, limiter on  (shock forms at t≈0.32) ----
    print("Running sine wave problem (k=1, N=80, LF flux)…")
    dg, snaps = run(problem='sine', N=80, k=1, t_final=0.80, flux='LF',
                    use_limiter=True, snap_times=(0.0, 0.15, 0.30, 0.50, 0.80))
    fig1 = plot_simulation(dg, snaps, problem='sine',
                           save_path=os.path.join(out, 'sine_k1.png'))
    print(f"  → saved sine_k1.png  ({len(snaps)} snapshots)")

    # ---- Problem 2: sine wave, k=2, limiter on  ----
    print("Running sine wave problem (k=2, N=60, LF flux)…")
    dg, snaps = run(problem='sine', N=60, k=2, t_final=0.80, flux='LF',
                    use_limiter=True, snap_times=(0.0, 0.15, 0.30, 0.50, 0.80))
    fig2 = plot_simulation(dg, snaps, problem='sine',
                           save_path=os.path.join(out, 'sine_k2.png'))
    print(f"  → saved sine_k2.png")

    # ---- Problem 3: top-hat  ----
    print("Running top-hat problem (k=1, N=80)…")
    dg, snaps = run(problem='tophat', N=80, k=1, t_final=0.60, flux='Godunov',
                    use_limiter=True, snap_times=(0.0, 0.10, 0.20, 0.40, 0.60))
    fig3 = plot_simulation(dg, snaps, problem='tophat',
                           save_path=os.path.join(out, 'tophat_k1.png'))
    print(f"  → saved tophat_k1.png")

    # ---- Problem 4: LF vs Godunov comparison  ----
    print("Running LF vs Godunov flux comparison…")
    fig4 = plot_flux_comparison(N=64, k=1,
                                save_path=os.path.join(out, 'flux_comparison.png'))
    print(f"  → saved flux_comparison.png")

    # ---- Convergence study  ----
    print("Running convergence study…")
    conv_results = convergence_study(k_list=(1, 2, 3),
                                     N_list=(16, 32, 64, 128),
                                     t_test=0.20)
    fig5 = plot_convergence(conv_results,
                            save_path=os.path.join(out, 'convergence.png'))
    print(f"  → saved convergence.png")

    plt.close('all')
    print("\nAll done. Outputs in burgers_dg/outputs/")
