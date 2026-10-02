"""TVB minmod slope limiter (Cockburn & Shu) for modal Legendre coefficients."""

import numpy as np


def minmod(*args):
    """Componentwise minmod: smallest magnitude if all args share a sign, else 0."""
    a = np.stack(args)
    same_sign = np.all(a > 0, axis=0) | np.all(a < 0, axis=0)
    return np.where(same_sign, np.sign(a[0]) * np.min(np.abs(a), axis=0), 0.0)


class TVBLimiter:
    """Generalized slope limiter with the TVB correction.

    For cell k with mean ubar_k, let
        du_R = u_h(x_{k+1/2}^-) - ubar_k,   du_L = ubar_k - u_h(x_{k-1/2}^+),
        D+ = ubar_{k+1} - ubar_k,           D- = ubar_k - ubar_{k-1}.
    The cell is *troubled* if the modified minmod
        m~(d, D+, D-) = d                if |d| <= M h^2
                        minmod(d, D+, D-) otherwise
    changes du_R or du_L.  A troubled cell keeps its mean, gets the linear
    slope (m~(du_R) + m~(du_L)) / 2 and loses all modes above degree 1.
    Untroubled cells are left untouched, so smooth regions keep full order.

    The cell mean is never changed, so the limiter preserves the total mass.
    Units: M multiplies h^2 (h = cell width in x), so M ~ |u_xx| scale.
    """

    def __init__(self, M=1.0):
        self.M = M

    def __call__(self, c, h):
        if c.shape[1] < 2:
            return c, np.zeros(c.shape[0], dtype=bool)
        m = np.arange(c.shape[1])
        ubar = c[:, 0]
        du_R = c[:, 1:].sum(axis=1)                       # sum_m>=1 c_m P_m(+1)
        du_L = -(c[:, 1:] * (-1.0) ** m[1:]).sum(axis=1)  # ubar - u(-1)
        Dp = np.roll(ubar, -1) - ubar
        Dm = ubar - np.roll(ubar, 1)

        tol = self.M * h * h
        mR = np.where(np.abs(du_R) <= tol, du_R, minmod(du_R, Dp, Dm))
        mL = np.where(np.abs(du_L) <= tol, du_L, minmod(du_L, Dp, Dm))
        # Relative tolerance avoids flagging cells over round-off differences.
        scale = 1e-12 * (np.abs(ubar) + 1.0)
        troubled = (np.abs(mR - du_R) > scale) | (np.abs(mL - du_L) > scale)

        if np.any(troubled):
            c = c.copy()
            c[troubled, 1] = 0.5 * (mR[troubled] + mL[troubled])
            c[troubled, 2:] = 0.0
        return c, troubled
