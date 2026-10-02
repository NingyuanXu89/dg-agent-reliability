"""Strong-stability-preserving Runge-Kutta steppers for dU/dt = L(U).

Each stepper takes the spatial operator ``L``, a limiter ``lim`` (applied after
every stage, as in the RKDG method of Cockburn & Shu), the state ``U`` and the
step size ``dt``.
"""


def ssprk3_step(L, lim, U, dt):
    """Third-order, three-stage SSP-RK (Shu & Osher 1988)."""
    u1 = lim(U + dt * L(U))
    u2 = lim(0.75 * U + 0.25 * (u1 + dt * L(u1)))
    return lim(U / 3.0 + 2.0 / 3.0 * (u2 + dt * L(u2)))


def ssprk104_step(L, lim, U, dt):
    """Fourth-order, ten-stage SSP-RK in low-storage form (Ketcheson 2008)."""
    q1 = U.copy()
    q2 = U.copy()
    for _ in range(5):
        q1 = lim(q1 + dt / 6.0 * L(q1))
    q2 = q2 / 25.0 + 9.0 / 25.0 * q1
    q1 = lim(15.0 * q2 - 5.0 * q1)
    for _ in range(4):
        q1 = lim(q1 + dt / 6.0 * L(q1))
    return lim(q2 + 0.6 * q1 + 0.1 * dt * L(q1))


INTEGRATORS = {
    "ssprk3": ssprk3_step,
    "ssprk104": ssprk104_step,
}

ORDER = {"ssprk3": 3, "ssprk104": 4}
